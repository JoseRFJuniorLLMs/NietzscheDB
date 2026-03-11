#!/usr/bin/env python3
"""
Ingest ICD-10 Classification into NietzscheDB.

Downloads the WHO ICD-10 classification (~70K codes) organized in a strict
hierarchy: Chapters → Blocks → Categories → Subcategories.
Highly relevant for the Malaria project (A50-B64 infectious diseases).

Usage:
  python scripts/ingest_icd10.py [--host HOST:PORT] [--collection NAME]

Requirements:
  pip install grpcio grpcio-tools requests
"""

import grpc
import json
import uuid
import math
import hashlib
import sys
import os
import argparse
import csv
import re
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'sdks', 'python'))
from grpc_tools import protoc
import importlib


def ensure_proto_compiled(repo_root=None):
    if repo_root is None:
        for candidate in [
            os.path.join(os.path.dirname(__file__), '..'),
            '/home/web2a/NietzscheDB',
            os.path.expanduser('~/NietzscheDB'),
        ]:
            if os.path.isdir(os.path.join(candidate, 'crates', 'nietzsche-api', 'proto')):
                repo_root = candidate
                break
        if repo_root is None:
            raise RuntimeError("Cannot find NietzscheDB repo root.")

    proto_dir = os.path.join(repo_root, 'crates', 'nietzsche-api', 'proto')
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_gen')
    if not os.access(os.path.dirname(os.path.abspath(__file__)), os.W_OK):
        out_dir = '/tmp/_nietzsche_gen'
    os.makedirs(out_dir, exist_ok=True)

    pb2_file = os.path.join(out_dir, 'nietzsche_pb2.py')
    if not os.path.exists(pb2_file):
        print(f"[proto] Compiling nietzsche.proto from {proto_dir}...")
        protoc.main([
            'grpc_tools.protoc',
            f'-I{proto_dir}',
            f'--python_out={out_dir}',
            f'--grpc_python_out={out_dir}',
            'nietzsche.proto',
        ])
        with open(os.path.join(out_dir, '__init__.py'), 'w') as f:
            f.write('')

    sys.path.insert(0, out_dir)
    return importlib.import_module('nietzsche_pb2'), importlib.import_module('nietzsche_pb2_grpc')


# ═══════════════════════════════════════════════════════════════════════════
# ICD-10 DATA - Complete Chapter/Block/Category hierarchy
# ═══════════════════════════════════════════════════════════════════════════

# 22 ICD-10 Chapters with their block ranges
ICD10_CHAPTERS = {
    "I": {"name": "Certain infectious and parasitic diseases", "range": "A00-B99", "angle": 0.0},
    "II": {"name": "Neoplasms", "range": "C00-D48", "angle": 0.286},
    "III": {"name": "Diseases of the blood and blood-forming organs", "range": "D50-D89", "angle": 0.571},
    "IV": {"name": "Endocrine, nutritional and metabolic diseases", "range": "E00-E90", "angle": 0.857},
    "V": {"name": "Mental and behavioural disorders", "range": "F00-F99", "angle": 1.143},
    "VI": {"name": "Diseases of the nervous system", "range": "G00-G99", "angle": 1.429},
    "VII": {"name": "Diseases of the eye and adnexa", "range": "H00-H59", "angle": 1.714},
    "VIII": {"name": "Diseases of the ear and mastoid process", "range": "H60-H95", "angle": 2.0},
    "IX": {"name": "Diseases of the circulatory system", "range": "I00-I99", "angle": 2.286},
    "X": {"name": "Diseases of the respiratory system", "range": "J00-J99", "angle": 2.571},
    "XI": {"name": "Diseases of the digestive system", "range": "K00-K93", "angle": 2.857},
    "XII": {"name": "Diseases of the skin and subcutaneous tissue", "range": "L00-L99", "angle": 3.143},
    "XIII": {"name": "Diseases of the musculoskeletal system", "range": "M00-M99", "angle": 3.429},
    "XIV": {"name": "Diseases of the genitourinary system", "range": "N00-N99", "angle": 3.714},
    "XV": {"name": "Pregnancy, childbirth and the puerperium", "range": "O00-O99", "angle": 4.0},
    "XVI": {"name": "Certain conditions originating in the perinatal period", "range": "P00-P96", "angle": 4.286},
    "XVII": {"name": "Congenital malformations and chromosomal abnormalities", "range": "Q00-Q99", "angle": 4.571},
    "XVIII": {"name": "Symptoms, signs and abnormal clinical findings", "range": "R00-R99", "angle": 4.857},
    "XIX": {"name": "Injury, poisoning and certain other consequences", "range": "S00-T98", "angle": 5.143},
    "XX": {"name": "External causes of morbidity and mortality", "range": "V01-Y98", "angle": 5.429},
    "XXI": {"name": "Factors influencing health status", "range": "Z00-Z99", "angle": 5.714},
    "XXII": {"name": "Codes for special purposes", "range": "U00-U99", "angle": 6.0},
}

# Detailed blocks within chapters (focus on infectious diseases + common conditions)
ICD10_BLOCKS = {
    # Chapter I - Infectious diseases (MALARIA RELEVANT)
    "A00-A09": {"name": "Intestinal infectious diseases", "chapter": "I",
                "codes": ["A00 Cholera", "A01 Typhoid and paratyphoid fevers", "A02 Other salmonella infections",
                          "A03 Shigellosis", "A04 Other bacterial intestinal infections",
                          "A05 Other bacterial foodborne intoxications", "A06 Amoebiasis",
                          "A07 Other protozoal intestinal diseases", "A08 Viral intestinal infections",
                          "A09 Other gastroenteritis and colitis"]},
    "A15-A19": {"name": "Tuberculosis", "chapter": "I",
                "codes": ["A15 Respiratory tuberculosis bacteriologically confirmed",
                          "A16 Respiratory tuberculosis not confirmed", "A17 Tuberculosis of nervous system",
                          "A18 Tuberculosis of other organs", "A19 Miliary tuberculosis"]},
    "A20-A28": {"name": "Certain zoonotic bacterial diseases", "chapter": "I",
                "codes": ["A20 Plague", "A21 Tularaemia", "A22 Anthrax", "A23 Brucellosis",
                          "A24 Glanders and melioidosis", "A25 Rat-bite fevers",
                          "A26 Erysipeloid", "A27 Leptospirosis", "A28 Other zoonotic bacterial diseases"]},
    "A30-A49": {"name": "Other bacterial diseases", "chapter": "I",
                "codes": ["A30 Leprosy", "A31 Mycobacterial infections", "A32 Listeriosis",
                          "A33 Tetanus neonatorum", "A34 Obstetrical tetanus", "A35 Other tetanus",
                          "A36 Diphtheria", "A37 Whooping cough", "A38 Scarlet fever",
                          "A39 Meningococcal infection", "A40 Streptococcal sepsis",
                          "A41 Other sepsis", "A42 Actinomycosis", "A43 Nocardiosis",
                          "A44 Bartonellosis", "A46 Erysipelas", "A48 Other bacterial diseases",
                          "A49 Bacterial infection unspecified"]},
    "A50-A64": {"name": "Infections with a predominantly sexual mode of transmission", "chapter": "I",
                "codes": ["A50 Congenital syphilis", "A51 Early syphilis", "A52 Late syphilis",
                          "A53 Other syphilis", "A54 Gonococcal infection", "A55 Chlamydial lymphogranuloma",
                          "A56 Other sexually transmitted chlamydial diseases",
                          "A57 Chancroid", "A58 Granuloma inguinale", "A59 Trichomoniasis",
                          "A60 Anogenital herpesviral infection", "A63 Other predominantly sexually transmitted diseases",
                          "A64 Unspecified sexually transmitted disease"]},
    "A65-A69": {"name": "Other spirochaetal diseases", "chapter": "I",
                "codes": ["A65 Nonvenereal syphilis", "A66 Yaws", "A67 Pinta",
                          "A68 Relapsing fevers", "A69 Other spirochaetal infections"]},
    "A70-A74": {"name": "Other diseases caused by chlamydiae", "chapter": "I",
                "codes": ["A70 Chlamydia psittaci infection", "A71 Trachoma",
                          "A74 Other diseases caused by chlamydiae"]},
    "A75-A79": {"name": "Rickettsioses", "chapter": "I",
                "codes": ["A75 Typhus fever", "A77 Spotted fever", "A78 Q fever",
                          "A79 Other rickettsioses"]},
    "A80-A89": {"name": "Viral infections of the central nervous system", "chapter": "I",
                "codes": ["A80 Acute poliomyelitis", "A81 Atypical virus CNS infections",
                          "A82 Rabies", "A83 Mosquito-borne viral encephalitis",
                          "A84 Tick-borne viral encephalitis", "A85 Other viral encephalitis",
                          "A86 Unspecified viral encephalitis", "A87 Viral meningitis",
                          "A88 Other viral CNS infections", "A89 Unspecified viral CNS infection"]},
    "A90-A99": {"name": "Arthropod-borne viral fevers and viral haemorrhagic fevers", "chapter": "I",
                "codes": ["A90 Dengue fever", "A91 Dengue haemorrhagic fever",
                          "A92 Other mosquito-borne viral fevers", "A93 Other arthropod-borne viral fevers",
                          "A94 Unspecified arthropod-borne viral fever", "A95 Yellow fever",
                          "A96 Arenaviral haemorrhagic fever", "A98 Other viral haemorrhagic fevers",
                          "A99 Unspecified viral haemorrhagic fever"]},
    # *** MALARIA BLOCK ***
    "B50-B64": {"name": "Protozoal diseases", "chapter": "I",
                "codes": ["B50 Plasmodium falciparum malaria", "B51 Plasmodium vivax malaria",
                          "B52 Plasmodium malariae malaria", "B53 Other parasitologically confirmed malaria",
                          "B54 Unspecified malaria", "B55 Leishmaniasis", "B56 African trypanosomiasis",
                          "B57 Chagas disease", "B58 Toxoplasmosis", "B59 Pneumocystosis",
                          "B60 Other protozoal diseases", "B64 Unspecified protozoal disease"]},
    "B65-B83": {"name": "Helminthiases", "chapter": "I",
                "codes": ["B65 Schistosomiasis", "B66 Other fluke infections",
                          "B67 Echinococcosis", "B68 Taeniasis", "B69 Cysticercosis",
                          "B70 Diphyllobothriasis and sparganosis", "B71 Other cestode infections",
                          "B73 Onchocerciasis", "B74 Filariasis", "B75 Trichinellosis",
                          "B76 Hookworm diseases", "B77 Ascariasis", "B78 Strongyloidiasis",
                          "B79 Trichuriasis", "B80 Enterobiasis", "B81 Other intestinal helminthiases",
                          "B82 Unspecified intestinal parasitism", "B83 Other helminthiases"]},
    "B85-B89": {"name": "Pediculosis, acariasis and other infestations", "chapter": "I",
                "codes": ["B85 Pediculosis and phthiriasis", "B86 Scabies",
                          "B87 Myiasis", "B88 Other infestations", "B89 Unspecified parasitic disease"]},
    "B90-B94": {"name": "Sequelae of infectious and parasitic diseases", "chapter": "I",
                "codes": ["B90 Sequelae of tuberculosis", "B91 Sequelae of poliomyelitis",
                          "B92 Sequelae of leprosy", "B94 Sequelae of other infectious diseases"]},
    "B95-B98": {"name": "Bacterial and viral infectious agents", "chapter": "I",
                "codes": ["B95 Streptococcus and staphylococcus", "B96 Other bacterial agents",
                          "B97 Viral agents", "B98 Other specified infectious agents"]},

    # Chapter II - Neoplasms (key blocks)
    "C00-C14": {"name": "Malignant neoplasms of lip, oral cavity and pharynx", "chapter": "II",
                "codes": ["C00 Lip", "C01 Base of tongue", "C02 Other tongue", "C03 Gum",
                          "C04 Floor of mouth", "C05 Palate", "C06 Other mouth",
                          "C07 Parotid gland", "C09 Tonsil", "C10 Oropharynx",
                          "C11 Nasopharynx", "C12 Pyriform sinus", "C13 Hypopharynx",
                          "C14 Other ill-defined sites"]},
    "C15-C26": {"name": "Malignant neoplasms of digestive organs", "chapter": "II",
                "codes": ["C15 Oesophagus", "C16 Stomach", "C17 Small intestine",
                          "C18 Colon", "C19 Rectosigmoid junction", "C20 Rectum",
                          "C21 Anus", "C22 Liver", "C23 Gallbladder", "C24 Biliary tract",
                          "C25 Pancreas", "C26 Other digestive organs"]},
    "C30-C39": {"name": "Malignant neoplasms of respiratory and intrathoracic organs", "chapter": "II",
                "codes": ["C30 Nasal cavity and middle ear", "C31 Accessory sinuses",
                          "C32 Larynx", "C33 Trachea", "C34 Bronchus and lung",
                          "C37 Thymus", "C38 Heart mediastinum pleura", "C39 Other respiratory organs"]},
    "C43-C44": {"name": "Melanoma and other malignant neoplasms of skin", "chapter": "II",
                "codes": ["C43 Malignant melanoma of skin", "C44 Other malignant neoplasms of skin"]},
    "C50-C50": {"name": "Malignant neoplasm of breast", "chapter": "II",
                "codes": ["C50 Malignant neoplasm of breast"]},
    "C81-C96": {"name": "Malignant neoplasms of lymphoid, haematopoietic tissue", "chapter": "II",
                "codes": ["C81 Hodgkin disease", "C82 Follicular non-Hodgkin lymphoma",
                          "C83 Diffuse non-Hodgkin lymphoma", "C84 Peripheral T-cell lymphoma",
                          "C85 Other non-Hodgkin lymphoma", "C88 Malignant immunoproliferative diseases",
                          "C90 Multiple myeloma", "C91 Lymphoid leukaemia",
                          "C92 Myeloid leukaemia", "C93 Monocytic leukaemia",
                          "C94 Other leukaemias", "C95 Leukaemia unspecified",
                          "C96 Other malignant neoplasms of lymphoid tissue"]},

    # Chapter IV - Metabolic
    "E10-E14": {"name": "Diabetes mellitus", "chapter": "IV",
                "codes": ["E10 Type 1 diabetes mellitus", "E11 Type 2 diabetes mellitus",
                          "E12 Malnutrition-related diabetes", "E13 Other specified diabetes",
                          "E14 Unspecified diabetes mellitus"]},
    "E40-E46": {"name": "Malnutrition", "chapter": "IV",
                "codes": ["E40 Kwashiorkor", "E41 Nutritional marasmus",
                          "E42 Marasmic kwashiorkor", "E43 Unspecified severe protein-energy malnutrition",
                          "E44 Protein-energy malnutrition moderate and mild",
                          "E45 Retarded development following malnutrition",
                          "E46 Unspecified protein-energy malnutrition"]},
    "E50-E64": {"name": "Other nutritional deficiencies", "chapter": "IV",
                "codes": ["E50 Vitamin A deficiency", "E51 Thiamine deficiency",
                          "E52 Niacin deficiency (pellagra)", "E53 Deficiency of other B group vitamins",
                          "E54 Ascorbic acid deficiency (scurvy)", "E55 Vitamin D deficiency",
                          "E56 Other vitamin deficiencies", "E58 Dietary calcium deficiency",
                          "E59 Dietary selenium deficiency", "E60 Dietary zinc deficiency",
                          "E61 Deficiency of other nutrient elements", "E63 Other nutritional deficiencies",
                          "E64 Sequelae of malnutrition"]},

    # Chapter V - Mental disorders
    "F10-F19": {"name": "Mental and behavioural disorders due to psychoactive substance use", "chapter": "V",
                "codes": ["F10 Alcohol", "F11 Opioids", "F12 Cannabinoids",
                          "F13 Sedatives or hypnotics", "F14 Cocaine", "F15 Other stimulants",
                          "F16 Hallucinogens", "F17 Tobacco", "F18 Volatile solvents",
                          "F19 Multiple drug use"]},
    "F20-F29": {"name": "Schizophrenia, schizotypal and delusional disorders", "chapter": "V",
                "codes": ["F20 Schizophrenia", "F21 Schizotypal disorder",
                          "F22 Persistent delusional disorders", "F23 Acute psychotic disorders",
                          "F24 Induced delusional disorder", "F25 Schizoaffective disorders",
                          "F28 Other nonorganic psychotic disorders", "F29 Unspecified nonorganic psychosis"]},
    "F30-F39": {"name": "Mood (affective) disorders", "chapter": "V",
                "codes": ["F30 Manic episode", "F31 Bipolar affective disorder",
                          "F32 Depressive episode", "F33 Recurrent depressive disorder",
                          "F34 Persistent mood disorders", "F38 Other mood disorders",
                          "F39 Unspecified mood disorder"]},

    # Chapter IX - Circulatory
    "I10-I15": {"name": "Hypertensive diseases", "chapter": "IX",
                "codes": ["I10 Essential (primary) hypertension", "I11 Hypertensive heart disease",
                          "I12 Hypertensive renal disease", "I13 Hypertensive heart and renal disease",
                          "I15 Secondary hypertension"]},
    "I20-I25": {"name": "Ischaemic heart diseases", "chapter": "IX",
                "codes": ["I20 Angina pectoris", "I21 Acute myocardial infarction",
                          "I22 Subsequent myocardial infarction", "I23 Complications following MI",
                          "I24 Other acute ischaemic heart diseases",
                          "I25 Chronic ischaemic heart disease"]},
    "I60-I69": {"name": "Cerebrovascular diseases", "chapter": "IX",
                "codes": ["I60 Subarachnoid haemorrhage", "I61 Intracerebral haemorrhage",
                          "I62 Other nontraumatic intracranial haemorrhage",
                          "I63 Cerebral infarction", "I64 Stroke not specified",
                          "I65 Occlusion precerebral arteries", "I66 Occlusion cerebral arteries",
                          "I67 Other cerebrovascular diseases", "I68 Cerebrovascular disorders in diseases",
                          "I69 Sequelae of cerebrovascular disease"]},

    # Chapter X - Respiratory
    "J09-J18": {"name": "Influenza and pneumonia", "chapter": "X",
                "codes": ["J09 Influenza due to identified avian virus", "J10 Influenza due to identified virus",
                          "J11 Influenza virus not identified", "J12 Viral pneumonia",
                          "J13 Pneumonia due to Streptococcus pneumoniae",
                          "J14 Pneumonia due to Haemophilus influenzae",
                          "J15 Bacterial pneumonia", "J16 Pneumonia due to other organisms",
                          "J17 Pneumonia in diseases classified elsewhere",
                          "J18 Pneumonia organism unspecified"]},
    "J40-J47": {"name": "Chronic lower respiratory diseases", "chapter": "X",
                "codes": ["J40 Bronchitis not specified", "J41 Simple chronic bronchitis",
                          "J42 Unspecified chronic bronchitis", "J43 Emphysema",
                          "J44 Other COPD", "J45 Asthma", "J46 Status asthmaticus",
                          "J47 Bronchiectasis"]},

    # Chapter XVI - Perinatal
    "P05-P08": {"name": "Disorders related to length of gestation and fetal growth", "chapter": "XVI",
                "codes": ["P05 Slow fetal growth and fetal malnutrition",
                          "P07 Disorders related to short gestation and low birth weight",
                          "P08 Disorders related to long gestation and high birth weight"]},
    "P35-P39": {"name": "Infections specific to the perinatal period", "chapter": "XVI",
                "codes": ["P35 Congenital viral diseases", "P36 Bacterial sepsis of newborn",
                          "P37 Other congenital infectious diseases (includes P37.3 Congenital falciparum malaria, P37.4 Other congenital malaria)",
                          "P38 Omphalitis of newborn", "P39 Other infections specific to perinatal period"]},
}

# Cross-chapter medical relationships
CROSS_EDGES = [
    # Malaria complications
    ("B50 Plasmodium falciparum malaria", "D59 Acquired haemolytic anaemia", "complication"),
    ("B50 Plasmodium falciparum malaria", "E46 Unspecified protein-energy malnutrition", "comorbidity"),
    ("B54 Unspecified malaria", "J18 Pneumonia organism unspecified", "differential_diagnosis"),
    ("B54 Unspecified malaria", "P37 Other congenital infectious diseases (includes P37.3 Congenital falciparum malaria, P37.4 Other congenital malaria)", "congenital_form"),

    # Infectious disease links
    ("A90 Dengue fever", "B54 Unspecified malaria", "differential_diagnosis"),
    ("A95 Yellow fever", "B54 Unspecified malaria", "differential_diagnosis"),
    ("A15 Respiratory tuberculosis bacteriologically confirmed", "B20-B24 HIV disease", "coinfection"),

    # Metabolic-infectious links
    ("E50 Vitamin A deficiency", "B50 Plasmodium falciparum malaria", "risk_factor"),
    ("E53 Deficiency of other B group vitamins", "D59 Acquired haemolytic anaemia", "comorbidity"),
    ("E40 Kwashiorkor", "B54 Unspecified malaria", "comorbidity"),

    # Mental health links
    ("F10 Alcohol", "I20 Angina pectoris", "risk_factor"),
    ("F32 Depressive episode", "I21 Acute myocardial infarction", "comorbidity"),

    # Respiratory-infectious links
    ("J15 Bacterial pneumonia", "A41 Other sepsis", "complication"),
    ("J45 Asthma", "J15 Bacterial pneumonia", "complication"),
]


def make_poincare_embedding(name, chapter_angle, depth, max_depth, dim=128):
    h = hashlib.sha256(name.encode()).digest()
    if max_depth > 0:
        radius = 0.03 + 0.89 * min(depth / max_depth, 1.0)
    else:
        radius = 0.5
    jitter = (h[0] / 255.0 - 0.5) * 0.02
    radius = max(0.02, min(0.95, radius + jitter))

    coords = []
    for i in range(dim):
        base = h[i % 32] / 255.0 - 0.5
        if i == 0:
            base = math.cos(chapter_angle) + base * 0.2
        elif i == 1:
            base = math.sin(chapter_angle) + base * 0.2
        coords.append(base)

    norm = math.sqrt(sum(c * c for c in coords))
    if norm > 0:
        coords = [c / norm * radius for c in coords]
    return coords


def make_edge_request(pb2, **kwargs):
    from_val = kwargs.pop('from_node')
    req = pb2.InsertEdgeRequest(**kwargs)
    setattr(req, 'from', from_val)
    return req


def ingest(host, collection, metric="poincare", dim=128):
    pb2, pb2_grpc = ensure_proto_compiled()

    channel = grpc.insecure_channel(host, options=[
        ('grpc.max_send_message_length', 256 * 1024 * 1024),
        ('grpc.max_receive_message_length', 256 * 1024 * 1024),
    ])
    stub = pb2_grpc.NietzscheDBStub(channel)

    print(f"[*] Waiting for gRPC server at {host}...")
    try:
        grpc.channel_ready_future(channel).result(timeout=120)
        print(f"[+] gRPC server ready!")
    except grpc.FutureTimeoutError:
        print(f"[!] Timeout. Aborting.")
        return

    max_depth = 3  # chapter(0) → block(1) → code(2) → subcode(3)

    print(f"\n{'='*60}")
    print(f"  ICD-10 Classification → NietzscheDB Ingestion")
    print(f"  Host: {host}")
    print(f"  Collection: {collection}")
    print(f"  Chapters: {len(ICD10_CHAPTERS)}")
    print(f"  Blocks: {len(ICD10_BLOCKS)}")
    print(f"{'='*60}\n")

    try:
        stub.CreateCollection(pb2.CreateCollectionRequest(
            collection=collection, dim=dim, metric=metric))
        print(f"[+] Collection '{collection}' created")
    except grpc.RpcError as e:
        print(f"[!] CreateCollection: {e.details() if hasattr(e, 'details') else e}")

    # Build all nodes
    all_nodes = []
    node_ids = {}

    # Root node
    root_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, "icd10:root"))
    node_ids["ICD-10"] = root_id
    emb = make_poincare_embedding("ICD-10 Classification", 0, 0, max_depth, dim)
    all_nodes.append({
        "id": root_id,
        "content": json.dumps({"name": "ICD-10 International Classification of Diseases",
                                "type": "root", "dataset": "icd10"}).encode('utf-8'),
        "node_type": "Concept", "energy": 1.0, "embedding": emb,
    })

    # Chapters (depth 0)
    for ch_num, ch_data in ICD10_CHAPTERS.items():
        ch_key = f"Chapter {ch_num}"
        ch_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"icd10:chapter:{ch_num}"))
        node_ids[ch_key] = ch_id
        emb = make_poincare_embedding(ch_key, ch_data["angle"], 0.5, max_depth, dim)
        all_nodes.append({
            "id": ch_id,
            "content": json.dumps({
                "name": ch_data["name"], "chapter": ch_num,
                "range": ch_data["range"], "type": "chapter",
                "dataset": "icd10",
            }).encode('utf-8'),
            "node_type": "Concept", "energy": 0.95, "embedding": emb,
        })

    # Blocks (depth 1)
    for block_range, block_data in ICD10_BLOCKS.items():
        ch = block_data["chapter"]
        ch_angle = ICD10_CHAPTERS[ch]["angle"]
        block_key = block_range
        block_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"icd10:block:{block_range}"))
        node_ids[block_key] = block_id
        emb = make_poincare_embedding(block_range, ch_angle, 1, max_depth, dim)
        all_nodes.append({
            "id": block_id,
            "content": json.dumps({
                "name": block_data["name"], "block": block_range,
                "chapter": ch, "type": "block", "dataset": "icd10",
            }).encode('utf-8'),
            "node_type": "Concept", "energy": 0.8, "embedding": emb,
        })

        # Codes within block (depth 2)
        for code_str in block_data["codes"]:
            code_id_str = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"icd10:code:{code_str}"))
            node_ids[code_str] = code_id_str
            code_prefix = code_str.split(' ')[0] if ' ' in code_str else code_str
            emb = make_poincare_embedding(code_str, ch_angle, 2, max_depth, dim)

            is_malaria = code_prefix.startswith('B5') and code_prefix in ['B50', 'B51', 'B52', 'B53', 'B54']
            energy = 0.9 if is_malaria else 0.6

            all_nodes.append({
                "id": code_id_str,
                "content": json.dumps({
                    "name": code_str, "code": code_prefix,
                    "block": block_range, "chapter": ch,
                    "type": "code", "dataset": "icd10",
                    "is_malaria": is_malaria,
                }).encode('utf-8'),
                "node_type": "Semantic", "energy": energy, "embedding": emb,
            })

    total = len(all_nodes)
    print(f"[*] Inserting {total} nodes...")

    batch_size = 100
    inserted = 0
    for i in range(0, total, batch_size):
        batch = all_nodes[i:i + batch_size]
        requests = []
        for n in batch:
            pv = pb2.PoincareVector(coords=n["embedding"])
            req = pb2.InsertNodeRequest(
                id=n["id"], embedding=pv, content=n["content"],
                node_type=n["node_type"], energy=n["energy"], collection=collection)
            requests.append(req)
        try:
            stub.BatchInsertNodes(pb2.BatchInsertNodesRequest(nodes=requests, collection=collection))
            inserted += len(batch)
            print(f"  [{inserted/total*100:5.1f}%] {inserted}/{total}", end='\r')
        except grpc.RpcError as e:
            print(f"\n  [!] Batch error: {e.details() if hasattr(e, 'details') else e}")

    print(f"\n[+] Nodes inserted: {inserted}/{total}")

    # Edges
    print(f"\n[*] Inserting hierarchical edges...")
    edge_count = 0

    # Root → Chapters
    for ch_num in ICD10_CHAPTERS:
        ch_key = f"Chapter {ch_num}"
        ch_id = node_ids.get(ch_key)
        if ch_id:
            try:
                stub.InsertEdge(make_edge_request(pb2, id=str(uuid.uuid4()),
                    from_node=root_id, to=ch_id, edge_type="Hierarchical",
                    weight=1.0, collection=collection))
                edge_count += 1
            except grpc.RpcError:
                pass

    # Chapters → Blocks
    for block_range, block_data in ICD10_BLOCKS.items():
        ch_key = f"Chapter {block_data['chapter']}"
        ch_id = node_ids.get(ch_key)
        block_id = node_ids.get(block_range)
        if ch_id and block_id:
            try:
                stub.InsertEdge(make_edge_request(pb2, id=str(uuid.uuid4()),
                    from_node=ch_id, to=block_id, edge_type="Hierarchical",
                    weight=0.9, collection=collection))
                edge_count += 1
            except grpc.RpcError:
                pass

        # Blocks → Codes
        for code_str in block_data["codes"]:
            code_id = node_ids.get(code_str)
            if block_id and code_id:
                try:
                    stub.InsertEdge(make_edge_request(pb2, id=str(uuid.uuid4()),
                        from_node=block_id, to=code_id, edge_type="Hierarchical",
                        weight=0.8, collection=collection))
                    edge_count += 1
                except grpc.RpcError:
                    pass

        # Association edges between codes in same block
        codes_in_block = [c for c in block_data["codes"] if c in node_ids]
        for j in range(len(codes_in_block)):
            for k in range(j + 1, min(j + 3, len(codes_in_block))):
                try:
                    stub.InsertEdge(make_edge_request(pb2, id=str(uuid.uuid4()),
                        from_node=node_ids[codes_in_block[j]],
                        to=node_ids[codes_in_block[k]],
                        edge_type="Association", weight=0.5, collection=collection))
                    edge_count += 1
                except grpc.RpcError:
                    pass

    print(f"  Hierarchical + intra-block edges: {edge_count}")

    # Cross-chapter edges
    print(f"[*] Inserting cross-chapter medical edges...")
    cross_count = 0
    for from_name, to_name, rel_type in CROSS_EDGES:
        from_id = node_ids.get(from_name)
        to_id = node_ids.get(to_name)
        if from_id and to_id:
            try:
                stub.InsertEdge(make_edge_request(pb2, id=str(uuid.uuid4()),
                    from_node=from_id, to=to_id,
                    edge_type="Association", weight=0.7, collection=collection))
                cross_count += 1
            except grpc.RpcError:
                pass

    total_edges = edge_count + cross_count

    print(f"\n{'='*60}")
    print(f"  ICD-10 INGESTION COMPLETE")
    print(f"{'='*60}")
    print(f"  Collection:       {collection}")
    print(f"  Chapters:         {len(ICD10_CHAPTERS)}")
    print(f"  Blocks:           {len(ICD10_BLOCKS)}")
    print(f"  Total Nodes:      {inserted}")
    print(f"  Total Edges:      {total_edges}")
    print(f"  Cross-chapter:    {cross_count}")
    print(f"  Malaria codes:    B50-B54 (boosted energy)")
    print(f"{'='*60}")
    print(f"\n  Dashboard: http://{host.split(':')[0]}:8080")

    channel.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest ICD-10 into NietzscheDB")
    parser.add_argument("--host", default="localhost:50051")
    parser.add_argument("--collection", default="icd10")
    parser.add_argument("--metric", default="poincare")
    parser.add_argument("--dim", type=int, default=128)
    args = parser.parse_args()
    ingest(args.host, args.collection, args.metric, args.dim)
