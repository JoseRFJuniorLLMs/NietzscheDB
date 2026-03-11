#!/usr/bin/env python3
"""
Ingest PubMed Biomedical Knowledge Graph into NietzscheDB.

Builds a biomedical knowledge graph with:
- Diseases, drugs, genes, proteins, pathways
- Drug-disease, drug-gene, gene-disease relationships
- Focus on tropical/infectious diseases relevant to malaria

Usage:
  python scripts/ingest_pubmed_kg.py [--host HOST:PORT] [--collection NAME]

Requirements:
  pip install grpcio grpcio-tools
"""

import grpc
import json
import uuid
import math
import hashlib
import sys
import os
import argparse
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
# PUBMED BIOMEDICAL KNOWLEDGE GRAPH
# ═══════════════════════════════════════════════════════════════════════════

# Domain categories with angular sectors
DOMAINS = {
    "Diseases": 0.0,
    "Drugs": 1.0,
    "Genes": 2.0,
    "Proteins": 3.0,
    "Pathways": 4.0,
    "Organisms": 5.0,
    "Anatomy": 6.0,
}

# Diseases organized by system
DISEASES = {
    "Infectious Diseases": {
        "depth": 0.15,
        "items": [
            {"name": "Malaria", "mesh": "D008288", "synonyms": ["paludism"]},
            {"name": "Cerebral malaria", "mesh": "D016779"},
            {"name": "Severe malaria anaemia", "mesh": "D000740"},
            {"name": "Tuberculosis", "mesh": "D014376"},
            {"name": "HIV/AIDS", "mesh": "D000163"},
            {"name": "Dengue", "mesh": "D003715"},
            {"name": "Yellow fever", "mesh": "D015004"},
            {"name": "Ebola virus disease", "mesh": "D019142"},
            {"name": "COVID-19", "mesh": "D000086382"},
            {"name": "Influenza", "mesh": "D007251"},
            {"name": "Hepatitis B", "mesh": "D006509"},
            {"name": "Hepatitis C", "mesh": "D006526"},
            {"name": "Cholera", "mesh": "D002771"},
            {"name": "Typhoid fever", "mesh": "D014435"},
            {"name": "Leishmaniasis", "mesh": "D007896"},
            {"name": "Trypanosomiasis", "mesh": "D014352"},
            {"name": "Schistosomiasis", "mesh": "D012552"},
            {"name": "Onchocerciasis", "mesh": "D009855"},
            {"name": "Lymphatic filariasis", "mesh": "D004605"},
            {"name": "Rabies", "mesh": "D011818"},
            {"name": "Meningitis", "mesh": "D008581"},
            {"name": "Sepsis", "mesh": "D018805"},
            {"name": "Pneumonia", "mesh": "D011014"},
            {"name": "Diarrhoeal disease", "mesh": "D003967"},
        ]
    },
    "Cardiovascular Diseases": {
        "depth": 0.15,
        "items": [
            {"name": "Hypertension", "mesh": "D006973"},
            {"name": "Coronary artery disease", "mesh": "D003324"},
            {"name": "Heart failure", "mesh": "D006333"},
            {"name": "Stroke", "mesh": "D020521"},
            {"name": "Atrial fibrillation", "mesh": "D001281"},
            {"name": "Myocardial infarction", "mesh": "D009203"},
            {"name": "Atherosclerosis", "mesh": "D050197"},
            {"name": "Cardiomyopathy", "mesh": "D009202"},
            {"name": "Deep vein thrombosis", "mesh": "D020246"},
            {"name": "Pulmonary embolism", "mesh": "D011655"},
        ]
    },
    "Cancers": {
        "depth": 0.15,
        "items": [
            {"name": "Lung cancer", "mesh": "D002289"},
            {"name": "Breast cancer", "mesh": "D001943"},
            {"name": "Colorectal cancer", "mesh": "D015179"},
            {"name": "Prostate cancer", "mesh": "D011471"},
            {"name": "Liver cancer", "mesh": "D006528"},
            {"name": "Leukaemia", "mesh": "D007938"},
            {"name": "Lymphoma", "mesh": "D008223"},
            {"name": "Melanoma", "mesh": "D008545"},
            {"name": "Pancreatic cancer", "mesh": "D010190"},
            {"name": "Cervical cancer", "mesh": "D002583"},
        ]
    },
    "Metabolic Diseases": {
        "depth": 0.15,
        "items": [
            {"name": "Type 2 diabetes", "mesh": "D003924"},
            {"name": "Type 1 diabetes", "mesh": "D003922"},
            {"name": "Obesity", "mesh": "D009765"},
            {"name": "Metabolic syndrome", "mesh": "D024821"},
            {"name": "Hyperlipidaemia", "mesh": "D006949"},
            {"name": "Gout", "mesh": "D006073"},
            {"name": "Phenylketonuria", "mesh": "D010661"},
        ]
    },
    "Neurological Diseases": {
        "depth": 0.15,
        "items": [
            {"name": "Alzheimer disease", "mesh": "D000544"},
            {"name": "Parkinson disease", "mesh": "D010300"},
            {"name": "Epilepsy", "mesh": "D004827"},
            {"name": "Multiple sclerosis", "mesh": "D009103"},
            {"name": "Migraine", "mesh": "D008881"},
            {"name": "Depression", "mesh": "D003866"},
            {"name": "Schizophrenia", "mesh": "D012559"},
            {"name": "Autism spectrum disorder", "mesh": "D000067877"},
        ]
    },
    "Haematological Diseases": {
        "depth": 0.15,
        "items": [
            {"name": "Sickle cell disease", "mesh": "D000755"},
            {"name": "Thalassaemia", "mesh": "D013789"},
            {"name": "Iron deficiency anaemia", "mesh": "D018798"},
            {"name": "G6PD deficiency", "mesh": "D005955"},
            {"name": "Haemophilia", "mesh": "D006467"},
            {"name": "Aplastic anaemia", "mesh": "D000741"},
            {"name": "DIC", "mesh": "D004211"},
        ]
    },
}

# Drugs with targets and indications
DRUGS = {
    "Antimalarials": {
        "depth": 0.15,
        "items": [
            {"name": "Artemether", "target": "haem polymerisation", "class": "artemisinin"},
            {"name": "Lumefantrine", "target": "haem polymerisation", "class": "aryl amino alcohol"},
            {"name": "Artesunate", "target": "haem polymerisation", "class": "artemisinin"},
            {"name": "Amodiaquine", "target": "haem polymerisation", "class": "4-aminoquinoline"},
            {"name": "Chloroquine", "target": "haem polymerisation", "class": "4-aminoquinoline"},
            {"name": "Quinine", "target": "haem polymerisation", "class": "cinchona alkaloid"},
            {"name": "Mefloquine", "target": "haem polymerisation", "class": "quinoline methanol"},
            {"name": "Primaquine", "target": "mitochondrial electron transport", "class": "8-aminoquinoline"},
            {"name": "Tafenoquine", "target": "mitochondrial electron transport", "class": "8-aminoquinoline"},
            {"name": "Atovaquone", "target": "cytochrome bc1 complex", "class": "naphthoquinone"},
            {"name": "Proguanil", "target": "DHFR", "class": "biguanide"},
            {"name": "Sulfadoxine", "target": "DHPS", "class": "sulfonamide"},
            {"name": "Pyrimethamine", "target": "DHFR", "class": "diaminopyrimidine"},
            {"name": "Piperaquine", "target": "haem polymerisation", "class": "bisquinoline"},
        ]
    },
    "Antibiotics": {
        "depth": 0.15,
        "items": [
            {"name": "Amoxicillin", "target": "cell wall synthesis", "class": "penicillin"},
            {"name": "Azithromycin", "target": "50S ribosome", "class": "macrolide"},
            {"name": "Ciprofloxacin", "target": "DNA gyrase", "class": "fluoroquinolone"},
            {"name": "Doxycycline", "target": "30S ribosome", "class": "tetracycline"},
            {"name": "Metronidazole", "target": "DNA", "class": "nitroimidazole"},
            {"name": "Ceftriaxone", "target": "cell wall synthesis", "class": "cephalosporin"},
            {"name": "Vancomycin", "target": "cell wall synthesis", "class": "glycopeptide"},
            {"name": "Rifampicin", "target": "RNA polymerase", "class": "rifamycin"},
            {"name": "Isoniazid", "target": "mycolic acid synthesis", "class": "isonicotinic acid"},
        ]
    },
    "Antivirals": {
        "depth": 0.15,
        "items": [
            {"name": "Oseltamivir", "target": "neuraminidase", "class": "neuraminidase inhibitor"},
            {"name": "Tenofovir", "target": "reverse transcriptase", "class": "NRTI"},
            {"name": "Emtricitabine", "target": "reverse transcriptase", "class": "NRTI"},
            {"name": "Dolutegravir", "target": "integrase", "class": "INSTI"},
            {"name": "Sofosbuvir", "target": "NS5B polymerase", "class": "nucleotide analogue"},
            {"name": "Remdesivir", "target": "RNA-dependent RNA polymerase", "class": "nucleotide analogue"},
        ]
    },
    "Cardiovascular Drugs": {
        "depth": 0.15,
        "items": [
            {"name": "Atorvastatin", "target": "HMG-CoA reductase", "class": "statin"},
            {"name": "Amlodipine", "target": "L-type calcium channel", "class": "CCB"},
            {"name": "Lisinopril", "target": "ACE", "class": "ACE inhibitor"},
            {"name": "Metoprolol", "target": "beta-1 receptor", "class": "beta-blocker"},
            {"name": "Warfarin", "target": "VKORC1", "class": "anticoagulant"},
            {"name": "Aspirin", "target": "COX-1", "class": "NSAID"},
        ]
    },
}

# Key genes/proteins
GENES = {
    "Malaria-related genes": {
        "depth": 0.18,
        "items": [
            {"name": "PfKelch13 (K13)", "function": "artemisinin resistance marker", "organism": "P. falciparum"},
            {"name": "PfCRT", "function": "chloroquine resistance transporter", "organism": "P. falciparum"},
            {"name": "PfMDR1", "function": "multidrug resistance protein", "organism": "P. falciparum"},
            {"name": "PfDHFR", "function": "dihydrofolate reductase (antifolate target)", "organism": "P. falciparum"},
            {"name": "PfDHPS", "function": "dihydropteroate synthase (sulfa target)", "organism": "P. falciparum"},
            {"name": "PfAMA1", "function": "apical membrane antigen (vaccine target)", "organism": "P. falciparum"},
            {"name": "PfMSP1", "function": "merozoite surface protein (vaccine target)", "organism": "P. falciparum"},
            {"name": "PfCSP", "function": "circumsporozoite protein (RTS,S target)", "organism": "P. falciparum"},
            {"name": "PfHRP2", "function": "histidine-rich protein (RDT target)", "organism": "P. falciparum"},
            {"name": "PfPlasmepsin 2/3", "function": "piperaquine resistance marker", "organism": "P. falciparum"},
            {"name": "HBB (haemoglobin beta)", "function": "sickle cell trait protects against malaria", "organism": "Human"},
            {"name": "G6PD", "function": "glucose-6-phosphate dehydrogenase (primaquine sensitivity)", "organism": "Human"},
            {"name": "DARC/Duffy antigen", "function": "P. vivax receptor (Duffy-negative = resistant)", "organism": "Human"},
            {"name": "HLA-B53", "function": "MHC class I (malaria protection)", "organism": "Human"},
            {"name": "TNF-alpha", "function": "inflammatory cytokine in severe malaria", "organism": "Human"},
            {"name": "IL-10", "function": "anti-inflammatory cytokine in malaria", "organism": "Human"},
        ]
    },
    "Cancer genes": {
        "depth": 0.18,
        "items": [
            {"name": "TP53", "function": "tumour suppressor", "organism": "Human"},
            {"name": "BRCA1", "function": "DNA repair", "organism": "Human"},
            {"name": "BRCA2", "function": "DNA repair", "organism": "Human"},
            {"name": "KRAS", "function": "GTPase signalling", "organism": "Human"},
            {"name": "EGFR", "function": "receptor tyrosine kinase", "organism": "Human"},
            {"name": "HER2/ERBB2", "function": "receptor tyrosine kinase", "organism": "Human"},
            {"name": "MYC", "function": "transcription factor", "organism": "Human"},
            {"name": "RB1", "function": "cell cycle regulation", "organism": "Human"},
            {"name": "APC", "function": "Wnt pathway regulation", "organism": "Human"},
            {"name": "BCR-ABL", "function": "fusion oncogene (CML)", "organism": "Human"},
        ]
    },
    "Immune system genes": {
        "depth": 0.18,
        "items": [
            {"name": "IL-6", "function": "inflammatory cytokine", "organism": "Human"},
            {"name": "IFN-gamma", "function": "macrophage activation", "organism": "Human"},
            {"name": "TLR4", "function": "innate immune receptor (LPS)", "organism": "Human"},
            {"name": "TLR9", "function": "innate immune receptor (CpG DNA)", "organism": "Human"},
            {"name": "CD4", "function": "T-helper cell marker", "organism": "Human"},
            {"name": "CD8", "function": "cytotoxic T-cell marker", "organism": "Human"},
            {"name": "PD-1/PDCD1", "function": "immune checkpoint", "organism": "Human"},
            {"name": "CTLA-4", "function": "immune checkpoint", "organism": "Human"},
        ]
    },
}

# Biological pathways
PATHWAYS = [
    {"name": "Glycolysis", "genes": ["HK1", "PFK1", "PKM"], "diseases": ["Type 2 diabetes"]},
    {"name": "Haem biosynthesis", "genes": ["ALAS1", "FECH"], "diseases": ["Malaria"]},
    {"name": "Haemoglobin degradation (Plasmodium)", "genes": ["PfPlasmepsin 2/3"],
     "diseases": ["Malaria"], "drugs": ["Chloroquine", "Artemether"]},
    {"name": "Folate metabolism", "genes": ["PfDHFR", "PfDHPS"],
     "diseases": ["Malaria"], "drugs": ["Sulfadoxine", "Pyrimethamine"]},
    {"name": "Apoptosis (programmed cell death)", "genes": ["TP53", "BCR-ABL"],
     "diseases": ["Lung cancer", "Leukaemia"]},
    {"name": "NF-kB signalling", "genes": ["TNF-alpha", "IL-6", "TLR4"],
     "diseases": ["Sepsis", "Cerebral malaria"]},
    {"name": "PI3K/AKT/mTOR pathway", "genes": ["EGFR", "KRAS"],
     "diseases": ["Breast cancer", "Lung cancer"]},
    {"name": "Wnt/beta-catenin pathway", "genes": ["APC", "MYC"],
     "diseases": ["Colorectal cancer"]},
    {"name": "JAK/STAT signalling", "genes": ["IFN-gamma", "IL-6"],
     "diseases": ["Leukaemia", "Multiple sclerosis"]},
    {"name": "Complement cascade", "genes": ["TNF-alpha"],
     "diseases": ["Cerebral malaria", "Sepsis"]},
    {"name": "Pentose phosphate pathway", "genes": ["G6PD"],
     "diseases": ["G6PD deficiency", "Malaria"], "drugs": ["Primaquine"]},
    {"name": "Coagulation cascade", "genes": [],
     "diseases": ["DIC", "Deep vein thrombosis"], "drugs": ["Warfarin"]},
    {"name": "Renin-angiotensin system", "genes": [],
     "diseases": ["Hypertension"], "drugs": ["Lisinopril"]},
    {"name": "HMG-CoA reductase pathway", "genes": [],
     "diseases": ["Hyperlipidaemia", "Atherosclerosis"], "drugs": ["Atorvastatin"]},
]

# Drug-disease relationships (treats)
DRUG_DISEASE_EDGES = [
    ("Artemether", "Malaria"), ("Lumefantrine", "Malaria"),
    ("Artesunate", "Severe malaria anaemia"), ("Artesunate", "Cerebral malaria"),
    ("Chloroquine", "Malaria"), ("Quinine", "Severe malaria anaemia"),
    ("Primaquine", "Malaria"), ("Atovaquone", "Malaria"),
    ("Doxycycline", "Malaria"), ("Doxycycline", "Cholera"),
    ("Amoxicillin", "Pneumonia"), ("Azithromycin", "Cholera"),
    ("Ciprofloxacin", "Typhoid fever"), ("Ceftriaxone", "Meningitis"),
    ("Rifampicin", "Tuberculosis"), ("Isoniazid", "Tuberculosis"),
    ("Metronidazole", "Diarrhoeal disease"),
    ("Tenofovir", "HIV/AIDS"), ("Dolutegravir", "HIV/AIDS"),
    ("Sofosbuvir", "Hepatitis C"), ("Remdesivir", "COVID-19"),
    ("Oseltamivir", "Influenza"),
    ("Atorvastatin", "Hyperlipidaemia"), ("Amlodipine", "Hypertension"),
    ("Lisinopril", "Heart failure"), ("Metoprolol", "Heart failure"),
    ("Warfarin", "Atrial fibrillation"), ("Aspirin", "Coronary artery disease"),
]

# Gene-disease relationships
GENE_DISEASE_EDGES = [
    ("PfKelch13 (K13)", "Malaria"), ("PfCRT", "Malaria"),
    ("PfMDR1", "Malaria"), ("PfDHFR", "Malaria"),
    ("PfCSP", "Malaria"), ("PfHRP2", "Malaria"),
    ("HBB (haemoglobin beta)", "Sickle cell disease"),
    ("HBB (haemoglobin beta)", "Malaria"),
    ("G6PD", "G6PD deficiency"), ("G6PD", "Malaria"),
    ("DARC/Duffy antigen", "Malaria"),
    ("TNF-alpha", "Cerebral malaria"), ("TNF-alpha", "Sepsis"),
    ("IL-10", "Malaria"),
    ("TP53", "Lung cancer"), ("TP53", "Breast cancer"),
    ("BRCA1", "Breast cancer"), ("BRCA2", "Breast cancer"),
    ("KRAS", "Lung cancer"), ("KRAS", "Colorectal cancer"), ("KRAS", "Pancreatic cancer"),
    ("EGFR", "Lung cancer"), ("HER2/ERBB2", "Breast cancer"),
    ("BCR-ABL", "Leukaemia"), ("APC", "Colorectal cancer"),
    ("PD-1/PDCD1", "Melanoma"), ("CTLA-4", "Melanoma"),
]


def make_poincare_embedding(name, domain, depth, dim=128):
    h = hashlib.sha256(f"{domain}/{name}".encode()).digest()
    domain_angle = DOMAINS.get(domain, 0.0)

    radius = max(0.02, min(0.95, depth))
    jitter = (h[0] / 255.0 - 0.5) * 0.02
    radius = max(0.02, min(0.95, radius + jitter))

    coords = []
    for i in range(dim):
        base = h[i % 32] / 255.0 - 0.5
        if i == 0:
            base = math.cos(domain_angle) + base * 0.2
        elif i == 1:
            base = math.sin(domain_angle) + base * 0.2
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

    print(f"\n{'='*60}")
    print(f"  PubMed Biomedical KG → NietzscheDB Ingestion")
    print(f"  Host: {host} | Collection: {collection}")
    print(f"{'='*60}\n")

    try:
        stub.CreateCollection(pb2.CreateCollectionRequest(
            collection=collection, dim=dim, metric=metric))
        print(f"[+] Collection '{collection}' created")
    except grpc.RpcError as e:
        print(f"[!] CreateCollection: {e.details() if hasattr(e, 'details') else e}")

    all_nodes = []
    node_ids = {}
    edges = []

    # Root
    root_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, "pubmed-kg:root"))
    node_ids["PubMed KG"] = root_id
    all_nodes.append({
        "id": root_id,
        "content": json.dumps({"name": "PubMed Biomedical Knowledge Graph",
                                "type": "root", "dataset": "pubmed_kg"}).encode('utf-8'),
        "node_type": "Concept", "energy": 1.0,
        "embedding": make_poincare_embedding("PubMed KG", "Diseases", 0.02, dim),
    })

    # Domain roots
    for domain in DOMAINS:
        did = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"pubmed-kg:domain:{domain}"))
        node_ids[domain] = did
        all_nodes.append({
            "id": did,
            "content": json.dumps({"name": domain, "type": "domain",
                                    "dataset": "pubmed_kg"}).encode('utf-8'),
            "node_type": "Concept", "energy": 0.95,
            "embedding": make_poincare_embedding(domain, domain, 0.06, dim),
        })
        edges.append({"from": root_id, "to": did, "type": "Hierarchical", "weight": 1.0})

    # Diseases
    for category, cat_data in DISEASES.items():
        cat_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"pubmed-kg:disease-cat:{category}"))
        node_ids[category] = cat_id
        all_nodes.append({
            "id": cat_id,
            "content": json.dumps({"name": category, "type": "disease_category",
                                    "dataset": "pubmed_kg"}).encode('utf-8'),
            "node_type": "Concept", "energy": 0.85,
            "embedding": make_poincare_embedding(category, "Diseases", 0.12, dim),
        })
        edges.append({"from": node_ids["Diseases"], "to": cat_id, "type": "Hierarchical", "weight": 0.9})

        for item in cat_data["items"]:
            iid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"pubmed-kg:disease:{item['name']}"))
            node_ids[item["name"]] = iid
            is_malaria = "malaria" in item["name"].lower()
            content = {
                "name": item["name"], "type": "disease",
                "mesh_id": item.get("mesh", ""),
                "category": category, "dataset": "pubmed_kg",
            }
            if item.get("synonyms"):
                content["synonyms"] = item["synonyms"]
            all_nodes.append({
                "id": iid,
                "content": json.dumps(content).encode('utf-8'),
                "node_type": "Semantic", "energy": 0.9 if is_malaria else 0.6,
                "embedding": make_poincare_embedding(item["name"], "Diseases", 0.25, dim),
            })
            edges.append({"from": cat_id, "to": iid, "type": "Hierarchical", "weight": 0.8})

    # Drugs
    for category, cat_data in DRUGS.items():
        cat_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"pubmed-kg:drug-cat:{category}"))
        node_ids[category] = cat_id
        all_nodes.append({
            "id": cat_id,
            "content": json.dumps({"name": category, "type": "drug_category",
                                    "dataset": "pubmed_kg"}).encode('utf-8'),
            "node_type": "Concept", "energy": 0.85,
            "embedding": make_poincare_embedding(category, "Drugs", 0.12, dim),
        })
        edges.append({"from": node_ids["Drugs"], "to": cat_id, "type": "Hierarchical", "weight": 0.9})

        for item in cat_data["items"]:
            iid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"pubmed-kg:drug:{item['name']}"))
            node_ids[item["name"]] = iid
            content = {
                "name": item["name"], "type": "drug",
                "target": item["target"], "drug_class": item["class"],
                "category": category, "dataset": "pubmed_kg",
            }
            all_nodes.append({
                "id": iid,
                "content": json.dumps(content).encode('utf-8'),
                "node_type": "Semantic", "energy": 0.65,
                "embedding": make_poincare_embedding(item["name"], "Drugs", 0.25, dim),
            })
            edges.append({"from": cat_id, "to": iid, "type": "Hierarchical", "weight": 0.8})

    # Genes
    for category, cat_data in GENES.items():
        cat_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"pubmed-kg:gene-cat:{category}"))
        node_ids[category] = cat_id
        all_nodes.append({
            "id": cat_id,
            "content": json.dumps({"name": category, "type": "gene_category",
                                    "dataset": "pubmed_kg"}).encode('utf-8'),
            "node_type": "Concept", "energy": 0.85,
            "embedding": make_poincare_embedding(category, "Genes", 0.12, dim),
        })
        edges.append({"from": node_ids["Genes"], "to": cat_id, "type": "Hierarchical", "weight": 0.9})

        for item in cat_data["items"]:
            iid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"pubmed-kg:gene:{item['name']}"))
            node_ids[item["name"]] = iid
            content = {
                "name": item["name"], "type": "gene",
                "function": item["function"], "organism": item["organism"],
                "category": category, "dataset": "pubmed_kg",
            }
            all_nodes.append({
                "id": iid,
                "content": json.dumps(content).encode('utf-8'),
                "node_type": "Semantic", "energy": 0.65,
                "embedding": make_poincare_embedding(item["name"], "Genes", 0.28, dim),
            })
            edges.append({"from": cat_id, "to": iid, "type": "Hierarchical", "weight": 0.8})

    # Pathways
    for pw in PATHWAYS:
        pid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"pubmed-kg:pathway:{pw['name']}"))
        node_ids[pw["name"]] = pid
        content = {
            "name": pw["name"], "type": "pathway",
            "dataset": "pubmed_kg",
        }
        all_nodes.append({
            "id": pid,
            "content": json.dumps(content).encode('utf-8'),
            "node_type": "Concept", "energy": 0.75,
            "embedding": make_poincare_embedding(pw["name"], "Pathways", 0.20, dim),
        })
        edges.append({"from": node_ids["Pathways"], "to": pid, "type": "Hierarchical", "weight": 0.9})

        # Pathway → genes
        for gene in pw.get("genes", []):
            if gene in node_ids:
                edges.append({"from": pid, "to": node_ids[gene], "type": "Association", "weight": 0.7})

        # Pathway → diseases
        for disease in pw.get("diseases", []):
            if disease in node_ids:
                edges.append({"from": pid, "to": node_ids[disease], "type": "Association", "weight": 0.6})

        # Pathway → drugs
        for drug in pw.get("drugs", []):
            if drug in node_ids:
                edges.append({"from": pid, "to": node_ids[drug], "type": "Association", "weight": 0.5})

    # Drug-disease edges (treats)
    for drug_name, disease_name in DRUG_DISEASE_EDGES:
        if drug_name in node_ids and disease_name in node_ids:
            edges.append({"from": node_ids[drug_name], "to": node_ids[disease_name],
                          "type": "Association", "weight": 0.8})

    # Gene-disease edges
    for gene_name, disease_name in GENE_DISEASE_EDGES:
        if gene_name in node_ids and disease_name in node_ids:
            edges.append({"from": node_ids[gene_name], "to": node_ids[disease_name],
                          "type": "Association", "weight": 0.7})

    # Insert nodes
    total = len(all_nodes)
    batch_size = 100
    inserted = 0
    print(f"[*] Inserting {total} nodes...")

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

    # Insert edges
    print(f"\n[*] Inserting {len(edges)} edges...")
    edge_count = 0
    edge_batch = []
    for e in edges:
        edge_batch.append(e)
        if len(edge_batch) >= 100:
            reqs = []
            for eb in edge_batch:
                req = make_edge_request(pb2, id=str(uuid.uuid4()),
                    from_node=eb["from"], to=eb["to"],
                    edge_type=eb["type"], weight=eb["weight"], collection=collection)
                reqs.append(req)
            try:
                stub.BatchInsertEdges(pb2.BatchInsertEdgesRequest(edges=reqs, collection=collection))
                edge_count += len(edge_batch)
            except grpc.RpcError as e:
                print(f"\n  [!] Edge error: {e.details() if hasattr(e, 'details') else e}")
            edge_batch = []
            print(f"  Edges: {edge_count}", end='\r')

    if edge_batch:
        reqs = []
        for eb in edge_batch:
            req = make_edge_request(pb2, id=str(uuid.uuid4()),
                from_node=eb["from"], to=eb["to"],
                edge_type=eb["type"], weight=eb["weight"], collection=collection)
            reqs.append(req)
        try:
            stub.BatchInsertEdges(pb2.BatchInsertEdgesRequest(edges=reqs, collection=collection))
            edge_count += len(edge_batch)
        except grpc.RpcError:
            pass

    print(f"\n[+] Edges inserted: {edge_count}")

    # Stats
    n_diseases = sum(len(c["items"]) for c in DISEASES.values())
    n_drugs = sum(len(c["items"]) for c in DRUGS.values())
    n_genes = sum(len(c["items"]) for c in GENES.values())

    print(f"\n{'='*60}")
    print(f"  PUBMED BIOMEDICAL KG INGESTION COMPLETE")
    print(f"{'='*60}")
    print(f"  Collection:       {collection}")
    print(f"  Total Nodes:      {inserted}")
    print(f"  Total Edges:      {edge_count}")
    print(f"  Diseases:         {n_diseases}")
    print(f"  Drugs:            {n_drugs}")
    print(f"  Genes/Proteins:   {n_genes}")
    print(f"  Pathways:         {len(PATHWAYS)}")
    print(f"  Drug→Disease:     {len(DRUG_DISEASE_EDGES)}")
    print(f"  Gene→Disease:     {len(GENE_DISEASE_EDGES)}")
    print(f"{'='*60}")
    print(f"\n  The knowledge graph connects drugs, diseases, genes,")
    print(f"  and biological pathways. Malaria nodes are boosted!")
    print(f"\n  Dashboard: http://{host.split(':')[0]}:8080")

    channel.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PubMed KG into NietzscheDB")
    parser.add_argument("--host", default="localhost:50051")
    parser.add_argument("--collection", default="pubmed_kg")
    parser.add_argument("--metric", default="poincare")
    parser.add_argument("--dim", type=int, default=128)
    args = parser.parse_args()
    ingest(args.host, args.collection, args.metric, args.dim)
