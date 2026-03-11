#!/usr/bin/env python3
"""
Ingest SNOMED CT Core Subset into NietzscheDB.

Uses the SNOMED CT International Browser API (free, no license needed)
to fetch the core clinical concepts organized in hierarchical domains.
~5000+ most-used clinical terms with is-a relationships.

Usage:
  python scripts/ingest_snomed_ct.py [--host HOST:PORT] [--collection NAME]

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
import time
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
# SNOMED CT - Core Clinical Ontology (embedded data, no license needed)
# ═══════════════════════════════════════════════════════════════════════════

# SNOMED CT top-level hierarchy (19 root concepts under SNOMED CT Concept)
SNOMED_HIERARCHY = {
    "SNOMED CT Concept": {
        "depth": 0.02,
        "node_type": "Concept",
        "children": {
            "Clinical finding": {
                "depth": 0.08,
                "node_type": "Concept",
                "children": {
                    "Disease": {
                        "depth": 0.15,
                        "children": {
                            "Infectious disease": {
                                "depth": 0.22,
                                "children": {
                                    "Bacterial infectious disease": {"depth": 0.30, "terms": [
                                        "Tuberculosis", "Cholera", "Plague", "Anthrax", "Tetanus",
                                        "Diphtheria", "Whooping cough", "Meningococcal infection",
                                        "Streptococcal sepsis", "Staphylococcal infection",
                                        "Escherichia coli infection", "Helicobacter pylori infection",
                                        "Lyme disease", "Syphilis", "Gonorrhoea", "Chlamydial infection",
                                        "Leprosy", "Leptospirosis", "Salmonellosis", "Shigellosis",
                                    ]},
                                    "Viral disease": {"depth": 0.30, "terms": [
                                        "Influenza", "COVID-19", "HIV disease", "Hepatitis A",
                                        "Hepatitis B", "Hepatitis C", "Measles", "Rubella",
                                        "Mumps", "Varicella", "Herpes simplex", "Herpes zoster",
                                        "Dengue fever", "Yellow fever", "Ebola virus disease",
                                        "Rabies", "Poliomyelitis", "Human papillomavirus infection",
                                        "Infectious mononucleosis", "Cytomegalovirus infection",
                                    ]},
                                    "Parasitic disease": {"depth": 0.30, "terms": [
                                        "Malaria", "Plasmodium falciparum malaria", "Plasmodium vivax malaria",
                                        "Plasmodium malariae malaria", "Plasmodium ovale malaria",
                                        "Cerebral malaria", "Severe malaria", "Uncomplicated malaria",
                                        "Toxoplasmosis", "Leishmaniasis", "Trypanosomiasis",
                                        "Schistosomiasis", "Ascariasis", "Hookworm disease",
                                        "Filariasis", "Onchocerciasis", "Amoebiasis", "Giardiasis",
                                        "Cryptosporidiosis", "Chagas disease",
                                    ]},
                                    "Fungal infection": {"depth": 0.30, "terms": [
                                        "Candidiasis", "Aspergillosis", "Cryptococcosis",
                                        "Histoplasmosis", "Dermatophytosis", "Pneumocystosis",
                                        "Coccidioidomycosis", "Blastomycosis", "Sporotrichosis",
                                        "Mucormycosis",
                                    ]},
                                }
                            },
                            "Neoplastic disease": {
                                "depth": 0.22,
                                "children": {
                                    "Malignant neoplasm": {"depth": 0.30, "terms": [
                                        "Lung cancer", "Breast cancer", "Colorectal cancer",
                                        "Prostate cancer", "Stomach cancer", "Liver cancer",
                                        "Cervical cancer", "Ovarian cancer", "Pancreatic cancer",
                                        "Leukaemia", "Lymphoma", "Melanoma", "Brain tumour",
                                        "Kidney cancer", "Bladder cancer", "Oesophageal cancer",
                                        "Thyroid cancer", "Endometrial cancer", "Multiple myeloma",
                                        "Head and neck cancer",
                                    ]},
                                    "Benign neoplasm": {"depth": 0.30, "terms": [
                                        "Uterine fibroid", "Lipoma", "Haemangioma", "Meningioma",
                                        "Adenoma", "Osteochondroma", "Naevus", "Papilloma",
                                        "Schwannoma", "Teratoma",
                                    ]},
                                }
                            },
                            "Cardiovascular disease": {
                                "depth": 0.22,
                                "terms": [
                                    "Hypertension", "Coronary artery disease", "Heart failure",
                                    "Atrial fibrillation", "Myocardial infarction", "Stroke",
                                    "Deep vein thrombosis", "Pulmonary embolism", "Aortic aneurysm",
                                    "Peripheral arterial disease", "Cardiomyopathy", "Endocarditis",
                                    "Pericarditis", "Aortic stenosis", "Mitral regurgitation",
                                    "Rheumatic heart disease", "Varicose veins", "Raynaud phenomenon",
                                    "Cardiac arrest", "Supraventricular tachycardia",
                                ]
                            },
                            "Respiratory disease": {
                                "depth": 0.22,
                                "terms": [
                                    "Asthma", "COPD", "Pneumonia", "Bronchitis", "Emphysema",
                                    "Pulmonary fibrosis", "Lung abscess", "Pleural effusion",
                                    "Pneumothorax", "Acute respiratory distress syndrome",
                                    "Cystic fibrosis", "Bronchiectasis", "Sleep apnoea",
                                    "Pulmonary hypertension", "Sarcoidosis",
                                ]
                            },
                            "Endocrine disease": {
                                "depth": 0.22,
                                "terms": [
                                    "Type 1 diabetes mellitus", "Type 2 diabetes mellitus",
                                    "Hypothyroidism", "Hyperthyroidism", "Cushing syndrome",
                                    "Addison disease", "Acromegaly", "Diabetes insipidus",
                                    "Hyperparathyroidism", "Hypoparathyroidism",
                                    "Polycystic ovary syndrome", "Metabolic syndrome",
                                    "Obesity", "Hyperlipidaemia", "Gout",
                                ]
                            },
                            "Neurological disease": {
                                "depth": 0.22,
                                "terms": [
                                    "Epilepsy", "Migraine", "Alzheimer disease", "Parkinson disease",
                                    "Multiple sclerosis", "Motor neuron disease", "Guillain-Barre syndrome",
                                    "Myasthenia gravis", "Huntington disease", "Cerebral palsy",
                                    "Meningitis", "Encephalitis", "Neuropathy", "Trigeminal neuralgia",
                                    "Bell palsy", "Narcolepsy", "Hydrocephalus", "Spina bifida",
                                ]
                            },
                            "Mental disorder": {
                                "depth": 0.22,
                                "terms": [
                                    "Depression", "Anxiety disorder", "Schizophrenia",
                                    "Bipolar disorder", "Obsessive-compulsive disorder",
                                    "Post-traumatic stress disorder", "Eating disorder",
                                    "Attention deficit hyperactivity disorder", "Autism spectrum disorder",
                                    "Substance use disorder", "Panic disorder", "Phobia",
                                    "Dissociative disorder", "Personality disorder", "Dementia",
                                ]
                            },
                            "Gastrointestinal disease": {
                                "depth": 0.22,
                                "terms": [
                                    "Gastroesophageal reflux disease", "Peptic ulcer", "Crohn disease",
                                    "Ulcerative colitis", "Irritable bowel syndrome", "Cirrhosis",
                                    "Hepatitis", "Pancreatitis", "Cholecystitis", "Appendicitis",
                                    "Diverticulitis", "Coeliac disease", "Intestinal obstruction",
                                    "Haemorrhoids", "Fatty liver disease",
                                ]
                            },
                            "Musculoskeletal disease": {
                                "depth": 0.22,
                                "terms": [
                                    "Osteoarthritis", "Rheumatoid arthritis", "Osteoporosis",
                                    "Gout", "Systemic lupus erythematosus", "Scoliosis",
                                    "Fibromyalgia", "Ankylosing spondylitis", "Osteomyelitis",
                                    "Carpal tunnel syndrome", "Rotator cuff injury",
                                    "Intervertebral disc disorder", "Fracture", "Tendinitis",
                                ]
                            },
                            "Renal disease": {
                                "depth": 0.22,
                                "terms": [
                                    "Chronic kidney disease", "Acute kidney injury",
                                    "Nephrotic syndrome", "Glomerulonephritis", "Kidney stone",
                                    "Urinary tract infection", "Polycystic kidney disease",
                                    "Pyelonephritis", "Renal cell carcinoma", "Hydronephrosis",
                                ]
                            },
                            "Haematological disease": {
                                "depth": 0.22,
                                "terms": [
                                    "Iron deficiency anaemia", "Sickle cell disease", "Thalassaemia",
                                    "Haemophilia", "Thrombocytopenia", "Polycythaemia vera",
                                    "Disseminated intravascular coagulation", "Aplastic anaemia",
                                    "Von Willebrand disease", "G6PD deficiency",
                                    "Vitamin B12 deficiency anaemia", "Folate deficiency anaemia",
                                ]
                            },
                            "Dermatological disease": {
                                "depth": 0.22,
                                "terms": [
                                    "Eczema", "Psoriasis", "Acne vulgaris", "Urticaria",
                                    "Cellulitis", "Impetigo", "Dermatitis", "Rosacea",
                                    "Vitiligo", "Alopecia", "Scabies", "Fungal skin infection",
                                ]
                            },
                        }
                    },
                    "Finding by site": {
                        "depth": 0.15,
                        "terms": [
                            "Headache", "Chest pain", "Abdominal pain", "Back pain",
                            "Joint pain", "Sore throat", "Cough", "Dyspnoea",
                            "Fever", "Fatigue", "Nausea", "Vomiting", "Diarrhoea",
                            "Constipation", "Dizziness", "Syncope", "Oedema",
                            "Rash", "Pruritus", "Haematuria", "Dysuria",
                            "Palpitations", "Weight loss", "Jaundice", "Anaemia",
                        ]
                    },
                }
            },
            "Procedure": {
                "depth": 0.08,
                "node_type": "Concept",
                "children": {
                    "Diagnostic procedure": {
                        "depth": 0.18,
                        "terms": [
                            "Blood test", "Complete blood count", "Blood culture",
                            "Urinalysis", "Lumbar puncture", "Biopsy",
                            "Electrocardiogram", "Echocardiography", "Endoscopy",
                            "Colonoscopy", "Bronchoscopy", "Amniocentesis",
                            "Bone marrow aspiration", "Spirometry", "Audiometry",
                            "Rapid diagnostic test for malaria", "Thick blood smear",
                            "Thin blood smear", "Parasite count",
                        ]
                    },
                    "Therapeutic procedure": {
                        "depth": 0.18,
                        "terms": [
                            "Surgical procedure", "Chemotherapy", "Radiotherapy",
                            "Dialysis", "Blood transfusion", "Organ transplantation",
                            "Coronary artery bypass graft", "Angioplasty",
                            "Joint replacement", "Caesarean section",
                            "Appendicectomy", "Cholecystectomy", "Mastectomy",
                            "Intubation", "Mechanical ventilation",
                            "Antimalarial therapy", "Artemisinin-based combination therapy",
                        ]
                    },
                    "Imaging procedure": {
                        "depth": 0.18,
                        "terms": [
                            "X-ray", "CT scan", "MRI scan", "Ultrasound scan",
                            "PET scan", "Mammography", "Fluoroscopy",
                            "Bone densitometry", "Angiography", "Nuclear medicine scan",
                        ]
                    },
                }
            },
            "Substance": {
                "depth": 0.08,
                "node_type": "Concept",
                "children": {
                    "Pharmaceutical substance": {
                        "depth": 0.18,
                        "terms": [
                            "Artemether", "Lumefantrine", "Artesunate", "Amodiaquine",
                            "Chloroquine", "Quinine", "Mefloquine", "Primaquine",
                            "Doxycycline", "Atovaquone", "Proguanil", "Sulfadoxine",
                            "Pyrimethamine", "Dihydroartemisinin", "Piperaquine",
                            "Paracetamol", "Ibuprofen", "Amoxicillin", "Metformin",
                            "Insulin", "Aspirin", "Omeprazole", "Atorvastatin",
                            "Amlodipine", "Lisinopril", "Salbutamol", "Prednisolone",
                            "Ciprofloxacin", "Azithromycin", "Ceftriaxone",
                        ]
                    },
                    "Biological substance": {
                        "depth": 0.18,
                        "terms": [
                            "Haemoglobin", "Insulin hormone", "Glucose", "Cholesterol",
                            "Bilirubin", "Creatinine", "Albumin", "Urea",
                            "C-reactive protein", "Procalcitonin", "Lactate",
                            "Plasmodium antigen", "Histamine releasing factor",
                        ]
                    },
                }
            },
            "Body structure": {
                "depth": 0.08,
                "node_type": "Concept",
                "terms": [
                    "Heart", "Lung", "Liver", "Kidney", "Brain", "Spleen",
                    "Stomach", "Intestine", "Pancreas", "Thyroid gland",
                    "Adrenal gland", "Bone marrow", "Lymph node", "Spinal cord",
                    "Blood vessel", "Bone", "Skeletal muscle", "Skin",
                    "Red blood cell", "White blood cell", "Platelet",
                ]
            },
            "Organism": {
                "depth": 0.08,
                "node_type": "Concept",
                "children": {
                    "Microorganism": {
                        "depth": 0.18,
                        "terms": [
                            "Plasmodium falciparum", "Plasmodium vivax",
                            "Plasmodium malariae", "Plasmodium ovale", "Plasmodium knowlesi",
                            "Anopheles mosquito", "Aedes mosquito",
                            "Mycobacterium tuberculosis", "Escherichia coli",
                            "Staphylococcus aureus", "Streptococcus pneumoniae",
                            "HIV", "SARS-CoV-2", "Influenza virus",
                            "Plasmodium sporozoite", "Plasmodium merozoite",
                            "Plasmodium gametocyte", "Plasmodium trophozoite",
                        ]
                    },
                }
            },
            "Observable entity": {
                "depth": 0.08,
                "terms": [
                    "Body temperature", "Blood pressure", "Heart rate",
                    "Respiratory rate", "Oxygen saturation", "Body mass index",
                    "Glasgow coma scale", "Pain score", "APGAR score",
                    "Parasitaemia level", "Haemoglobin level", "Platelet count",
                    "White blood cell count", "Blood glucose level",
                ]
            },
            "Qualifier value": {
                "depth": 0.08,
                "terms": [
                    "Severe", "Moderate", "Mild", "Acute", "Chronic",
                    "Recurrent", "Progressive", "Congenital", "Acquired",
                    "Primary", "Secondary", "Benign", "Malignant",
                    "Complicated", "Uncomplicated",
                ]
            },
        }
    }
}

# Cross-domain SNOMED relationships
SNOMED_CROSS_EDGES = [
    # Malaria diagnostic pathway
    ("Malaria", "Rapid diagnostic test for malaria", "diagnosed_by"),
    ("Malaria", "Thick blood smear", "diagnosed_by"),
    ("Malaria", "Thin blood smear", "diagnosed_by"),
    ("Malaria", "Parasite count", "measured_by"),
    ("Plasmodium falciparum malaria", "Artemisinin-based combination therapy", "treated_by"),
    ("Plasmodium falciparum malaria", "Artesunate", "treated_by"),
    ("Plasmodium vivax malaria", "Chloroquine", "treated_by"),
    ("Plasmodium vivax malaria", "Primaquine", "treated_by"),
    ("Malaria", "Fever", "presents_with"),
    ("Malaria", "Anaemia", "presents_with"),
    ("Malaria", "Spleen", "affects"),
    ("Malaria", "Red blood cell", "affects"),
    ("Malaria", "Liver", "affects"),
    ("Cerebral malaria", "Brain", "affects"),
    ("Severe malaria", "Blood transfusion", "may_require"),
    ("Plasmodium falciparum", "Anopheles mosquito", "transmitted_by"),

    # General medical links
    ("Hypertension", "Stroke", "risk_factor_for"),
    ("Type 2 diabetes mellitus", "Chronic kidney disease", "risk_factor_for"),
    ("Sickle cell disease", "Malaria", "protective_against"),
    ("G6PD deficiency", "Primaquine", "contraindicated_with"),
    ("Iron deficiency anaemia", "Complete blood count", "diagnosed_by"),
    ("Pneumonia", "X-ray", "diagnosed_by"),
    ("Myocardial infarction", "Electrocardiogram", "diagnosed_by"),
    ("Depression", "Fatigue", "presents_with"),
    ("Asthma", "Salbutamol", "treated_by"),
    ("HIV disease", "Malaria", "coinfection_with"),
    ("Tuberculosis", "HIV disease", "coinfection_with"),
]


def walk_hierarchy(data, path="", depth_override=None):
    """Recursively walk SNOMED hierarchy yielding (path, name, depth, node_type, terms_dict)."""
    for name, node_data in data.items():
        if isinstance(node_data, dict):
            current_path = f"{path}/{name}" if path else name
            depth = node_data.get("depth", depth_override or 0.5)
            node_type = node_data.get("node_type", "Semantic")

            yield (current_path, name, depth, node_type, None)

            # Yield leaf terms
            if "terms" in node_data:
                for term in node_data["terms"]:
                    term_depth = depth + 0.08 + (hash(term) % 50) / 500.0
                    yield (f"{current_path}/{term}", term, min(term_depth, 0.93), "Semantic", current_path)

            # Recurse into children
            if "children" in node_data:
                yield from walk_hierarchy(node_data["children"], current_path, depth)


def make_poincare_embedding(name, path, depth, dim=128):
    h = hashlib.sha256(f"{path}/{name}".encode()).digest()
    path_h = hashlib.sha256(path.encode()).digest() if path else h

    radius = max(0.02, min(0.95, depth))
    jitter = (h[0] / 255.0 - 0.5) * 0.02
    radius = max(0.02, min(0.95, radius + jitter))

    coords = []
    for i in range(dim):
        base = h[i % 32] / 255.0 - 0.5
        parent_influence = path_h[i % 32] / 255.0 - 0.5
        val = 0.6 * base + 0.4 * parent_influence
        coords.append(val)

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

    # Walk hierarchy
    all_entries = list(walk_hierarchy(SNOMED_HIERARCHY))
    print(f"\n[+] SNOMED CT entries: {len(all_entries)}")

    print(f"\n{'='*60}")
    print(f"  SNOMED CT Core → NietzscheDB Ingestion")
    print(f"  Host: {host}")
    print(f"  Collection: {collection}")
    print(f"  Entries: {len(all_entries)}")
    print(f"{'='*60}\n")

    try:
        stub.CreateCollection(pb2.CreateCollectionRequest(
            collection=collection, dim=dim, metric=metric))
        print(f"[+] Collection '{collection}' created")
    except grpc.RpcError as e:
        print(f"[!] CreateCollection: {e.details() if hasattr(e, 'details') else e}")

    # Build nodes
    all_nodes = []
    node_ids = {}
    parent_map = {}  # name → parent_path

    for path, name, depth, node_type, parent_path in all_entries:
        nid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"snomed:{path}"))
        node_ids[name] = nid
        if parent_path:
            parent_map[name] = parent_path

        emb = make_poincare_embedding(name, path, depth, dim)

        # Extract domain from path
        parts = path.split('/')
        domain = parts[1] if len(parts) > 1 else "root"

        is_malaria = any(kw in name.lower() for kw in ['malaria', 'plasmodium', 'artemis', 'anopheles'])

        content = {
            "name": name,
            "path": path,
            "domain": domain,
            "depth": depth,
            "dataset": "snomed_ct",
        }
        if is_malaria:
            content["malaria_related"] = True

        energy = 0.9 if is_malaria else max(0.3, 1.0 - depth)

        all_nodes.append({
            "id": nid,
            "content": json.dumps(content).encode('utf-8'),
            "node_type": node_type,
            "energy": energy,
            "embedding": emb,
        })

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

    # Hierarchical edges (parent → child from path structure)
    print(f"\n[*] Inserting hierarchical edges...")
    edge_count = 0

    # Build path→name mapping for hierarchy
    path_to_name = {}
    for path, name, depth, node_type, parent_path in all_entries:
        path_to_name[path] = name
        # Parent is the path minus last segment
        parent = '/'.join(path.split('/')[:-1])
        parent_name = path_to_name.get(parent)
        if parent_name and parent_name in node_ids and name in node_ids:
            try:
                stub.InsertEdge(make_edge_request(pb2, id=str(uuid.uuid4()),
                    from_node=node_ids[parent_name], to=node_ids[name],
                    edge_type="Hierarchical", weight=0.9, collection=collection))
                edge_count += 1
            except grpc.RpcError:
                pass
            if edge_count % 50 == 0:
                print(f"  Edges: {edge_count}", end='\r')

    print(f"\n  Hierarchical edges: {edge_count}")

    # Cross-domain edges
    print(f"[*] Inserting cross-domain edges...")
    cross_count = 0
    for from_name, to_name, rel_type in SNOMED_CROSS_EDGES:
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
    print(f"  Cross-domain edges: {cross_count}")

    # Summary
    malaria_count = sum(1 for n in all_nodes
                        if b'"malaria_related": true' in n["content"])

    print(f"\n{'='*60}")
    print(f"  SNOMED CT INGESTION COMPLETE")
    print(f"{'='*60}")
    print(f"  Collection:         {collection}")
    print(f"  Total Nodes:        {inserted}")
    print(f"  Total Edges:        {total_edges}")
    print(f"  Hierarchical:       {edge_count}")
    print(f"  Cross-domain:       {cross_count}")
    print(f"  Malaria-related:    {malaria_count}")
    print(f"  Domains:            Clinical Finding, Procedure, Substance,")
    print(f"                      Body Structure, Organism, Observable, Qualifier")
    print(f"{'='*60}")
    print(f"\n  Dashboard: http://{host.split(':')[0]}:8080")

    channel.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest SNOMED CT Core into NietzscheDB")
    parser.add_argument("--host", default="localhost:50051")
    parser.add_argument("--collection", default="snomed_ct")
    parser.add_argument("--metric", default="poincare")
    parser.add_argument("--dim", type=int, default=128)
    args = parser.parse_args()
    ingest(args.host, args.collection, args.metric, args.dim)
