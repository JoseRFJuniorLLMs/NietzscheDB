#!/usr/bin/env python3
"""
Ingest Malaria Atlas Project data into NietzscheDB.

Builds a comprehensive malaria knowledge graph with:
- Epidemiological data by country/region (prevalence, incidence, mortality)
- Vector species (Anopheles mosquitoes) and their distribution
- Interventions (ITNs, IRS, ACTs) and coverage data
- Drug resistance markers and geographic spread
- Climate/environmental factors

Focus on Angola, São Tomé e Príncipe, and Sub-Saharan Africa.

Usage:
  python scripts/ingest_malaria_atlas.py [--host HOST:PORT] [--collection NAME]

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
# MALARIA ATLAS DATA - Epidemiology, Vectors, Interventions
# ═══════════════════════════════════════════════════════════════════════════

# Geographic hierarchy: Region → Country → Province
MALARIA_GEO = {
    "Sub-Saharan Africa": {
        "depth": 0.06,
        "countries": {
            "Angola": {
                "depth": 0.15,
                "iso": "AGO", "lat": -12.3, "lon": 17.5,
                "prevalence_pf": 0.35, "incidence_per_1000": 215,
                "deaths_2022": 15000, "population_at_risk": 33000000,
                "itn_coverage": 0.42, "irs_coverage": 0.08,
                "provinces": [
                    {"name": "Luanda", "lat": -8.84, "lon": 13.23, "prevalence": 0.15, "urban": True},
                    {"name": "Benguela", "lat": -12.58, "lon": 13.41, "prevalence": 0.25, "urban": False},
                    {"name": "Huambo", "lat": -12.78, "lon": 15.73, "prevalence": 0.30, "urban": False},
                    {"name": "Huíla", "lat": -14.92, "lon": 13.49, "prevalence": 0.28, "urban": False},
                    {"name": "Cabinda", "lat": -5.55, "lon": 12.19, "prevalence": 0.40, "urban": False},
                    {"name": "Malanje", "lat": -9.54, "lon": 16.34, "prevalence": 0.45, "urban": False},
                    {"name": "Lunda Norte", "lat": -7.77, "lon": 20.41, "prevalence": 0.50, "urban": False},
                    {"name": "Lunda Sul", "lat": -10.72, "lon": 20.39, "prevalence": 0.48, "urban": False},
                    {"name": "Moxico", "lat": -13.43, "lon": 21.44, "prevalence": 0.55, "urban": False},
                    {"name": "Uíge", "lat": -7.61, "lon": 15.06, "prevalence": 0.42, "urban": False},
                    {"name": "Zaire", "lat": -6.27, "lon": 14.24, "prevalence": 0.38, "urban": False},
                    {"name": "Bengo", "lat": -8.45, "lon": 13.56, "prevalence": 0.32, "urban": False},
                    {"name": "Kwanza Norte", "lat": -9.17, "lon": 14.97, "prevalence": 0.35, "urban": False},
                    {"name": "Kwanza Sul", "lat": -10.55, "lon": 14.86, "prevalence": 0.33, "urban": False},
                    {"name": "Cunene", "lat": -16.27, "lon": 16.14, "prevalence": 0.18, "urban": False},
                    {"name": "Namibe", "lat": -15.19, "lon": 12.15, "prevalence": 0.10, "urban": False},
                    {"name": "Cuando Cubango", "lat": -15.60, "lon": 18.49, "prevalence": 0.35, "urban": False},
                    {"name": "Bié", "lat": -12.39, "lon": 17.67, "prevalence": 0.40, "urban": False},
                ],
            },
            "São Tomé e Príncipe": {
                "depth": 0.15,
                "iso": "STP", "lat": 0.19, "lon": 6.61,
                "prevalence_pf": 0.02, "incidence_per_1000": 12,
                "deaths_2022": 5, "population_at_risk": 220000,
                "itn_coverage": 0.75, "irs_coverage": 0.45,
                "provinces": [
                    {"name": "Água Grande", "lat": 0.30, "lon": 6.73, "prevalence": 0.01, "urban": True},
                    {"name": "Mé-Zóchi", "lat": 0.28, "lon": 6.63, "prevalence": 0.02, "urban": False},
                    {"name": "Cantagalo", "lat": 0.23, "lon": 6.59, "prevalence": 0.03, "urban": False},
                    {"name": "Caué", "lat": 0.13, "lon": 6.57, "prevalence": 0.04, "urban": False},
                    {"name": "Lembá", "lat": 0.37, "lon": 6.55, "prevalence": 0.02, "urban": False},
                    {"name": "Lobata", "lat": 0.35, "lon": 6.66, "prevalence": 0.02, "urban": False},
                    {"name": "Príncipe (RAP)", "lat": 1.62, "lon": 7.40, "prevalence": 0.01, "urban": False},
                ],
            },
            "Mozambique": {
                "depth": 0.15,
                "iso": "MOZ", "lat": -18.67, "lon": 35.53,
                "prevalence_pf": 0.40, "incidence_per_1000": 280,
                "deaths_2022": 18000, "population_at_risk": 31000000,
                "itn_coverage": 0.55, "irs_coverage": 0.15,
                "provinces": [],
            },
            "Nigeria": {
                "depth": 0.15,
                "iso": "NGA", "lat": 9.08, "lon": 7.49,
                "prevalence_pf": 0.27, "incidence_per_1000": 290,
                "deaths_2022": 130000, "population_at_risk": 210000000,
                "itn_coverage": 0.50, "irs_coverage": 0.05,
                "provinces": [],
            },
            "Democratic Republic of Congo": {
                "depth": 0.15,
                "iso": "COD", "lat": -4.04, "lon": 21.76,
                "prevalence_pf": 0.32, "incidence_per_1000": 300,
                "deaths_2022": 40000, "population_at_risk": 95000000,
                "itn_coverage": 0.45, "irs_coverage": 0.03,
                "provinces": [],
            },
            "Tanzania": {
                "depth": 0.15,
                "iso": "TZA", "lat": -6.37, "lon": 34.89,
                "prevalence_pf": 0.14, "incidence_per_1000": 120,
                "deaths_2022": 9500, "population_at_risk": 61000000,
                "itn_coverage": 0.65, "irs_coverage": 0.10,
                "provinces": [],
            },
            "Kenya": {
                "depth": 0.15,
                "iso": "KEN", "lat": -0.02, "lon": 37.91,
                "prevalence_pf": 0.08, "incidence_per_1000": 70,
                "deaths_2022": 5200, "population_at_risk": 35000000,
                "itn_coverage": 0.70, "irs_coverage": 0.12,
                "provinces": [],
            },
            "Uganda": {
                "depth": 0.15,
                "iso": "UGA", "lat": 1.37, "lon": 32.29,
                "prevalence_pf": 0.18, "incidence_per_1000": 200,
                "deaths_2022": 12000, "population_at_risk": 46000000,
                "itn_coverage": 0.60, "irs_coverage": 0.20,
                "provinces": [],
            },
            "Ghana": {
                "depth": 0.15,
                "iso": "GHA", "lat": 7.95, "lon": -1.02,
                "prevalence_pf": 0.20, "incidence_per_1000": 180,
                "deaths_2022": 7800, "population_at_risk": 32000000,
                "itn_coverage": 0.55, "irs_coverage": 0.08,
                "provinces": [],
            },
            "Cameroon": {
                "depth": 0.15,
                "iso": "CMR", "lat": 7.37, "lon": 12.35,
                "prevalence_pf": 0.25, "incidence_per_1000": 220,
                "deaths_2022": 8500, "population_at_risk": 27000000,
                "itn_coverage": 0.50, "irs_coverage": 0.06,
                "provinces": [],
            },
        }
    },
    "South Asia": {
        "depth": 0.06,
        "countries": {
            "India": {
                "depth": 0.15,
                "iso": "IND", "lat": 20.59, "lon": 78.96,
                "prevalence_pf": 0.03, "incidence_per_1000": 15,
                "deaths_2022": 5500, "population_at_risk": 700000000,
                "itn_coverage": 0.30, "irs_coverage": 0.35,
                "provinces": [],
            },
        }
    },
    "Southeast Asia": {
        "depth": 0.06,
        "countries": {
            "Myanmar": {
                "depth": 0.15,
                "iso": "MMR", "lat": 21.91, "lon": 95.96,
                "prevalence_pf": 0.01, "incidence_per_1000": 8,
                "deaths_2022": 200, "population_at_risk": 30000000,
                "itn_coverage": 0.60, "irs_coverage": 0.05,
                "provinces": [],
            },
        }
    },
    "Americas": {
        "depth": 0.06,
        "countries": {
            "Brazil": {
                "depth": 0.15,
                "iso": "BRA", "lat": -14.24, "lon": -51.93,
                "prevalence_pf": 0.005, "incidence_per_1000": 5,
                "deaths_2022": 60, "population_at_risk": 50000000,
                "itn_coverage": 0.20, "irs_coverage": 0.02,
                "provinces": [],
            },
        }
    },
}

# Anopheles vector species
VECTOR_SPECIES = [
    {"name": "Anopheles gambiae", "region": "Sub-Saharan Africa", "behaviour": "anthropophilic, endophagic",
     "resistance": ["pyrethroids", "DDT"], "primary": True},
    {"name": "Anopheles funestus", "region": "Sub-Saharan Africa", "behaviour": "anthropophilic, endophagic",
     "resistance": ["pyrethroids"], "primary": True},
    {"name": "Anopheles arabiensis", "region": "Sub-Saharan Africa", "behaviour": "zoophilic, exophagic",
     "resistance": ["pyrethroids", "organophosphates"], "primary": True},
    {"name": "Anopheles coluzzii", "region": "West Africa", "behaviour": "anthropophilic",
     "resistance": ["pyrethroids", "carbamates"], "primary": True},
    {"name": "Anopheles melas", "region": "West Africa coastal", "behaviour": "brackish water breeding",
     "resistance": [], "primary": False},
    {"name": "Anopheles moucheti", "region": "Central Africa", "behaviour": "forest mosquito",
     "resistance": [], "primary": False},
    {"name": "Anopheles nili", "region": "Central/West Africa", "behaviour": "river breeding",
     "resistance": [], "primary": False},
    {"name": "Anopheles stephensi", "region": "South Asia, Horn of Africa (invasive)", "behaviour": "urban adapted",
     "resistance": ["pyrethroids", "organophosphates"], "primary": True},
    {"name": "Anopheles dirus", "region": "Southeast Asia", "behaviour": "forest mosquito",
     "resistance": [], "primary": False},
    {"name": "Anopheles minimus", "region": "Southeast Asia", "behaviour": "foothill streams",
     "resistance": [], "primary": False},
    {"name": "Anopheles darlingi", "region": "South America", "behaviour": "anthropophilic",
     "resistance": [], "primary": True},
]

# Antimalarial drugs and resistance
ANTIMALARIALS = [
    {"name": "Artemether-Lumefantrine (AL)", "type": "ACT", "first_line": True,
     "resistance_markers": ["K13 C580Y", "K13 R539T"], "countries_resistant": ["Myanmar", "Cambodia"]},
    {"name": "Artesunate-Amodiaquine (ASAQ)", "type": "ACT", "first_line": True,
     "resistance_markers": ["pfcrt K76T"], "countries_resistant": []},
    {"name": "Dihydroartemisinin-Piperaquine (DHA-PPQ)", "type": "ACT", "first_line": True,
     "resistance_markers": ["plasmepsin 2/3 amplification"], "countries_resistant": ["Cambodia", "Vietnam"]},
    {"name": "Artesunate-Sulfadoxine-Pyrimethamine (AS-SP)", "type": "ACT", "first_line": False,
     "resistance_markers": ["dhfr triple mutant", "dhps double mutant"],
     "countries_resistant": ["Tanzania", "Kenya", "Uganda"]},
    {"name": "Chloroquine", "type": "4-aminoquinoline", "first_line": False,
     "resistance_markers": ["pfcrt K76T", "pfmdr1 N86Y"],
     "countries_resistant": ["Most of Sub-Saharan Africa"]},
    {"name": "Quinine", "type": "cinchona alkaloid", "first_line": False,
     "resistance_markers": [], "countries_resistant": []},
    {"name": "Primaquine", "type": "8-aminoquinoline", "first_line": False,
     "resistance_markers": [], "countries_resistant": []},
    {"name": "Sulfadoxine-Pyrimethamine (SP/IPTp)", "type": "antifolate", "first_line": False,
     "resistance_markers": ["dhfr triple mutant", "dhps K540E"],
     "countries_resistant": ["East Africa"]},
    {"name": "Mefloquine", "type": "quinoline methanol", "first_line": False,
     "resistance_markers": ["pfmdr1 amplification"],
     "countries_resistant": ["Thailand-Cambodia border"]},
]

# Interventions
INTERVENTIONS = [
    {"name": "Insecticide-Treated Nets (ITNs)", "type": "vector_control",
     "description": "Long-lasting insecticidal nets for sleeping", "effectiveness": 0.50},
    {"name": "Indoor Residual Spraying (IRS)", "type": "vector_control",
     "description": "Spraying insecticide on indoor walls", "effectiveness": 0.40},
    {"name": "Intermittent Preventive Treatment in Pregnancy (IPTp)", "type": "chemoprevention",
     "description": "SP doses during pregnancy at ANC visits", "effectiveness": 0.35},
    {"name": "Seasonal Malaria Chemoprevention (SMC)", "type": "chemoprevention",
     "description": "SPAQ given to children during rainy season", "effectiveness": 0.75},
    {"name": "RTS,S/AS01 Vaccine (Mosquirix)", "type": "vaccine",
     "description": "First WHO-recommended malaria vaccine", "effectiveness": 0.30},
    {"name": "R21/Matrix-M Vaccine", "type": "vaccine",
     "description": "Second-generation malaria vaccine, higher efficacy", "effectiveness": 0.75},
    {"name": "Larval Source Management (LSM)", "type": "vector_control",
     "description": "Larviciding and habitat modification", "effectiveness": 0.25},
    {"name": "Rapid Diagnostic Tests (RDTs)", "type": "diagnostics",
     "description": "HRP2/pLDH antigen-based point-of-care tests", "effectiveness": 0.95},
    {"name": "Microscopy", "type": "diagnostics",
     "description": "Gold standard thick/thin blood smear", "effectiveness": 0.98},
    {"name": "Community Health Workers (CHWs)", "type": "health_systems",
     "description": "Community-level case management and referral", "effectiveness": 0.60},
]

# Plasmodium species
PARASITES = [
    {"name": "Plasmodium falciparum", "severity": "most lethal", "prevalence_africa": 0.99,
     "lifecycle": ["sporozoite", "liver schizont", "merozoite", "trophozoite", "schizont", "gametocyte"]},
    {"name": "Plasmodium vivax", "severity": "relapsing", "prevalence_africa": 0.01,
     "lifecycle": ["sporozoite", "hypnozoite", "merozoite", "trophozoite", "schizont", "gametocyte"]},
    {"name": "Plasmodium malariae", "severity": "chronic", "prevalence_africa": 0.005,
     "lifecycle": ["sporozoite", "merozoite", "trophozoite", "schizont", "gametocyte"]},
    {"name": "Plasmodium ovale", "severity": "relapsing mild", "prevalence_africa": 0.005,
     "lifecycle": ["sporozoite", "hypnozoite", "merozoite", "trophozoite", "schizont", "gametocyte"]},
    {"name": "Plasmodium knowlesi", "severity": "zoonotic dangerous", "prevalence_africa": 0.0,
     "lifecycle": ["sporozoite", "merozoite", "trophozoite", "schizont", "gametocyte"]},
]

# Environmental factors
ENVIRONMENTAL_FACTORS = [
    {"name": "Temperature (25-30°C optimal)", "type": "climate", "impact": "vector breeding and parasite development"},
    {"name": "Rainfall and humidity", "type": "climate", "impact": "breeding site availability"},
    {"name": "Altitude (<1500m high risk)", "type": "geography", "impact": "vector range limitation"},
    {"name": "Urbanization", "type": "socioeconomic", "impact": "reduced transmission in cities"},
    {"name": "Deforestation", "type": "environmental_change", "impact": "new breeding habitats"},
    {"name": "Climate change", "type": "climate", "impact": "expanding endemic zones to highlands"},
    {"name": "El Niño Southern Oscillation", "type": "climate", "impact": "epidemic risk in East Africa"},
    {"name": "Irrigation and dams", "type": "water_management", "impact": "permanent breeding sites"},
    {"name": "Population mobility", "type": "socioeconomic", "impact": "importation of cases"},
    {"name": "Poverty and housing quality", "type": "socioeconomic", "impact": "exposure risk"},
]


def make_poincare_embedding(name, category, depth, lat=0, lon=0, dim=128):
    h = hashlib.sha256(f"{category}/{name}".encode()).digest()
    cat_h = hashlib.sha256(category.encode()).digest()

    radius = max(0.02, min(0.95, depth))
    jitter = (h[0] / 255.0 - 0.5) * 0.02
    radius = max(0.02, min(0.95, radius + jitter))

    coords = []
    for i in range(dim):
        base = h[i % 32] / 255.0 - 0.5
        cat_val = cat_h[i % 32] / 255.0 - 0.5
        val = 0.5 * base + 0.3 * cat_val
        # Encode geo coords in first 4 dims if available
        if i == 0 and lat != 0:
            val += lat / 180.0 * 0.2
        elif i == 1 and lon != 0:
            val += lon / 360.0 * 0.2
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

    print(f"\n{'='*60}")
    print(f"  Malaria Atlas → NietzscheDB Ingestion")
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

    # === ROOT ===
    root_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, "malaria-atlas:root"))
    node_ids["Malaria Atlas"] = root_id
    all_nodes.append({
        "id": root_id,
        "content": json.dumps({"name": "Malaria Atlas Project", "type": "root",
                                "dataset": "malaria_atlas"}).encode('utf-8'),
        "node_type": "Concept", "energy": 1.0,
        "embedding": make_poincare_embedding("Malaria Atlas", "root", 0.02, dim=dim),
    })

    # === DOMAIN ROOT NODES ===
    domains = ["Geography", "Vectors", "Parasites", "Drugs", "Interventions", "Environment"]
    for d in domains:
        did = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"malaria-atlas:domain:{d}"))
        node_ids[d] = did
        all_nodes.append({
            "id": did,
            "content": json.dumps({"name": d, "type": "domain",
                                    "dataset": "malaria_atlas"}).encode('utf-8'),
            "node_type": "Concept", "energy": 0.95,
            "embedding": make_poincare_embedding(d, "domain", 0.06, dim=dim),
        })
        edges.append({"from": root_id, "to": did, "type": "Hierarchical", "weight": 1.0})

    # === GEOGRAPHIC DATA ===
    for region_name, region_data in MALARIA_GEO.items():
        rid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"malaria-atlas:region:{region_name}"))
        node_ids[region_name] = rid
        all_nodes.append({
            "id": rid,
            "content": json.dumps({"name": region_name, "type": "region",
                                    "dataset": "malaria_atlas"}).encode('utf-8'),
            "node_type": "Concept", "energy": 0.9,
            "embedding": make_poincare_embedding(region_name, "geography", 0.10, dim=dim),
        })
        edges.append({"from": node_ids["Geography"], "to": rid, "type": "Hierarchical", "weight": 1.0})

        for country_name, country_data in region_data["countries"].items():
            cid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"malaria-atlas:country:{country_name}"))
            node_ids[country_name] = cid

            content = {
                "name": country_name, "type": "country",
                "iso": country_data["iso"],
                "lat": country_data["lat"], "lon": country_data["lon"],
                "prevalence_pf": country_data["prevalence_pf"],
                "incidence_per_1000": country_data["incidence_per_1000"],
                "deaths_2022": country_data["deaths_2022"],
                "population_at_risk": country_data["population_at_risk"],
                "itn_coverage": country_data["itn_coverage"],
                "irs_coverage": country_data["irs_coverage"],
                "dataset": "malaria_atlas",
            }
            is_focus = country_name in ["Angola", "São Tomé e Príncipe"]
            energy = 0.95 if is_focus else 0.7

            all_nodes.append({
                "id": cid,
                "content": json.dumps(content).encode('utf-8'),
                "node_type": "Concept", "energy": energy,
                "embedding": make_poincare_embedding(country_name, "geography",
                    country_data["depth"], country_data["lat"], country_data["lon"], dim),
            })
            edges.append({"from": rid, "to": cid, "type": "Hierarchical", "weight": 0.9})

            # Provinces
            for prov in country_data.get("provinces", []):
                pid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"malaria-atlas:province:{country_name}/{prov['name']}"))
                node_ids[prov["name"]] = pid
                prov_content = {
                    "name": prov["name"], "type": "province",
                    "country": country_name,
                    "lat": prov["lat"], "lon": prov["lon"],
                    "prevalence": prov["prevalence"],
                    "urban": prov["urban"],
                    "dataset": "malaria_atlas",
                }
                all_nodes.append({
                    "id": pid,
                    "content": json.dumps(prov_content).encode('utf-8'),
                    "node_type": "Semantic", "energy": 0.6,
                    "embedding": make_poincare_embedding(prov["name"], "geography",
                        0.30, prov["lat"], prov["lon"], dim),
                })
                edges.append({"from": cid, "to": pid, "type": "Hierarchical", "weight": 0.8})

    # === VECTOR SPECIES ===
    for vec in VECTOR_SPECIES:
        vid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"malaria-atlas:vector:{vec['name']}"))
        node_ids[vec["name"]] = vid
        content = {
            "name": vec["name"], "type": "vector_species",
            "region": vec["region"], "behaviour": vec["behaviour"],
            "insecticide_resistance": vec["resistance"],
            "primary_vector": vec["primary"],
            "dataset": "malaria_atlas",
        }
        all_nodes.append({
            "id": vid,
            "content": json.dumps(content).encode('utf-8'),
            "node_type": "Semantic", "energy": 0.7 if vec["primary"] else 0.5,
            "embedding": make_poincare_embedding(vec["name"], "vectors", 0.25, dim=dim),
        })
        edges.append({"from": node_ids["Vectors"], "to": vid, "type": "Hierarchical", "weight": 0.9})

    # === PARASITES ===
    for par in PARASITES:
        pid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"malaria-atlas:parasite:{par['name']}"))
        node_ids[par["name"]] = pid
        content = {
            "name": par["name"], "type": "parasite",
            "severity": par["severity"],
            "prevalence_africa": par["prevalence_africa"],
            "lifecycle_stages": par["lifecycle"],
            "dataset": "malaria_atlas",
        }
        all_nodes.append({
            "id": pid,
            "content": json.dumps(content).encode('utf-8'),
            "node_type": "Concept", "energy": 0.85,
            "embedding": make_poincare_embedding(par["name"], "parasites", 0.20, dim=dim),
        })
        edges.append({"from": node_ids["Parasites"], "to": pid, "type": "Hierarchical", "weight": 0.9})

        # Lifecycle stages
        for stage in par["lifecycle"]:
            stage_key = f"{par['name']}_{stage}"
            sid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"malaria-atlas:stage:{stage_key}"))
            node_ids[stage_key] = sid
            all_nodes.append({
                "id": sid,
                "content": json.dumps({"name": stage, "parasite": par["name"],
                    "type": "lifecycle_stage", "dataset": "malaria_atlas"}).encode('utf-8'),
                "node_type": "Semantic", "energy": 0.5,
                "embedding": make_poincare_embedding(stage_key, "parasites", 0.35, dim=dim),
            })
            edges.append({"from": pid, "to": sid, "type": "Hierarchical", "weight": 0.7})

    # === ANTIMALARIALS ===
    for drug in ANTIMALARIALS:
        did = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"malaria-atlas:drug:{drug['name']}"))
        node_ids[drug["name"]] = did
        content = {
            "name": drug["name"], "type": "antimalarial",
            "drug_class": drug["type"],
            "first_line": drug["first_line"],
            "resistance_markers": drug["resistance_markers"],
            "countries_with_resistance": drug["countries_resistant"],
            "dataset": "malaria_atlas",
        }
        all_nodes.append({
            "id": did,
            "content": json.dumps(content).encode('utf-8'),
            "node_type": "Semantic", "energy": 0.75,
            "embedding": make_poincare_embedding(drug["name"], "drugs", 0.25, dim=dim),
        })
        edges.append({"from": node_ids["Drugs"], "to": did, "type": "Hierarchical", "weight": 0.9})

        # Link to resistant countries
        for country in drug["countries_resistant"]:
            if country in node_ids:
                edges.append({"from": did, "to": node_ids[country],
                              "type": "Association", "weight": 0.6})

    # === INTERVENTIONS ===
    for itv in INTERVENTIONS:
        iid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"malaria-atlas:intervention:{itv['name']}"))
        node_ids[itv["name"]] = iid
        content = {
            "name": itv["name"], "type": "intervention",
            "intervention_type": itv["type"],
            "description": itv["description"],
            "effectiveness": itv["effectiveness"],
            "dataset": "malaria_atlas",
        }
        all_nodes.append({
            "id": iid,
            "content": json.dumps(content).encode('utf-8'),
            "node_type": "Semantic", "energy": 0.7,
            "embedding": make_poincare_embedding(itv["name"], "interventions", 0.25, dim=dim),
        })
        edges.append({"from": node_ids["Interventions"], "to": iid, "type": "Hierarchical", "weight": 0.9})

    # === ENVIRONMENTAL FACTORS ===
    for env in ENVIRONMENTAL_FACTORS:
        eid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"malaria-atlas:env:{env['name']}"))
        node_ids[env["name"]] = eid
        content = {
            "name": env["name"], "type": "environmental_factor",
            "factor_type": env["type"], "impact": env["impact"],
            "dataset": "malaria_atlas",
        }
        all_nodes.append({
            "id": eid,
            "content": json.dumps(content).encode('utf-8'),
            "node_type": "Semantic", "energy": 0.6,
            "embedding": make_poincare_embedding(env["name"], "environment", 0.25, dim=dim),
        })
        edges.append({"from": node_ids["Environment"], "to": eid, "type": "Hierarchical", "weight": 0.9})

    # === CROSS-DOMAIN EDGES ===
    # Vectors → Parasites (transmission)
    for vec in VECTOR_SPECIES:
        if "Africa" in vec["region"]:
            for par in ["Plasmodium falciparum", "Plasmodium malariae", "Plasmodium ovale"]:
                if vec["name"] in node_ids and par in node_ids:
                    edges.append({"from": node_ids[vec["name"]], "to": node_ids[par],
                                  "type": "Association", "weight": 0.8})

    # P. falciparum → ACT drugs
    pf_id = node_ids.get("Plasmodium falciparum")
    if pf_id:
        for drug in ANTIMALARIALS:
            if drug["type"] == "ACT" and drug["name"] in node_ids:
                edges.append({"from": pf_id, "to": node_ids[drug["name"]],
                              "type": "Association", "weight": 0.7})

    # Vectors → ITN/IRS interventions
    for vec in VECTOR_SPECIES:
        if vec["primary"] and vec["name"] in node_ids:
            for itv_name in ["Insecticide-Treated Nets (ITNs)", "Indoor Residual Spraying (IRS)"]:
                if itv_name in node_ids:
                    edges.append({"from": node_ids[vec["name"]], "to": node_ids[itv_name],
                                  "type": "Association", "weight": 0.6})

    # Insert all nodes
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
                    edge_type=eb["type"], weight=eb["weight"],
                    collection=collection)
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
                edge_type=eb["type"], weight=eb["weight"],
                collection=collection)
            reqs.append(req)
        try:
            stub.BatchInsertEdges(pb2.BatchInsertEdgesRequest(edges=reqs, collection=collection))
            edge_count += len(edge_batch)
        except grpc.RpcError:
            pass

    print(f"\n[+] Edges inserted: {edge_count}")

    # Count stats
    countries = sum(len(r["countries"]) for r in MALARIA_GEO.values())
    provinces = sum(len(c.get("provinces", []))
                    for r in MALARIA_GEO.values()
                    for c in r["countries"].values())

    print(f"\n{'='*60}")
    print(f"  MALARIA ATLAS INGESTION COMPLETE")
    print(f"{'='*60}")
    print(f"  Collection:       {collection}")
    print(f"  Total Nodes:      {inserted}")
    print(f"  Total Edges:      {edge_count}")
    print(f"  Countries:        {countries}")
    print(f"  Provinces:        {provinces} (Angola: 18, STP: 7)")
    print(f"  Vector species:   {len(VECTOR_SPECIES)}")
    print(f"  Parasites:        {len(PARASITES)}")
    print(f"  Antimalarials:    {len(ANTIMALARIALS)}")
    print(f"  Interventions:    {len(INTERVENTIONS)}")
    print(f"  Env. factors:     {len(ENVIRONMENTAL_FACTORS)}")
    print(f"{'='*60}")
    print(f"\n  The Poincaré ball now contains a complete malaria")
    print(f"  epidemiological atlas with Angola focus!")
    print(f"\n  Dashboard: http://{host.split(':')[0]}:8080")

    channel.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest Malaria Atlas into NietzscheDB")
    parser.add_argument("--host", default="localhost:50051")
    parser.add_argument("--collection", default="malaria_atlas")
    parser.add_argument("--metric", default="poincare")
    parser.add_argument("--dim", type=int, default=128)
    args = parser.parse_args()
    ingest(args.host, args.collection, args.metric, args.dim)
