#!/usr/bin/env python3
"""
Ingest OpenStreetMap Angola + São Tomé data into NietzscheDB.

Builds a geographic knowledge graph of Angola and STP with:
- Provinces, districts, municipalities
- Cities, towns, villages
- Hospitals, clinics, health centres (critical for malaria project)
- Roads, rivers, landmarks
- Points of interest

Uses embedded data (no API required) covering the most important locations.

Usage:
  python scripts/ingest_osm_angola.py [--host HOST:PORT] [--collection NAME]

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
# ANGOLA & STP GEOGRAPHIC DATA
# ═══════════════════════════════════════════════════════════════════════════

COUNTRIES = {
    "Angola": {
        "lat": -12.3, "lon": 17.5, "capital": "Luanda",
        "area_km2": 1246700, "population": 33000000,
        "provinces": {
            "Luanda": {
                "lat": -8.84, "lon": 13.23, "capital": "Luanda",
                "population": 8300000,
                "cities": [
                    {"name": "Luanda", "lat": -8.839, "lon": 13.234, "type": "capital", "pop": 2800000},
                    {"name": "Viana", "lat": -8.903, "lon": 13.377, "type": "city", "pop": 1600000},
                    {"name": "Cacuaco", "lat": -8.783, "lon": 13.367, "type": "city", "pop": 1200000},
                    {"name": "Cazenga", "lat": -8.828, "lon": 13.296, "type": "city", "pop": 900000},
                    {"name": "Belas", "lat": -8.977, "lon": 13.230, "type": "city", "pop": 1100000},
                    {"name": "Talatona", "lat": -8.953, "lon": 13.199, "type": "suburb", "pop": 200000},
                    {"name": "Ilha de Luanda", "lat": -8.798, "lon": 13.237, "type": "district", "pop": 50000},
                ],
                "health_facilities": [
                    {"name": "Hospital Josina Machel", "lat": -8.833, "lon": 13.230, "type": "hospital", "level": "central"},
                    {"name": "Hospital Américo Boavida", "lat": -8.838, "lon": 13.243, "type": "hospital", "level": "central"},
                    {"name": "Hospital Militar Principal", "lat": -8.837, "lon": 13.233, "type": "hospital", "level": "central"},
                    {"name": "Hospital Pediátrico David Bernardino", "lat": -8.830, "lon": 13.238, "type": "hospital", "level": "paediatric"},
                    {"name": "Maternidade Lucrécia Paím", "lat": -8.836, "lon": 13.237, "type": "hospital", "level": "maternity"},
                    {"name": "Clínica Sagrada Esperança", "lat": -8.835, "lon": 13.225, "type": "clinic", "level": "private"},
                    {"name": "Clínica Girassol", "lat": -8.844, "lon": 13.229, "type": "clinic", "level": "private"},
                    {"name": "Hospital Geral de Luanda (Prenda)", "lat": -8.854, "lon": 13.237, "type": "hospital", "level": "general"},
                    {"name": "Centro de Saúde do Cazenga", "lat": -8.828, "lon": 13.300, "type": "health_centre", "level": "primary"},
                    {"name": "Centro de Saúde de Viana", "lat": -8.907, "lon": 13.377, "type": "health_centre", "level": "primary"},
                ],
                "landmarks": [
                    {"name": "Fortaleza de São Miguel", "lat": -8.834, "lon": 13.229, "type": "historic"},
                    {"name": "Marginal de Luanda", "lat": -8.819, "lon": 13.243, "type": "promenade"},
                    {"name": "Aeroporto Internacional 4 de Fevereiro", "lat": -8.854, "lon": 13.232, "type": "airport"},
                    {"name": "Porto de Luanda", "lat": -8.792, "lon": 13.254, "type": "port"},
                    {"name": "Universidade Agostinho Neto", "lat": -8.827, "lon": 13.244, "type": "university"},
                    {"name": "Mercado de São Paulo", "lat": -8.842, "lon": 13.246, "type": "market"},
                    {"name": "Estádio 11 de Novembro", "lat": -8.947, "lon": 13.211, "type": "stadium"},
                ],
            },
            "Benguela": {
                "lat": -12.58, "lon": 13.41, "capital": "Benguela",
                "population": 2200000,
                "cities": [
                    {"name": "Benguela", "lat": -12.579, "lon": 13.405, "type": "capital", "pop": 555000},
                    {"name": "Lobito", "lat": -12.348, "lon": 13.536, "type": "city", "pop": 400000},
                    {"name": "Catumbela", "lat": -12.432, "lon": 13.547, "type": "city", "pop": 150000},
                    {"name": "Baía Farta", "lat": -12.617, "lon": 13.197, "type": "town", "pop": 50000},
                ],
                "health_facilities": [
                    {"name": "Hospital Geral de Benguela", "lat": -12.579, "lon": 13.405, "type": "hospital", "level": "provincial"},
                    {"name": "Hospital do Lobito", "lat": -12.350, "lon": 13.536, "type": "hospital", "level": "municipal"},
                ],
                "landmarks": [
                    {"name": "Caminho de Ferro de Benguela", "lat": -12.350, "lon": 13.540, "type": "railway"},
                    {"name": "Porto do Lobito", "lat": -12.330, "lon": 13.550, "type": "port"},
                ],
            },
            "Huambo": {
                "lat": -12.78, "lon": 15.73, "capital": "Huambo",
                "population": 2400000,
                "cities": [
                    {"name": "Huambo", "lat": -12.776, "lon": 15.735, "type": "capital", "pop": 600000},
                    {"name": "Caála", "lat": -12.851, "lon": 15.562, "type": "city", "pop": 200000},
                    {"name": "Longonjo", "lat": -12.907, "lon": 15.250, "type": "town", "pop": 40000},
                ],
                "health_facilities": [
                    {"name": "Hospital Central do Huambo", "lat": -12.776, "lon": 15.735, "type": "hospital", "level": "central"},
                ],
                "landmarks": [],
            },
            "Huíla": {
                "lat": -14.92, "lon": 13.49, "capital": "Lubango",
                "population": 2600000,
                "cities": [
                    {"name": "Lubango", "lat": -14.918, "lon": 13.497, "type": "capital", "pop": 460000},
                    {"name": "Matala", "lat": -14.733, "lon": 15.233, "type": "town", "pop": 60000},
                    {"name": "Chibia", "lat": -15.067, "lon": 13.683, "type": "town", "pop": 30000},
                ],
                "health_facilities": [
                    {"name": "Hospital Central da Huíla", "lat": -14.918, "lon": 13.497, "type": "hospital", "level": "central"},
                ],
                "landmarks": [
                    {"name": "Serra da Leba", "lat": -15.041, "lon": 13.354, "type": "natural"},
                    {"name": "Cristo Rei do Lubango", "lat": -14.932, "lon": 13.481, "type": "monument"},
                    {"name": "Fenda da Tundavala", "lat": -14.991, "lon": 13.407, "type": "natural"},
                ],
            },
            "Cabinda": {
                "lat": -5.55, "lon": 12.19, "capital": "Cabinda",
                "population": 720000,
                "cities": [
                    {"name": "Cabinda", "lat": -5.556, "lon": 12.190, "type": "capital", "pop": 500000},
                ],
                "health_facilities": [
                    {"name": "Hospital Provincial de Cabinda", "lat": -5.556, "lon": 12.190, "type": "hospital", "level": "provincial"},
                ],
                "landmarks": [
                    {"name": "Floresta do Maiombe", "lat": -4.80, "lon": 12.30, "type": "natural"},
                ],
            },
            "Malanje": {
                "lat": -9.54, "lon": 16.34, "capital": "Malanje",
                "population": 1100000,
                "cities": [
                    {"name": "Malanje", "lat": -9.541, "lon": 16.341, "type": "capital", "pop": 350000},
                ],
                "health_facilities": [
                    {"name": "Hospital Provincial de Malanje", "lat": -9.541, "lon": 16.341, "type": "hospital", "level": "provincial"},
                ],
                "landmarks": [
                    {"name": "Quedas de Kalandula", "lat": -9.073, "lon": 15.898, "type": "natural"},
                    {"name": "Pedras Negras de Pungo Andongo", "lat": -9.675, "lon": 15.597, "type": "natural"},
                ],
            },
            "Lunda Norte": {
                "lat": -7.77, "lon": 20.41, "capital": "Dundo",
                "population": 960000,
                "cities": [
                    {"name": "Dundo", "lat": -7.380, "lon": 20.834, "type": "capital", "pop": 200000},
                    {"name": "Lucapa", "lat": -8.417, "lon": 20.750, "type": "town", "pop": 60000},
                ],
                "health_facilities": [
                    {"name": "Hospital Provincial do Dundo", "lat": -7.380, "lon": 20.834, "type": "hospital", "level": "provincial"},
                ],
                "landmarks": [],
            },
            "Lunda Sul": {
                "lat": -10.72, "lon": 20.39, "capital": "Saurimo",
                "population": 600000,
                "cities": [
                    {"name": "Saurimo", "lat": -9.660, "lon": 20.399, "type": "capital", "pop": 200000},
                ],
                "health_facilities": [
                    {"name": "Hospital Provincial de Saurimo", "lat": -9.660, "lon": 20.399, "type": "hospital", "level": "provincial"},
                ],
                "landmarks": [],
            },
            "Moxico": {
                "lat": -13.43, "lon": 21.44, "capital": "Luena",
                "population": 850000,
                "cities": [
                    {"name": "Luena", "lat": -11.783, "lon": 19.917, "type": "capital", "pop": 180000},
                ],
                "health_facilities": [
                    {"name": "Hospital Provincial do Moxico", "lat": -11.783, "lon": 19.917, "type": "hospital", "level": "provincial"},
                ],
                "landmarks": [
                    {"name": "Nascente do Rio Zambeze", "lat": -11.35, "lon": 24.27, "type": "natural"},
                ],
            },
            "Uíge": {
                "lat": -7.61, "lon": 15.06, "capital": "Uíge",
                "population": 1600000,
                "cities": [
                    {"name": "Uíge", "lat": -7.609, "lon": 15.056, "type": "capital", "pop": 300000},
                    {"name": "Negage", "lat": -7.767, "lon": 15.267, "type": "town", "pop": 50000},
                ],
                "health_facilities": [
                    {"name": "Hospital Provincial do Uíge", "lat": -7.609, "lon": 15.056, "type": "hospital", "level": "provincial"},
                ],
                "landmarks": [],
            },
            "Zaire": {
                "lat": -6.27, "lon": 14.24, "capital": "M'banza-Kongo",
                "population": 650000,
                "cities": [
                    {"name": "M'banza-Kongo", "lat": -6.268, "lon": 14.240, "type": "capital", "pop": 150000},
                    {"name": "Soyo", "lat": -6.133, "lon": 12.383, "type": "city", "pop": 120000},
                ],
                "health_facilities": [
                    {"name": "Hospital Provincial do Zaire", "lat": -6.268, "lon": 14.240, "type": "hospital", "level": "provincial"},
                ],
                "landmarks": [
                    {"name": "M'banza-Kongo (UNESCO)", "lat": -6.268, "lon": 14.240, "type": "heritage"},
                ],
            },
            "Bengo": {
                "lat": -8.45, "lon": 13.56, "capital": "Caxito",
                "population": 400000,
                "cities": [
                    {"name": "Caxito", "lat": -8.583, "lon": 13.650, "type": "capital", "pop": 80000},
                ],
                "health_facilities": [
                    {"name": "Hospital Provincial do Bengo", "lat": -8.583, "lon": 13.650, "type": "hospital", "level": "provincial"},
                ],
                "landmarks": [
                    {"name": "Parque Nacional da Quiçama", "lat": -9.27, "lon": 13.83, "type": "national_park"},
                ],
            },
            "Kwanza Norte": {
                "lat": -9.17, "lon": 14.97, "capital": "N'dalatando",
                "population": 500000,
                "cities": [
                    {"name": "N'dalatando", "lat": -9.298, "lon": 14.912, "type": "capital", "pop": 80000},
                ],
                "health_facilities": [
                    {"name": "Hospital Provincial de Kwanza Norte", "lat": -9.298, "lon": 14.912, "type": "hospital", "level": "provincial"},
                ],
                "landmarks": [],
            },
            "Kwanza Sul": {
                "lat": -10.55, "lon": 14.86, "capital": "Sumbe",
                "population": 2100000,
                "cities": [
                    {"name": "Sumbe", "lat": -11.207, "lon": 13.842, "type": "capital", "pop": 80000},
                    {"name": "Porto Amboim", "lat": -10.722, "lon": 13.765, "type": "town", "pop": 40000},
                    {"name": "Gabela", "lat": -10.850, "lon": 14.367, "type": "town", "pop": 30000},
                ],
                "health_facilities": [
                    {"name": "Hospital Provincial de Kwanza Sul", "lat": -11.207, "lon": 13.842, "type": "hospital", "level": "provincial"},
                ],
                "landmarks": [],
            },
            "Cunene": {
                "lat": -16.27, "lon": 16.14, "capital": "Ondjiva",
                "population": 1100000,
                "cities": [
                    {"name": "Ondjiva", "lat": -17.067, "lon": 15.733, "type": "capital", "pop": 100000},
                    {"name": "Santa Clara", "lat": -17.45, "lon": 15.45, "type": "border_town", "pop": 30000},
                ],
                "health_facilities": [
                    {"name": "Hospital Provincial do Cunene", "lat": -17.067, "lon": 15.733, "type": "hospital", "level": "provincial"},
                ],
                "landmarks": [],
            },
            "Namibe": {
                "lat": -15.19, "lon": 12.15, "capital": "Moçâmedes",
                "population": 550000,
                "cities": [
                    {"name": "Moçâmedes (Namibe)", "lat": -15.196, "lon": 12.152, "type": "capital", "pop": 250000},
                    {"name": "Tômbua", "lat": -15.800, "lon": 11.867, "type": "town", "pop": 30000},
                ],
                "health_facilities": [
                    {"name": "Hospital Provincial do Namibe", "lat": -15.196, "lon": 12.152, "type": "hospital", "level": "provincial"},
                ],
                "landmarks": [
                    {"name": "Deserto do Namibe", "lat": -15.5, "lon": 12.0, "type": "natural"},
                    {"name": "Welwitschia mirabilis habitat", "lat": -15.3, "lon": 12.2, "type": "natural"},
                ],
            },
            "Cuando Cubango": {
                "lat": -15.60, "lon": 18.49, "capital": "Menongue",
                "population": 600000,
                "cities": [
                    {"name": "Menongue", "lat": -14.667, "lon": 17.683, "type": "capital", "pop": 150000},
                ],
                "health_facilities": [
                    {"name": "Hospital Provincial do Cuando Cubango", "lat": -14.667, "lon": 17.683, "type": "hospital", "level": "provincial"},
                ],
                "landmarks": [],
            },
            "Bié": {
                "lat": -12.39, "lon": 17.67, "capital": "Kuito",
                "population": 1500000,
                "cities": [
                    {"name": "Kuito", "lat": -12.383, "lon": 16.933, "type": "capital", "pop": 250000},
                    {"name": "Camacupa", "lat": -12.017, "lon": 17.483, "type": "town", "pop": 30000},
                ],
                "health_facilities": [
                    {"name": "Hospital Provincial do Bié", "lat": -12.383, "lon": 16.933, "type": "hospital", "level": "provincial"},
                ],
                "landmarks": [],
            },
        },
        "rivers": [
            {"name": "Rio Kwanza", "source_lat": -11.35, "source_lon": 17.43, "mouth_lat": -9.35, "mouth_lon": 13.15, "length_km": 960},
            {"name": "Rio Cunene", "source_lat": -12.43, "source_lon": 15.92, "mouth_lat": -17.27, "mouth_lon": 11.77, "length_km": 1050},
            {"name": "Rio Zambeze (nascente)", "source_lat": -11.35, "source_lon": 24.27, "mouth_lat": -11.35, "mouth_lon": 24.27, "length_km": 2574},
            {"name": "Rio Cubango/Okavango", "source_lat": -12.48, "source_lon": 16.40, "mouth_lat": -18.65, "mouth_lon": 22.75, "length_km": 1600},
            {"name": "Rio Catumbela", "source_lat": -12.20, "source_lon": 15.10, "mouth_lat": -12.43, "mouth_lon": 13.55, "length_km": 240},
            {"name": "Rio Lucala", "source_lat": -10.20, "source_lon": 15.80, "mouth_lat": -9.35, "mouth_lon": 14.30, "length_km": 330},
        ],
    },
    "São Tomé e Príncipe": {
        "lat": 0.19, "lon": 6.61, "capital": "São Tomé",
        "area_km2": 1001, "population": 220000,
        "provinces": {
            "Água Grande": {
                "lat": 0.30, "lon": 6.73, "capital": "São Tomé",
                "population": 80000,
                "cities": [
                    {"name": "São Tomé (cidade)", "lat": 0.337, "lon": 6.730, "type": "capital", "pop": 80000},
                ],
                "health_facilities": [
                    {"name": "Hospital Ayres de Menezes", "lat": 0.337, "lon": 6.730, "type": "hospital", "level": "central"},
                    {"name": "Centro de Saúde de Água Grande", "lat": 0.340, "lon": 6.725, "type": "health_centre", "level": "primary"},
                ],
                "landmarks": [
                    {"name": "Aeroporto Internacional de São Tomé", "lat": 0.378, "lon": 6.714, "type": "airport"},
                    {"name": "Forte de São Sebastião", "lat": 0.337, "lon": 6.730, "type": "historic"},
                    {"name": "Mercado Municipal de São Tomé", "lat": 0.336, "lon": 6.729, "type": "market"},
                ],
            },
            "Mé-Zóchi": {
                "lat": 0.28, "lon": 6.63, "capital": "Trindade",
                "population": 50000,
                "cities": [
                    {"name": "Trindade", "lat": 0.297, "lon": 6.685, "type": "capital", "pop": 20000},
                ],
                "health_facilities": [
                    {"name": "Centro de Saúde de Mé-Zóchi", "lat": 0.297, "lon": 6.685, "type": "health_centre", "level": "primary"},
                ],
                "landmarks": [
                    {"name": "Jardim Botânico de Bom Sucesso", "lat": 0.283, "lon": 6.618, "type": "natural"},
                ],
            },
            "Cantagalo": {
                "lat": 0.23, "lon": 6.59, "capital": "Santana",
                "population": 18000,
                "cities": [
                    {"name": "Santana", "lat": 0.256, "lon": 6.741, "type": "capital", "pop": 10000},
                ],
                "health_facilities": [],
                "landmarks": [],
            },
            "Caué": {
                "lat": 0.13, "lon": 6.57, "capital": "São João dos Angolares",
                "population": 7000,
                "cities": [
                    {"name": "São João dos Angolares", "lat": 0.178, "lon": 6.650, "type": "capital", "pop": 3000},
                ],
                "health_facilities": [],
                "landmarks": [
                    {"name": "Pico de São Tomé", "lat": 0.267, "lon": 6.576, "type": "natural"},
                ],
            },
            "Lembá": {
                "lat": 0.37, "lon": 6.55, "capital": "Neves",
                "population": 15000,
                "cities": [
                    {"name": "Neves", "lat": 0.370, "lon": 6.617, "type": "capital", "pop": 8000},
                ],
                "health_facilities": [],
                "landmarks": [],
            },
            "Lobata": {
                "lat": 0.35, "lon": 6.66, "capital": "Guadalupe",
                "population": 20000,
                "cities": [
                    {"name": "Guadalupe", "lat": 0.380, "lon": 6.669, "type": "capital", "pop": 8000},
                ],
                "health_facilities": [],
                "landmarks": [],
            },
            "Príncipe (RAP)": {
                "lat": 1.62, "lon": 7.40, "capital": "Santo António",
                "population": 8000,
                "cities": [
                    {"name": "Santo António do Príncipe", "lat": 1.639, "lon": 7.418, "type": "capital", "pop": 3000},
                ],
                "health_facilities": [
                    {"name": "Hospital do Príncipe", "lat": 1.639, "lon": 7.418, "type": "hospital", "level": "district"},
                ],
                "landmarks": [
                    {"name": "Reserva Biosfera da UNESCO (Príncipe)", "lat": 1.62, "lon": 7.40, "type": "natural"},
                ],
            },
        },
        "rivers": [],
    },
}


def make_poincare_embedding(name, category, depth, lat=0, lon=0, dim=128):
    h = hashlib.sha256(f"osm:{category}/{name}".encode()).digest()
    cat_h = hashlib.sha256(f"osm:{category}".encode()).digest()

    radius = max(0.02, min(0.95, depth))
    jitter = (h[0] / 255.0 - 0.5) * 0.02
    radius = max(0.02, min(0.95, radius + jitter))

    coords = []
    for i in range(dim):
        base = h[i % 32] / 255.0 - 0.5
        cat_val = cat_h[i % 32] / 255.0 - 0.5
        val = 0.5 * base + 0.3 * cat_val
        # Encode geo in first dims
        if i == 0:
            val += lat / 90.0 * 0.3
        elif i == 1:
            val += lon / 180.0 * 0.3
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
    print(f"  OpenStreetMap Angola+STP → NietzscheDB")
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
    root_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, "osm:root"))
    node_ids["OSM Root"] = root_id
    all_nodes.append({
        "id": root_id,
        "content": json.dumps({"name": "OpenStreetMap Angola & STP",
                                "type": "root", "dataset": "osm_angola"}).encode('utf-8'),
        "node_type": "Concept", "energy": 1.0,
        "embedding": make_poincare_embedding("root", "root", 0.02, dim=dim),
    })

    stats = {"countries": 0, "provinces": 0, "cities": 0,
             "health_facilities": 0, "landmarks": 0, "rivers": 0}

    for country_name, country_data in COUNTRIES.items():
        cid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"osm:country:{country_name}"))
        node_ids[country_name] = cid
        stats["countries"] += 1

        content = {
            "name": country_name, "type": "country",
            "lat": country_data["lat"], "lon": country_data["lon"],
            "capital": country_data["capital"],
            "area_km2": country_data["area_km2"],
            "population": country_data["population"],
            "dataset": "osm_angola",
        }
        all_nodes.append({
            "id": cid,
            "content": json.dumps(content).encode('utf-8'),
            "node_type": "Concept", "energy": 0.95,
            "embedding": make_poincare_embedding(country_name, "country", 0.06,
                country_data["lat"], country_data["lon"], dim),
        })
        edges.append({"from": root_id, "to": cid, "type": "Hierarchical", "weight": 1.0})

        # Provinces
        for prov_name, prov_data in country_data["provinces"].items():
            pid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"osm:province:{country_name}/{prov_name}"))
            node_ids[f"{country_name}/{prov_name}"] = pid
            stats["provinces"] += 1

            content = {
                "name": prov_name, "type": "province",
                "country": country_name,
                "lat": prov_data["lat"], "lon": prov_data["lon"],
                "capital": prov_data["capital"],
                "population": prov_data["population"],
                "dataset": "osm_angola",
            }
            all_nodes.append({
                "id": pid,
                "content": json.dumps(content).encode('utf-8'),
                "node_type": "Concept", "energy": 0.8,
                "embedding": make_poincare_embedding(prov_name, "province", 0.15,
                    prov_data["lat"], prov_data["lon"], dim),
            })
            edges.append({"from": cid, "to": pid, "type": "Hierarchical", "weight": 0.9})

            # Cities
            for city in prov_data.get("cities", []):
                city_id = str(uuid.uuid5(uuid.NAMESPACE_DNS,
                    f"osm:city:{country_name}/{prov_name}/{city['name']}"))
                node_ids[city["name"]] = city_id
                stats["cities"] += 1

                content = {
                    "name": city["name"], "type": city["type"],
                    "province": prov_name, "country": country_name,
                    "lat": city["lat"], "lon": city["lon"],
                    "population": city.get("pop", 0),
                    "dataset": "osm_angola",
                }
                all_nodes.append({
                    "id": city_id,
                    "content": json.dumps(content).encode('utf-8'),
                    "node_type": "Semantic", "energy": 0.6,
                    "embedding": make_poincare_embedding(city["name"], "city", 0.28,
                        city["lat"], city["lon"], dim),
                })
                edges.append({"from": pid, "to": city_id, "type": "Hierarchical", "weight": 0.8})

            # Health facilities (CRITICAL for malaria project)
            for hf in prov_data.get("health_facilities", []):
                hf_id = str(uuid.uuid5(uuid.NAMESPACE_DNS,
                    f"osm:health:{country_name}/{prov_name}/{hf['name']}"))
                node_ids[hf["name"]] = hf_id
                stats["health_facilities"] += 1

                content = {
                    "name": hf["name"], "type": hf["type"],
                    "level": hf["level"],
                    "province": prov_name, "country": country_name,
                    "lat": hf["lat"], "lon": hf["lon"],
                    "malaria_relevant": True,
                    "dataset": "osm_angola",
                }
                all_nodes.append({
                    "id": hf_id,
                    "content": json.dumps(content).encode('utf-8'),
                    "node_type": "Semantic", "energy": 0.75,
                    "embedding": make_poincare_embedding(hf["name"], "health", 0.35,
                        hf["lat"], hf["lon"], dim),
                })
                edges.append({"from": pid, "to": hf_id, "type": "Hierarchical", "weight": 0.8})

                # Link to nearest city
                if prov_data.get("cities"):
                    nearest_city = min(prov_data["cities"],
                        key=lambda c: (c["lat"]-hf["lat"])**2 + (c["lon"]-hf["lon"])**2)
                    nc_id = node_ids.get(nearest_city["name"])
                    if nc_id:
                        edges.append({"from": hf_id, "to": nc_id,
                                      "type": "Association", "weight": 0.6})

            # Landmarks
            for lm in prov_data.get("landmarks", []):
                lm_id = str(uuid.uuid5(uuid.NAMESPACE_DNS,
                    f"osm:landmark:{country_name}/{prov_name}/{lm['name']}"))
                node_ids[lm["name"]] = lm_id
                stats["landmarks"] += 1

                content = {
                    "name": lm["name"], "type": lm["type"],
                    "province": prov_name, "country": country_name,
                    "lat": lm["lat"], "lon": lm["lon"],
                    "dataset": "osm_angola",
                }
                all_nodes.append({
                    "id": lm_id,
                    "content": json.dumps(content).encode('utf-8'),
                    "node_type": "Semantic", "energy": 0.5,
                    "embedding": make_poincare_embedding(lm["name"], "landmark", 0.35,
                        lm["lat"], lm["lon"], dim),
                })
                edges.append({"from": pid, "to": lm_id, "type": "Hierarchical", "weight": 0.7})

        # Rivers
        for river in country_data.get("rivers", []):
            rid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"osm:river:{river['name']}"))
            node_ids[river["name"]] = rid
            stats["rivers"] += 1

            content = {
                "name": river["name"], "type": "river",
                "country": country_name,
                "source_lat": river["source_lat"], "source_lon": river["source_lon"],
                "mouth_lat": river["mouth_lat"], "mouth_lon": river["mouth_lon"],
                "length_km": river["length_km"],
                "dataset": "osm_angola",
            }
            all_nodes.append({
                "id": rid,
                "content": json.dumps(content).encode('utf-8'),
                "node_type": "Semantic", "energy": 0.55,
                "embedding": make_poincare_embedding(river["name"], "river", 0.25,
                    river["source_lat"], river["source_lon"], dim),
            })
            edges.append({"from": cid, "to": rid, "type": "Hierarchical", "weight": 0.7})

    # Spatial proximity edges between provinces
    print(f"[*] Computing spatial proximity edges...")
    _prov_keys = set()
    for _cn, _cd in COUNTRIES.items():
        for _pn in _cd["provinces"]:
            _prov_keys.add(f"{_cn}/{_pn}")
    province_nodes = [(k, node_ids[k]) for k in _prov_keys if k in node_ids]
    for i, (k1, id1) in enumerate(province_nodes):
        country1, prov1 = k1.split('/', 1)
        prov1_data = COUNTRIES[country1]["provinces"][prov1]
        for j in range(i + 1, len(province_nodes)):
            k2, id2 = province_nodes[j]
            country2, prov2 = k2.split('/', 1)
            if country1 != country2:
                continue
            prov2_data = COUNTRIES[country2]["provinces"][prov2]
            dist = math.sqrt((prov1_data["lat"] - prov2_data["lat"])**2 +
                             (prov1_data["lon"] - prov2_data["lon"])**2)
            if dist < 4.0:  # ~400km threshold
                weight = max(0.3, 1.0 - dist / 5.0)
                edges.append({"from": id1, "to": id2,
                              "type": "Association", "weight": weight})

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

    print(f"\n{'='*60}")
    print(f"  OSM ANGOLA+STP INGESTION COMPLETE")
    print(f"{'='*60}")
    print(f"  Collection:         {collection}")
    print(f"  Total Nodes:        {inserted}")
    print(f"  Total Edges:        {edge_count}")
    print(f"  Countries:          {stats['countries']}")
    print(f"  Provinces:          {stats['provinces']}")
    print(f"  Cities/Towns:       {stats['cities']}")
    print(f"  Health Facilities:  {stats['health_facilities']}")
    print(f"  Landmarks:          {stats['landmarks']}")
    print(f"  Rivers:             {stats['rivers']}")
    print(f"{'='*60}")
    print(f"\n  The Poincaré ball now has geographic context for EVA!")
    print(f"  All 18 provinces of Angola + 7 districts of STP mapped.")
    print(f"  Health facilities tagged for malaria integration.")
    print(f"\n  Dashboard: http://{host.split(':')[0]}:8080")

    channel.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest OSM Angola+STP into NietzscheDB")
    parser.add_argument("--host", default="localhost:50051")
    parser.add_argument("--collection", default="osm_angola")
    parser.add_argument("--metric", default="poincare")
    parser.add_argument("--dim", type=int, default=128)
    args = parser.parse_args()
    ingest(args.host, args.collection, args.metric, args.dim)
