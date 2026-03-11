#!/usr/bin/env python3
"""
Insert Technology Knowledge Galaxies into NietzscheDB.

Creates hierarchical, interconnected knowledge structures that form
star/galaxy patterns in the Poincaré ball visualization.

Collections created:
  - cs_knowledge    (CS ontology: ~3K nodes, ~15K edges, poincare metric)

Usage:
  python scripts/insert_tech_galaxies.py [--host HOST:PORT] [--collection NAME]
"""

import grpc
import json
import uuid
import math
import time
import hashlib
import sys
import os
import random
import argparse

# Add proto path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'sdks', 'python'))

# We'll use raw gRPC since the proto is well-defined
from grpc_tools import protoc
import importlib

def ensure_proto_compiled(repo_root=None):
    """Compile proto if needed."""
    if repo_root is None:
        # Try common locations
        for candidate in [
            os.path.join(os.path.dirname(__file__), '..'),
            '/home/web2a/NietzscheDB',
            os.path.expanduser('~/NietzscheDB'),
        ]:
            if os.path.isdir(os.path.join(candidate, 'crates', 'nietzsche-api', 'proto')):
                repo_root = candidate
                break
        if repo_root is None:
            raise RuntimeError("Cannot find NietzscheDB repo root. Pass --repo-root.")

    proto_dir = os.path.join(repo_root, 'crates', 'nietzsche-api', 'proto')
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_gen')
    # If script is in /tmp, use /tmp/_gen
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
        # Create __init__.py
        with open(os.path.join(out_dir, '__init__.py'), 'w') as f:
            f.write('')
        print(f"[proto] Compiled to {out_dir}")

    sys.path.insert(0, out_dir)
    pb2 = importlib.import_module('nietzsche_pb2')
    pb2_grpc = importlib.import_module('nietzsche_pb2_grpc')
    return pb2, pb2_grpc


# ═══════════════════════════════════════════════════════════════════════════
# TECHNOLOGY KNOWLEDGE ONTOLOGY
# ═══════════════════════════════════════════════════════════════════════════

# Each "galaxy" is a major CS domain. Each "star" is a sub-domain.
# Nodes within stars are concepts, algorithms, tools, etc.

GALAXIES = {
    # ─── GALAXY 1: Artificial Intelligence ────────────────────────────────
    "Artificial Intelligence": {
        "depth": 0.05,  # near center of Poincaré ball
        "stars": {
            "Machine Learning": {
                "depth": 0.15,
                "concepts": [
                    "Supervised Learning", "Unsupervised Learning", "Semi-Supervised Learning",
                    "Reinforcement Learning", "Transfer Learning", "Meta-Learning",
                    "Ensemble Methods", "Random Forest", "Gradient Boosting", "XGBoost",
                    "Support Vector Machines", "Decision Trees", "K-Nearest Neighbors",
                    "Naive Bayes", "Logistic Regression", "Linear Regression",
                    "Principal Component Analysis", "t-SNE", "UMAP",
                    "K-Means Clustering", "DBSCAN", "Hierarchical Clustering",
                    "Cross-Validation", "Overfitting", "Regularization",
                    "Feature Engineering", "Feature Selection", "Dimensionality Reduction",
                    "Bias-Variance Tradeoff", "Hyperparameter Tuning", "AutoML",
                ]
            },
            "Deep Learning": {
                "depth": 0.15,
                "concepts": [
                    "Neural Networks", "Backpropagation", "Gradient Descent",
                    "Convolutional Neural Networks", "Recurrent Neural Networks", "LSTM",
                    "Transformer", "Attention Mechanism", "Self-Attention",
                    "GPT", "BERT", "T5", "LLaMA", "Claude", "Gemini",
                    "Diffusion Models", "Stable Diffusion", "DALL-E",
                    "Generative Adversarial Networks", "Variational Autoencoders",
                    "ResNet", "VGG", "EfficientNet", "Vision Transformer",
                    "Batch Normalization", "Dropout", "Layer Normalization",
                    "Adam Optimizer", "SGD", "Learning Rate Scheduling",
                    "Mixed Precision Training", "Quantization", "Knowledge Distillation",
                    "Neural Architecture Search", "Pruning", "Model Compression",
                ]
            },
            "Natural Language Processing": {
                "depth": 0.2,
                "concepts": [
                    "Tokenization", "Word Embeddings", "Word2Vec", "GloVe", "FastText",
                    "Named Entity Recognition", "Part-of-Speech Tagging",
                    "Sentiment Analysis", "Text Classification", "Machine Translation",
                    "Question Answering", "Text Summarization", "Text Generation",
                    "Chatbots", "Dialogue Systems", "Intent Recognition",
                    "Information Retrieval", "Search Engines", "TF-IDF", "BM25",
                    "Semantic Search", "Dense Retrieval", "RAG",
                    "Prompt Engineering", "Few-Shot Learning", "Chain of Thought",
                    "RLHF", "Constitutional AI", "Alignment",
                ]
            },
            "Computer Vision": {
                "depth": 0.2,
                "concepts": [
                    "Image Classification", "Object Detection", "Image Segmentation",
                    "YOLO", "Faster R-CNN", "Mask R-CNN", "SSD",
                    "Optical Character Recognition", "Face Recognition", "Pose Estimation",
                    "Image Generation", "Style Transfer", "Super Resolution",
                    "3D Computer Vision", "Depth Estimation", "Point Clouds",
                    "Video Analysis", "Action Recognition", "Optical Flow",
                    "Medical Imaging", "Autonomous Driving Vision", "Satellite Imagery",
                ]
            },
            "Robotics & Agents": {
                "depth": 0.25,
                "concepts": [
                    "Robot Operating System", "SLAM", "Path Planning",
                    "Inverse Kinematics", "Motion Planning", "Manipulation",
                    "Autonomous Vehicles", "Drone Navigation", "Swarm Intelligence",
                    "Multi-Agent Systems", "Agent Architectures", "BDI Agents",
                    "Reward Shaping", "Sim-to-Real Transfer", "Imitation Learning",
                ]
            },
        }
    },

    # ─── GALAXY 2: Systems & Infrastructure ───────────────────────────────
    "Systems & Infrastructure": {
        "depth": 0.05,
        "stars": {
            "Operating Systems": {
                "depth": 0.15,
                "concepts": [
                    "Linux Kernel", "Windows NT", "macOS Darwin", "FreeBSD",
                    "Process Scheduling", "Memory Management", "Virtual Memory",
                    "File Systems", "ext4", "Btrfs", "ZFS", "NTFS",
                    "System Calls", "Interrupts", "Device Drivers",
                    "Containers", "Docker", "Podman", "LXC",
                    "Virtualization", "KVM", "Xen", "Hyper-V",
                    "Real-Time OS", "FreeRTOS", "Zephyr", "VxWorks",
                ]
            },
            "Distributed Systems": {
                "depth": 0.15,
                "concepts": [
                    "CAP Theorem", "Consistency Models", "Eventual Consistency",
                    "Consensus Algorithms", "Raft", "Paxos", "PBFT",
                    "Distributed Hash Tables", "Chord", "Kademlia",
                    "MapReduce", "Apache Spark", "Apache Flink",
                    "Message Queues", "Apache Kafka", "RabbitMQ", "NATS",
                    "Service Mesh", "Istio", "Linkerd", "Envoy",
                    "Load Balancing", "Consistent Hashing", "Sharding",
                    "Replication", "Leader Election", "Quorum",
                    "Distributed Tracing", "OpenTelemetry", "Jaeger",
                ]
            },
            "Cloud Computing": {
                "depth": 0.2,
                "concepts": [
                    "AWS", "Google Cloud", "Azure", "DigitalOcean",
                    "Kubernetes", "Helm", "Kustomize", "ArgoCD",
                    "Serverless", "AWS Lambda", "Cloud Functions", "Azure Functions",
                    "Infrastructure as Code", "Terraform", "Pulumi", "CloudFormation",
                    "CI/CD", "GitHub Actions", "GitLab CI", "Jenkins",
                    "Container Registry", "Artifact Management",
                    "Auto Scaling", "Spot Instances", "Reserved Instances",
                    "Cloud Storage", "S3", "GCS", "Blob Storage",
                    "CDN", "CloudFront", "Cloud CDN", "Fastly",
                ]
            },
            "Networking": {
                "depth": 0.2,
                "concepts": [
                    "TCP/IP", "UDP", "HTTP/2", "HTTP/3", "QUIC",
                    "WebSockets", "gRPC", "GraphQL", "REST API",
                    "DNS", "BGP", "OSPF", "MPLS",
                    "TLS/SSL", "mTLS", "Certificate Authority",
                    "IPv4", "IPv6", "NAT", "Subnetting",
                    "Software Defined Networking", "OpenFlow", "P4",
                    "5G", "WiFi 6", "Bluetooth LE", "LoRaWAN",
                    "VPN", "WireGuard", "IPSec", "Tor",
                ]
            },
        }
    },

    # ─── GALAXY 3: Data & Databases ───────────────────────────────────────
    "Data & Databases": {
        "depth": 0.05,
        "stars": {
            "Relational Databases": {
                "depth": 0.15,
                "concepts": [
                    "PostgreSQL", "MySQL", "SQLite", "Oracle DB", "SQL Server",
                    "ACID Properties", "Transactions", "Isolation Levels",
                    "Indexes", "B-Tree", "Hash Index", "GIN Index", "GiST",
                    "Query Optimization", "Query Plans", "Cost-Based Optimizer",
                    "Normalization", "Denormalization", "Schema Design",
                    "Stored Procedures", "Triggers", "Views", "Materialized Views",
                    "Connection Pooling", "PgBouncer", "Read Replicas",
                ]
            },
            "NoSQL Databases": {
                "depth": 0.15,
                "concepts": [
                    "MongoDB", "CouchDB", "DynamoDB", "Cassandra",
                    "Redis", "Memcached", "KeyDB",
                    "Neo4j", "ArangoDB", "JanusGraph", "NietzscheDB",
                    "Elasticsearch", "Solr", "Meilisearch", "Typesense",
                    "InfluxDB", "TimescaleDB", "QuestDB",
                    "Document Model", "Key-Value Store", "Column Family",
                    "Graph Database", "Time Series Database", "Vector Database",
                ]
            },
            "Vector Databases": {
                "depth": 0.2,
                "concepts": [
                    "NietzscheDB HNSW", "HNSW Algorithm", "IVF-Flat", "IVF-PQ",
                    "Product Quantization", "Scalar Quantization",
                    "Pinecone", "Weaviate", "Milvus", "Qdrant", "Chroma",
                    "Approximate Nearest Neighbors", "Exact kNN",
                    "Cosine Similarity", "Euclidean Distance", "Dot Product",
                    "Poincaré Distance", "Hyperbolic Embeddings", "Lorentz Model",
                    "CAGRA", "cuVS", "FAISS", "ScaNN", "Annoy",
                    "Embedding Models", "Sentence Transformers", "OpenAI Embeddings",
                ]
            },
            "Data Engineering": {
                "depth": 0.2,
                "concepts": [
                    "ETL", "ELT", "Data Pipelines", "Airflow", "Dagster", "Prefect",
                    "Data Warehouse", "Snowflake", "BigQuery", "Redshift",
                    "Data Lake", "Delta Lake", "Apache Iceberg", "Apache Hudi",
                    "Apache Parquet", "Apache Avro", "ORC",
                    "Data Catalog", "Data Lineage", "Data Quality",
                    "Stream Processing", "Apache Kafka Streams", "Apache Beam",
                    "CDC", "Debezium", "Change Data Capture",
                ]
            },
        }
    },

    # ─── GALAXY 4: Programming Languages & Paradigms ──────────────────────
    "Programming Languages": {
        "depth": 0.05,
        "stars": {
            "Systems Languages": {
                "depth": 0.15,
                "concepts": [
                    "Rust", "C", "C++", "Zig", "Nim",
                    "Ownership", "Borrowing", "Lifetimes", "Move Semantics",
                    "Memory Safety", "Buffer Overflow", "Use After Free",
                    "Zero-Cost Abstractions", "Inline Assembly",
                    "LLVM", "GCC", "Clang", "Compiler Design",
                    "Linking", "Static Libraries", "Dynamic Libraries",
                    "ABI", "Calling Conventions", "FFI",
                ]
            },
            "Application Languages": {
                "depth": 0.15,
                "concepts": [
                    "Python", "Java", "C#", "Kotlin", "Swift",
                    "Go", "TypeScript", "JavaScript", "Ruby", "PHP",
                    "Garbage Collection", "JIT Compilation", "AOT Compilation",
                    "Type Systems", "Static Typing", "Dynamic Typing",
                    "Generics", "Interfaces", "Traits", "Protocols",
                    "Async/Await", "Coroutines", "Green Threads",
                    "Package Managers", "Cargo", "npm", "pip", "Maven",
                ]
            },
            "Functional Languages": {
                "depth": 0.2,
                "concepts": [
                    "Haskell", "Elixir", "Erlang", "Clojure", "F#", "OCaml", "Scala",
                    "Pure Functions", "Immutability", "Referential Transparency",
                    "Monads", "Functors", "Applicatives",
                    "Pattern Matching", "Algebraic Data Types", "Sum Types",
                    "Higher-Order Functions", "Currying", "Partial Application",
                    "Lazy Evaluation", "Tail Call Optimization",
                    "Actor Model", "OTP", "BEAM VM", "Supervision Trees",
                ]
            },
            "Software Architecture": {
                "depth": 0.2,
                "concepts": [
                    "Microservices", "Monolith", "Modular Monolith",
                    "Event-Driven Architecture", "CQRS", "Event Sourcing",
                    "Domain-Driven Design", "Hexagonal Architecture", "Clean Architecture",
                    "Design Patterns", "Singleton", "Factory", "Observer", "Strategy",
                    "SOLID Principles", "DRY", "KISS", "YAGNI",
                    "API Gateway", "Backend for Frontend", "Strangler Fig Pattern",
                    "Circuit Breaker", "Bulkhead", "Retry Pattern", "Saga Pattern",
                ]
            },
        }
    },

    # ─── GALAXY 5: Security & Cryptography ────────────────────────────────
    "Security & Cryptography": {
        "depth": 0.05,
        "stars": {
            "Cryptography": {
                "depth": 0.15,
                "concepts": [
                    "AES", "ChaCha20", "RSA", "Elliptic Curve Cryptography",
                    "SHA-256", "SHA-3", "BLAKE3", "Argon2",
                    "Digital Signatures", "Ed25519", "ECDSA",
                    "Key Exchange", "Diffie-Hellman", "X25519",
                    "Zero-Knowledge Proofs", "zk-SNARKs", "zk-STARKs",
                    "Homomorphic Encryption", "Multi-Party Computation",
                    "Post-Quantum Cryptography", "Lattice-Based Crypto", "Kyber",
                ]
            },
            "Application Security": {
                "depth": 0.15,
                "concepts": [
                    "OWASP Top 10", "SQL Injection", "XSS", "CSRF",
                    "Authentication", "OAuth 2.0", "OpenID Connect", "JWT",
                    "Authorization", "RBAC", "ABAC", "Policy Engines",
                    "Web Application Firewall", "Rate Limiting", "CORS",
                    "Input Validation", "Output Encoding", "Parameterized Queries",
                    "Content Security Policy", "Subresource Integrity",
                    "Secrets Management", "HashiCorp Vault", "AWS KMS",
                ]
            },
            "Infrastructure Security": {
                "depth": 0.2,
                "concepts": [
                    "Firewalls", "IDS/IPS", "SIEM", "SOC",
                    "Penetration Testing", "Red Team", "Blue Team", "Purple Team",
                    "Vulnerability Scanning", "CVE", "CVSS",
                    "Network Segmentation", "Zero Trust Architecture",
                    "Container Security", "Supply Chain Security", "SBOM",
                    "Incident Response", "Forensics", "Threat Intelligence",
                ]
            },
        }
    },

    # ─── GALAXY 6: Web & Frontend ─────────────────────────────────────────
    "Web & Frontend": {
        "depth": 0.05,
        "stars": {
            "Frontend Frameworks": {
                "depth": 0.15,
                "concepts": [
                    "React", "Next.js", "Remix", "Vue.js", "Nuxt",
                    "Angular", "Svelte", "SvelteKit", "Solid.js", "Qwik",
                    "Virtual DOM", "Incremental DOM", "Signals",
                    "Server-Side Rendering", "Static Site Generation",
                    "Hydration", "Islands Architecture", "Partial Hydration",
                    "State Management", "Redux", "Zustand", "Jotai", "Pinia",
                    "CSS-in-JS", "Tailwind CSS", "Styled Components",
                ]
            },
            "Web Standards": {
                "depth": 0.2,
                "concepts": [
                    "HTML5", "CSS3", "Web Components", "Shadow DOM",
                    "Web Workers", "Service Workers", "PWA",
                    "WebAssembly", "WASM", "WASI",
                    "WebGL", "WebGPU", "Canvas API", "SVG",
                    "Web Audio API", "WebRTC", "WebTransport",
                    "IndexedDB", "Local Storage", "Cache API",
                    "Accessibility", "ARIA", "WCAG", "Screen Readers",
                ]
            },
            "Mobile Development": {
                "depth": 0.2,
                "concepts": [
                    "React Native", "Flutter", "Swift UI", "Jetpack Compose",
                    "Kotlin Multiplatform", "Capacitor", "Tauri",
                    "iOS Development", "Android Development",
                    "App Store Optimization", "Push Notifications",
                    "Mobile Testing", "Detox", "Espresso", "XCTest",
                ]
            },
        }
    },

    # ─── GALAXY 7: Mathematics & Theory ───────────────────────────────────
    "Mathematics & CS Theory": {
        "depth": 0.05,
        "stars": {
            "Algorithms": {
                "depth": 0.15,
                "concepts": [
                    "Sorting Algorithms", "QuickSort", "MergeSort", "HeapSort", "TimSort",
                    "Graph Algorithms", "Dijkstra", "A*", "BFS", "DFS",
                    "Dynamic Programming", "Memoization", "Tabulation",
                    "Greedy Algorithms", "Backtracking", "Branch and Bound",
                    "String Algorithms", "KMP", "Rabin-Karp", "Aho-Corasick",
                    "Computational Geometry", "Convex Hull", "Voronoi Diagram",
                    "Approximation Algorithms", "Randomized Algorithms",
                    "Amortized Analysis", "Big O Notation", "Space Complexity",
                ]
            },
            "Data Structures": {
                "depth": 0.15,
                "concepts": [
                    "Arrays", "Linked Lists", "Stacks", "Queues", "Deques",
                    "Hash Tables", "Hash Maps", "Bloom Filters", "Cuckoo Hashing",
                    "Binary Search Trees", "AVL Trees", "Red-Black Trees",
                    "B-Trees", "B+ Trees", "LSM Trees", "Skip Lists",
                    "Heaps", "Fibonacci Heaps", "Pairing Heaps",
                    "Tries", "Radix Trees", "Patricia Trees",
                    "Graphs", "Adjacency List", "Adjacency Matrix",
                    "Disjoint Set", "Segment Trees", "Fenwick Trees",
                ]
            },
            "Hyperbolic Geometry": {
                "depth": 0.2,
                "concepts": [
                    "Poincaré Ball Model", "Poincaré Half-Plane",
                    "Hyperboloid Model", "Klein Model", "Lorentz Model",
                    "Hyperbolic Distance", "Geodesics", "Möbius Transformations",
                    "Exponential Map", "Logarithmic Map", "Parallel Transport",
                    "Hyperbolic Neural Networks", "Hyperbolic Embeddings",
                    "Tree-Likeness", "Gromov Hyperbolicity", "Curvature",
                    "Riemannian Manifolds", "Tangent Space", "Riemannian SGD",
                    "Busemann Functions", "Horospheres", "Ideal Boundary",
                ]
            },
            "Category Theory & Logic": {
                "depth": 0.25,
                "concepts": [
                    "Category", "Functor", "Natural Transformation",
                    "Monad Laws", "Kleisli Category", "Adjunction",
                    "Lambda Calculus", "Type Theory", "Dependent Types",
                    "Curry-Howard Correspondence", "Propositions as Types",
                    "Formal Verification", "Model Checking", "Theorem Proving",
                    "Coq", "Lean", "Agda", "Isabelle",
                    "P vs NP", "NP-Completeness", "Complexity Classes",
                ]
            },
        }
    },

    # ─── GALAXY 8: Blockchain & Web3 ──────────────────────────────────────
    "Blockchain & Web3": {
        "depth": 0.05,
        "stars": {
            "Consensus & Protocols": {
                "depth": 0.15,
                "concepts": [
                    "Bitcoin", "Ethereum", "Solana", "Polkadot", "Cosmos",
                    "Proof of Work", "Proof of Stake", "Delegated PoS",
                    "Merkle Trees", "Patricia Merkle Trie",
                    "Layer 2", "Rollups", "Optimistic Rollups", "ZK-Rollups",
                    "State Channels", "Plasma", "Sidechains",
                    "Sharding", "Cross-Chain Bridges", "Interoperability",
                ]
            },
            "Smart Contracts & DeFi": {
                "depth": 0.2,
                "concepts": [
                    "Solidity", "Rust Smart Contracts", "Move Language",
                    "EVM", "WASM Smart Contracts",
                    "DeFi", "AMM", "Uniswap", "Lending Protocols", "Aave",
                    "NFTs", "ERC-721", "ERC-1155",
                    "DAOs", "Governance Tokens", "Snapshot Voting",
                    "Oracles", "Chainlink", "Band Protocol",
                ]
            },
        }
    },
}

# Cross-galaxy edges (bridges between galaxies)
CROSS_EDGES = [
    # AI ↔ Data
    ("Transformer", "Vector Database", "enables"),
    ("Embedding Models", "Semantic Search", "powers"),
    ("RAG", "Vector Database", "uses"),
    ("Deep Learning", "GPU", "requires"),
    ("Neural Networks", "CUDA", "accelerated_by"),
    ("NietzscheDB HNSW", "Poincaré Distance", "uses"),
    ("NietzscheDB HNSW", "CAGRA", "accelerated_by"),
    ("Hyperbolic Embeddings", "Poincaré Ball Model", "uses"),
    ("Hyperbolic Embeddings", "NietzscheDB", "stored_in"),

    # Systems ↔ Cloud
    ("Kubernetes", "Docker", "orchestrates"),
    ("Kubernetes", "Microservices", "deploys"),
    ("Terraform", "AWS", "provisions"),
    ("Terraform", "Google Cloud", "provisions"),
    ("gRPC", "Protocol Buffers", "uses"),
    ("gRPC", "HTTP/2", "runs_on"),

    # Languages ↔ Systems
    ("Rust", "Memory Safety", "guarantees"),
    ("Rust", "LLVM", "compiles_with"),
    ("Go", "Kubernetes", "powers"),
    ("Python", "Machine Learning", "dominates"),
    ("TypeScript", "React", "types"),

    # Security ↔ Crypto
    ("Zero-Knowledge Proofs", "ZK-Rollups", "enables"),
    ("Elliptic Curve Cryptography", "Bitcoin", "secures"),
    ("TLS/SSL", "HTTPS", "provides"),
    ("OAuth 2.0", "JWT", "uses"),

    # Web ↔ Architecture
    ("React", "Virtual DOM", "uses"),
    ("WebAssembly", "Rust", "compiled_from"),
    ("WebGPU", "GPU", "accesses"),
    ("Next.js", "Server-Side Rendering", "implements"),

    # Theory ↔ Practice
    ("B-Trees", "PostgreSQL", "used_by"),
    ("LSM Trees", "RocksDB", "used_by"),
    ("HNSW Algorithm", "NietzscheDB HNSW", "implemented_by"),
    ("Dijkstra", "Path Planning", "used_in"),
    ("Dynamic Programming", "Reinforcement Learning", "foundation_of"),
    ("Big O Notation", "Query Optimization", "analyzes"),

    # Hyperbolic ↔ NietzscheDB
    ("Riemannian SGD", "Hyperbolic Embeddings", "optimizes"),
    ("Poincaré Ball Model", "Hyperbolic Distance", "defines"),
    ("Lorentz Model", "Poincaré Ball Model", "equivalent_to"),
    ("Möbius Transformations", "Poincaré Ball Model", "operates_on"),
    ("Curvature", "Hyperbolic Geometry", "characterizes"),
]


def make_embedding(galaxy_name, star_name, concept_name, depth, dim=3072):
    """Generate a deterministic embedding that clusters by topic.

    Uses a hash-based approach to create embeddings where:
    - Same galaxy → similar direction (first few dims)
    - Same star → closer cluster (mid dims)
    - Depth controls magnitude (Poincaré ball radius)
    """
    # Create base direction from galaxy hash
    galaxy_hash = hashlib.sha256(galaxy_name.encode()).digest()
    star_hash = hashlib.sha256(f"{galaxy_name}/{star_name}".encode()).digest()
    concept_hash = hashlib.sha256(f"{galaxy_name}/{star_name}/{concept_name}".encode()).digest()

    coords = []
    for i in range(dim):
        # Mix galaxy, star, and concept hashes with different weights
        g_val = galaxy_hash[i % 32] / 255.0 - 0.5
        s_val = star_hash[i % 32] / 255.0 - 0.5
        c_val = concept_hash[i % 32] / 255.0 - 0.5

        # Galaxy direction dominates (0.5), star adds clustering (0.3), concept adds uniqueness (0.2)
        val = 0.5 * g_val + 0.3 * s_val + 0.2 * c_val

        # Add some variation based on position
        byte_idx = (i * 7 + 13) % 32
        val += 0.1 * (concept_hash[byte_idx] / 255.0 - 0.5)

        coords.append(val)

    # Normalize to unit sphere, then scale by depth (Poincaré ball radius)
    norm = math.sqrt(sum(c*c for c in coords))
    if norm > 0:
        # Add small random jitter to depth for visual spread
        jitter = (concept_hash[0] / 255.0 - 0.5) * 0.1
        target_radius = min(depth + jitter, 0.95)  # stay inside Poincaré ball
        target_radius = max(target_radius, 0.02)
        coords = [c / norm * target_radius for c in coords]

    return coords


def make_edge_request(pb2, **kwargs):
    """Create InsertEdgeRequest handling 'from' reserved keyword."""
    from_val = kwargs.pop('from_node')
    req = pb2.InsertEdgeRequest(**kwargs)
    setattr(req, 'from', from_val)
    return req


def insert_galaxies(host, collection, metric="poincare", dim=3072):
    """Insert all technology galaxies into NietzscheDB."""

    pb2, pb2_grpc = ensure_proto_compiled()

    channel = grpc.insecure_channel(host, options=[
        ('grpc.max_send_message_length', 256 * 1024 * 1024),
        ('grpc.max_receive_message_length', 256 * 1024 * 1024),
    ])
    stub = pb2_grpc.NietzscheDBStub(channel)

    # 1. Create collection
    print(f"\n{'='*60}")
    print(f"  NietzscheDB Technology Galaxy Ingestion")
    print(f"  Host: {host}")
    print(f"  Collection: {collection}")
    print(f"  Metric: {metric}")
    print(f"  Dim: {dim}")
    print(f"{'='*60}\n")

    try:
        resp = stub.CreateCollection(pb2.CreateCollectionRequest(
            collection=collection, dim=dim, metric=metric
        ))
        if resp.created:
            print(f"[+] Collection '{collection}' created (dim={dim}, metric={metric})")
        else:
            print(f"[=] Collection '{collection}' already exists")
    except grpc.RpcError as e:
        print(f"[!] CreateCollection: {e.details()}")

    # 2. Insert all nodes
    node_ids = {}  # concept_name → uuid
    total_nodes = 0
    total_edges = 0
    batch_nodes = []

    for galaxy_name, galaxy in GALAXIES.items():
        galaxy_depth = galaxy["depth"]

        # Insert galaxy root node
        galaxy_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"galaxy:{galaxy_name}"))
        node_ids[galaxy_name] = galaxy_id

        content = json.dumps({
            "name": galaxy_name,
            "type": "galaxy",
            "level": "root",
            "description": f"Major CS domain: {galaxy_name}",
        }).encode()

        embedding = make_embedding(galaxy_name, "", galaxy_name, galaxy_depth, dim)

        batch_nodes.append(pb2.InsertNodeRequest(
            id=galaxy_id,
            embedding=pb2.PoincareVector(coords=embedding, dim=dim),
            content=content,
            node_type="Concept",
            energy=1.0,
            collection=collection,
        ))
        total_nodes += 1

        for star_name, star in galaxy["stars"].items():
            star_depth = star["depth"]

            # Insert star node
            star_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"star:{galaxy_name}/{star_name}"))
            node_ids[star_name] = star_id

            content = json.dumps({
                "name": star_name,
                "type": "star",
                "galaxy": galaxy_name,
                "level": "domain",
                "description": f"Sub-domain of {galaxy_name}: {star_name}",
            }).encode()

            embedding = make_embedding(galaxy_name, star_name, star_name, star_depth, dim)

            batch_nodes.append(pb2.InsertNodeRequest(
                id=star_id,
                embedding=pb2.PoincareVector(coords=embedding, dim=dim),
                content=content,
                node_type="Concept",
                energy=0.9,
                collection=collection,
            ))
            total_nodes += 1

            # Insert concept nodes
            for concept_name in star["concepts"]:
                concept_depth = star_depth + 0.1 + random.random() * 0.15
                concept_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"concept:{concept_name}"))

                # Avoid duplicate IDs (same concept in multiple stars)
                if concept_name in node_ids:
                    continue

                node_ids[concept_name] = concept_id

                content = json.dumps({
                    "name": concept_name,
                    "type": "concept",
                    "star": star_name,
                    "galaxy": galaxy_name,
                    "level": "leaf",
                }).encode()

                embedding = make_embedding(galaxy_name, star_name, concept_name, concept_depth, dim)

                batch_nodes.append(pb2.InsertNodeRequest(
                    id=concept_id,
                    embedding=pb2.PoincareVector(coords=embedding, dim=dim),
                    content=content,
                    node_type="Semantic",
                    energy=0.8,
                    collection=collection,
                ))
                total_nodes += 1

    # Batch insert nodes
    print(f"\n[*] Inserting {total_nodes} nodes in batches of 50...")
    inserted = 0
    for i in range(0, len(batch_nodes), 50):
        batch = batch_nodes[i:i+50]
        try:
            resp = stub.BatchInsertNodes(pb2.BatchInsertNodesRequest(
                nodes=batch, collection=collection
            ))
            inserted += resp.inserted
            pct = (i + len(batch)) / len(batch_nodes) * 100
            print(f"  [{pct:5.1f}%] Inserted {resp.inserted} nodes (total: {inserted})", end='\r')
        except grpc.RpcError as e:
            print(f"\n  [!] Batch {i//50}: {e.details()}")
            # Fallback to individual inserts
            for node in batch:
                try:
                    stub.InsertNode(node)
                    inserted += 1
                except grpc.RpcError:
                    pass

    print(f"\n[+] Nodes inserted: {inserted}/{total_nodes}")

    # 3. Insert hierarchical edges (galaxy → star → concept)
    print(f"\n[*] Creating hierarchical edges...")
    edge_count = 0

    for galaxy_name, galaxy in GALAXIES.items():
        galaxy_id = node_ids.get(galaxy_name)
        if not galaxy_id:
            continue

        for star_name, star in galaxy["stars"].items():
            star_id = node_ids.get(star_name)
            if not star_id:
                continue

            # Galaxy → Star edge
            try:
                stub.InsertEdge(make_edge_request(pb2,
                    from_node=galaxy_id, to=star_id,
                    edge_type="Hierarchical", weight=1.0,
                    collection=collection
                ))
                edge_count += 1
            except grpc.RpcError:
                pass

            # Star → Concept edges
            for concept_name in star["concepts"]:
                concept_id = node_ids.get(concept_name)
                if not concept_id:
                    continue
                try:
                    stub.InsertEdge(make_edge_request(pb2,
                        from_node=star_id, to=concept_id,
                        edge_type="Hierarchical", weight=0.8,
                        collection=collection
                    ))
                    edge_count += 1
                except grpc.RpcError:
                    pass

            # Intra-star edges (concepts within same star are associated)
            concepts_in_star = [c for c in star["concepts"] if c in node_ids]
            for j in range(len(concepts_in_star)):
                for k in range(j+1, min(j+4, len(concepts_in_star))):
                    try:
                        stub.InsertEdge(make_edge_request(pb2,
                            from_node=node_ids[concepts_in_star[j]],
                            to=node_ids[concepts_in_star[k]],
                            edge_type="Association", weight=0.6,
                            collection=collection
                        ))
                        edge_count += 1
                    except grpc.RpcError:
                        pass

        print(f"  Galaxy '{galaxy_name}': edges created ({edge_count} total)", end='\r')

    print(f"\n[+] Hierarchical edges: {edge_count}")

    # 4. Cross-galaxy bridges
    print(f"\n[*] Creating cross-galaxy bridges...")
    bridge_count = 0
    for from_name, to_name, rel_type in CROSS_EDGES:
        from_id = node_ids.get(from_name)
        to_id = node_ids.get(to_name)
        if from_id and to_id:
            try:
                stub.InsertEdge(make_edge_request(pb2,
                    from_node=from_id, to=to_id,
                    edge_type="Association", weight=0.7,
                    collection=collection
                ))
                bridge_count += 1
            except grpc.RpcError:
                pass

    total_edges = edge_count + bridge_count
    print(f"[+] Cross-galaxy bridges: {bridge_count}")

    # 5. Summary
    print(f"\n{'='*60}")
    print(f"  INGESTION COMPLETE")
    print(f"{'='*60}")
    print(f"  Collection:     {collection}")
    print(f"  Galaxies:       {len(GALAXIES)}")
    print(f"  Stars:          {sum(len(g['stars']) for g in GALAXIES.values())}")
    print(f"  Total Nodes:    {inserted}")
    print(f"  Total Edges:    {total_edges}")
    print(f"  Cross Bridges:  {bridge_count}")
    print(f"{'='*60}")
    print(f"\n  The L-System + Agency engine will now grow these")
    print(f"  clusters organically over time, forming galaxies")
    print(f"  in the Poincaré ball visualization!")
    print(f"\n  Dashboard: http://{host.split(':')[0]}:8080")

    channel.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Insert Tech Galaxies into NietzscheDB")
    parser.add_argument("--host", default="localhost:50051", help="gRPC host:port")
    parser.add_argument("--collection", default="cs_knowledge", help="Collection name")
    parser.add_argument("--metric", default="poincare", help="Distance metric")
    parser.add_argument("--dim", type=int, default=3072, help="Embedding dimension")
    args = parser.parse_args()

    insert_galaxies(args.host, args.collection, args.metric, args.dim)
