#!/usr/bin/env python3
"""One-shot hydration of 19 phantom nodes with Claude-generated content."""
import sys, os, json, grpc, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'sdks', 'python'))
from nietzschedb.proto import nietzsche_pb2 as pb, nietzsche_pb2_grpc as rpc

cert_path = os.path.expanduser("~/AppData/Local/Temp/eva-cert.pem")
with open(cert_path, "rb") as f:
    cert = f.read()
ch = grpc.secure_channel("136.111.0.47:443", grpc.ssl_channel_credentials(root_certificates=cert),
    options=[("grpc.max_send_message_length", 256*1024*1024), ("grpc.max_receive_message_length", 256*1024*1024)])
grpc.channel_ready_future(ch).result(timeout=30)
stub = rpc.NietzscheDBStub(ch)
print("Connected.")

hydrations = [
    # ═══ knowledge_galaxies (10) ═══
    ("knowledge_galaxies", "000032e3-66f6-416b-9171-7fd735bfaf79", {
        "title": "Proteomics", "node_label": "concept",
        "definition": "The large-scale study of protein structure, function, and interactions within biological systems, bridging genetics and cellular biochemistry.",
        "tags": ["biology", "molecular", "proteins"], "keywords": ["proteome", "mass spectrometry", "protein folding"],
    }),
    ("knowledge_galaxies", "000146ca-55f5-4450-8917-671662288e0c", {
        "title": "Cosmic Inflation", "node_label": "concept",
        "definition": "The rapid exponential expansion of space in the early universe, occurring fractions of a second after the Big Bang, explaining the large-scale uniformity of the cosmos.",
        "tags": ["cosmology", "early-universe", "physics"], "keywords": ["inflation field", "flatness problem", "horizon problem"],
    }),
    ("knowledge_galaxies", "00019f64-c4a8-41cb-99ac-74b41d7f5c82", {
        "title": "Hawking Radiation", "node_label": "concept",
        "definition": "Theoretical thermal radiation predicted to be emitted by black holes due to quantum effects near the event horizon, implying black holes can slowly evaporate.",
        "tags": ["astrophysics", "quantum-gravity", "black-holes"], "keywords": ["event horizon", "virtual particles", "black hole thermodynamics"],
    }),
    ("knowledge_galaxies", "00038d28-e5c6-42cd-8dab-8b119d845a31", {
        "title": "Film Scoring", "node_label": "concept",
        "definition": "The art of composing and orchestrating music specifically for film, integrating musical narrative with visual storytelling to enhance emotional impact.",
        "tags": ["music", "cinema", "composition"], "keywords": ["leitmotif", "underscore", "diegetic music"],
    }),
    ("knowledge_galaxies", "00054bd2-466d-4c1c-bb3a-30d20a53fb5c", {
        "title": "Hermeneutics", "node_label": "concept",
        "definition": "The theory and methodology of interpretation, particularly of philosophical and literary texts, spanning from ancient exegesis to modern phenomenological approaches.",
        "tags": ["philosophy", "interpretation", "epistemology"], "keywords": ["Gadamer", "hermeneutic circle", "Verstehen"],
    }),
    ("knowledge_galaxies", "0006c437-c50c-4e4b-a541-d8e03dbdafde", {
        "title": "Molecular Evolution", "node_label": "concept",
        "definition": "The study of evolutionary change at the molecular level, analyzing DNA and protein sequences to reconstruct phylogenetic relationships and understand mechanisms of genetic change.",
        "tags": ["biology", "evolution", "genetics"], "keywords": ["neutral theory", "molecular clock", "sequence alignment"],
    }),
    ("knowledge_galaxies", "00085ab5-0498-4b6a-a8dd-d18b126eac4c", {
        "title": "Mise-en-scene", "node_label": "concept",
        "definition": "The arrangement of visual elements within a film frame including set design, lighting, costume, and actor positioning as a primary tool of cinematic expression.",
        "tags": ["cinema", "aesthetics", "visual-arts"], "keywords": ["framing", "visual composition", "auteur theory"],
    }),
    ("knowledge_galaxies", "00086477-cd8e-4c8d-9f19-2ddc1d31046c", {
        "title": "Quantum Field Theory", "node_label": "concept",
        "definition": "The theoretical framework combining quantum mechanics and special relativity to describe particle physics, where fields pervading spacetime give rise to particles as quantized excitations.",
        "tags": ["physics", "quantum", "particles"], "keywords": ["Feynman diagrams", "renormalization", "gauge symmetry"],
    }),
    ("knowledge_galaxies", "00094675-1403-445f-9f0c-f456e1c00209", {
        "title": "Lagrangian Mechanics", "node_label": "concept",
        "definition": "A reformulation of classical mechanics using the principle of least action, expressing dynamics through generalized coordinates and the Lagrangian function.",
        "tags": ["physics", "mechanics", "mathematics"], "keywords": ["Euler-Lagrange", "action principle", "generalized coordinates"],
    }),
    ("knowledge_galaxies", "000a2afb-ba2b-4950-9d25-2c81a553ebe8", {
        "title": "Neuroendocrinology", "node_label": "concept",
        "definition": "The study of interactions between the nervous and endocrine systems, particularly how stress hormones like cortisol affect neuroplasticity, learning, and cognition.",
        "tags": ["neuroscience", "endocrinology", "cognition"], "keywords": ["HPA axis", "cortisol", "stress response"],
    }),
    # ═══ tech_galaxies (9) ═══
    ("tech_galaxies", "0001bf6e-0aac-4311-a12e-4e84040a97f9", {
        "title": "Query Optimization", "node_label": "concept",
        "definition": "The process of selecting the most efficient execution strategy for a database query, involving cost-based analysis of query plans, index usage, and data distribution.",
        "tags": ["databases", "performance", "engineering"], "keywords": ["cost model", "index selection", "join ordering"],
    }),
    ("tech_galaxies", "000d5112-7467-4139-9eb3-917b628131a0", {
        "title": "ANN Indexing", "node_label": "concept",
        "definition": "Approximate Nearest Neighbor indexing structures (HNSW, IVF, LSH) that enable sub-linear similarity search in high-dimensional vector spaces, fundamental to vector databases.",
        "tags": ["vector-search", "databases", "algorithms"], "keywords": ["HNSW", "IVF", "locality-sensitive hashing"],
    }),
    ("tech_galaxies", "00125aa5-cfbc-42eb-8834-a40a621e7dfb", {
        "title": "Algebraic Data Types", "node_label": "concept",
        "definition": "Composite types formed by combining other types using sum (variants/enums) and product (tuples/structs) operations, foundational to functional programming and type theory.",
        "tags": ["type-theory", "functional-programming", "cs-theory"], "keywords": ["sum types", "product types", "pattern matching"],
    }),
    ("tech_galaxies", "00132146-eb47-478c-8c85-880dc41158c0", {
        "title": "Distributed Query Processing", "node_label": "concept",
        "definition": "Techniques for executing queries across distributed database nodes, coordinating query plans, joins, and aggregations over partitioned or replicated data stores.",
        "tags": ["databases", "distributed-systems", "engineering"], "keywords": ["query pushdown", "shuffle join", "partition pruning"],
    }),
    ("tech_galaxies", "001833d5-0d1d-47ff-9711-4b67d44d1176", {
        "title": "Type Systems", "node_label": "concept",
        "definition": "Formal systems that assign types to program expressions, enabling compile-time error detection and guiding software architecture through type-level abstractions.",
        "tags": ["programming-languages", "type-theory", "software"], "keywords": ["type inference", "polymorphism", "dependent types"],
    }),
    ("tech_galaxies", "00194031-1f98-4e3f-ba58-4f0a5f123e9b", {
        "title": "Computational Topology", "node_label": "concept",
        "definition": "The application of topological methods to computation, including persistent homology, simplicial complexes, and hyperbolic embeddings for hierarchical data representation.",
        "tags": ["mathematics", "cs-theory", "geometry"], "keywords": ["persistent homology", "simplicial complex", "Betti numbers"],
    }),
    ("tech_galaxies", "0019e68b-839e-443b-9e2d-eeebeb6cafc4", {
        "title": "ETL Pipelines", "node_label": "concept",
        "definition": "Extract-Transform-Load workflows that move data from source systems through transformation stages into analytical data stores, central to modern data engineering.",
        "tags": ["data-engineering", "databases", "infrastructure"], "keywords": ["data pipeline", "batch processing", "data warehouse"],
    }),
    ("tech_galaxies", "001ecf65-1dc4-4ed0-a52d-c5e764117a73", {
        "title": "Stream Processing", "node_label": "concept",
        "definition": "Real-time data processing paradigm that operates on continuous data streams rather than bounded datasets, enabling low-latency analytics and event-driven architectures.",
        "tags": ["data-engineering", "real-time", "architecture"], "keywords": ["event streaming", "windowing", "exactly-once semantics"],
    }),
    ("tech_galaxies", "0021f8b0-3e35-4a7b-8ba9-26b46e3abd40", {
        "title": "Geometric Algorithms", "node_label": "concept",
        "definition": "Algorithms operating on geometric objects in Euclidean, hyperbolic, or abstract metric spaces, including Voronoi diagrams, convex hulls, and geodesic computations.",
        "tags": ["algorithms", "geometry", "cs-theory"], "keywords": ["Voronoi", "convex hull", "geodesic distance"],
    }),
]

ok = 0
fail = 0
for coll, node_id, content in hydrations:
    try:
        resp = stub.GetNode(pb.NodeIdRequest(id=node_id, collection=coll))
        if not resp.found:
            print(f"  SKIP {node_id[:12]}: not found")
            fail += 1
            continue
        coords = list(resp.embedding.coords)
        ntype = resp.node_type or "Semantic"

        content["hydrated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        content["hydrated_by"] = "semantic-bloom-v1-claude"
        content["hydration_confidence"] = 0.85

        stub.DeleteNode(pb.NodeIdRequest(id=node_id, collection=coll))
        stub.InsertNode(pb.InsertNodeRequest(
            id=node_id,
            embedding=pb.PoincareVector(coords=coords, dim=len(coords)),
            content=json.dumps(content).encode(),
            node_type=ntype,
            energy=0.8,
            collection=coll,
        ))
        print(f"  OK {coll}/{node_id[:12]}: {content['title']}")
        ok += 1
    except Exception as e:
        print(f"  FAIL {node_id[:12]}: {e}")
        fail += 1

print(f"\nDone: {ok} hydrated, {fail} failed")
ch.close()
