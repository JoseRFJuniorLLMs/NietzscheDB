#!/usr/bin/env python3
"""
NietzscheDB — TGC Experiment: Dataset Preparation

Downloads the Cora citation network, remaps node IDs to contiguous [0, N),
splits edges 90/10 into train/test, and saves CSVs.

Output (in experiments/):
  - train_edges.csv  (source,target)
  - test_edges.csv   (source,target)
  - dataset_stats.txt

Usage:
  py -3 experiments/dataset_prepare.py
"""

import os
import sys
import random
import urllib.request
import tarfile
import tempfile
import shutil

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = SCRIPT_DIR  # experiments/

CORA_URL = "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz"
RANDOM_SEED = 42
TEST_RATIO = 0.10


def download_cora():
    """Download and extract Cora dataset. Returns path to cora.cites file."""
    # Check if we already have a local copy
    local_cites = os.path.join(OUTPUT_DIR, "cora.cites")
    local_content = os.path.join(OUTPUT_DIR, "cora.content")
    if os.path.exists(local_cites) and os.path.exists(local_content):
        print(f"  Found local {local_cites} + {local_content}")
        return local_cites

    print(f"  Downloading Cora from {CORA_URL}...")
    tmpdir = tempfile.mkdtemp()
    try:
        tgz_path = os.path.join(tmpdir, "cora.tgz")
        urllib.request.urlretrieve(CORA_URL, tgz_path)
        print(f"  Downloaded to {tgz_path}")

        with tarfile.open(tgz_path, "r:gz") as tar:
            tar.extractall(tmpdir)

        # Find cora.cites and cora.content in extracted files
        found_cites = False
        for root, dirs, files in os.walk(tmpdir):
            for f in files:
                if f == "cora.cites":
                    src = os.path.join(root, f)
                    shutil.copy2(src, local_cites)
                    print(f"  Extracted cora.cites -> {local_cites}")
                    found_cites = True
                if f == "cora.content":
                    src = os.path.join(root, f)
                    shutil.copy2(src, local_content)
                    print(f"  Extracted cora.content -> {local_content}")

        if not found_cites:
            raise FileNotFoundError("cora.cites not found in archive")
        return local_cites
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def parse_cora_cites(path):
    """Parse cora.cites file. Format: cited_paper_id<tab>citing_paper_id per line."""
    edges = []
    nodes = set()
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                cited = parts[0]
                citing = parts[1]
                if cited != citing:  # no self-loops
                    edges.append((cited, citing))
                    nodes.add(cited)
                    nodes.add(citing)
    return edges, nodes


def remap_ids(edges, nodes):
    """Remap string node IDs to contiguous integers [0, N)."""
    sorted_nodes = sorted(nodes)
    id_map = {old: new for new, old in enumerate(sorted_nodes)}
    remapped = [(id_map[u], id_map[v]) for u, v in edges]
    return remapped, len(sorted_nodes), id_map


def deduplicate(edges):
    """Remove duplicate edges (treat as undirected)."""
    seen = set()
    unique = []
    for u, v in edges:
        key = (min(u, v), max(u, v))
        if key not in seen:
            seen.add(key)
            unique.append((u, v))
    return unique


def split_edges(edges, test_ratio, seed):
    """Split edges into train and test sets."""
    random.seed(seed)
    shuffled = list(edges)
    random.shuffle(shuffled)
    n_test = int(len(shuffled) * test_ratio)
    test = shuffled[:n_test]
    train = shuffled[n_test:]
    return train, test


def save_csv(edges, path):
    """Save edge list as CSV with header."""
    with open(path, "w") as f:
        f.write("source,target\n")
        for u, v in edges:
            f.write(f"{u},{v}\n")


def generate_synthetic_fallback():
    """Generate synthetic SBM graph if Cora download fails."""
    print("  Generating synthetic Stochastic Block Model (fallback)...")
    random.seed(RANDOM_SEED)

    communities = 7
    nodes_per_comm = 386  # ~2702 total
    total = communities * nodes_per_comm
    p_intra = 0.008
    p_inter = 0.0004

    edges = []
    for i in range(total):
        ci = i // nodes_per_comm
        for j in range(i + 1, total):
            cj = j // nodes_per_comm
            p = p_intra if ci == cj else p_inter
            if random.random() < p:
                edges.append((i, j))

    print(f"  Generated {len(edges)} edges, {total} nodes")
    return edges, total


def main():
    print("=" * 64)
    print("  NietzscheDB: Dataset Preparation for TGC Experiment")
    print("=" * 64)
    print()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Try Cora, fallback to synthetic
    try:
        cites_path = download_cora()
        raw_edges, raw_nodes = parse_cora_cites(cites_path)
        print(f"  Raw: {len(raw_edges)} edges, {len(raw_nodes)} nodes")
        edges, n_nodes, _ = remap_ids(raw_edges, raw_nodes)
        edges = deduplicate(edges)
        print(f"  After dedup: {len(edges)} unique undirected edges, {n_nodes} nodes")
    except Exception as e:
        print(f"  Cora download failed: {e}")
        edges, n_nodes = generate_synthetic_fallback()
        edges = deduplicate(edges)

    # Split
    train, test = split_edges(edges, TEST_RATIO, RANDOM_SEED)
    print(f"  Train: {len(train)} edges ({100*(1-TEST_RATIO):.0f}%)")
    print(f"  Test:  {len(test)} edges ({100*TEST_RATIO:.0f}%)")

    # Verify all test nodes appear in train
    train_nodes = set()
    for u, v in train:
        train_nodes.add(u)
        train_nodes.add(v)

    test_filtered = [(u, v) for u, v in test if u in train_nodes and v in train_nodes]
    if len(test_filtered) < len(test):
        print(f"  Filtered test: {len(test)} -> {len(test_filtered)} (removed orphan test edges)")
        test = test_filtered

    # Export node labels from cora.content (if available)
    content_path = os.path.join(OUTPUT_DIR, "cora.content")
    labels_path = os.path.join(OUTPUT_DIR, "node_labels.csv")
    if os.path.exists(content_path):
        # cora.content format: paper_id<tab>word1<tab>...<tab>wordN<tab>class_label
        label_set = set()
        raw_labels = {}
        with open(content_path, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    paper_id = parts[0]
                    class_label = parts[-1]
                    raw_labels[paper_id] = class_label
                    label_set.add(class_label)

        # Map class names to integers
        label_names = sorted(label_set)
        label_map = {name: i for i, name in enumerate(label_names)}

        # Remap using same id_map from edges (need to get it)
        # Re-parse to get id_map
        try:
            cites_path2 = os.path.join(OUTPUT_DIR, "cora.cites")
            raw_edges2, raw_nodes2 = parse_cora_cites(cites_path2)
            _, _, id_map = remap_ids(raw_edges2, raw_nodes2)

            with open(labels_path, "w") as f:
                f.write("node_id,label,label_name\n")
                for paper_id, new_id in sorted(id_map.items(), key=lambda x: x[1]):
                    if paper_id in raw_labels:
                        lname = raw_labels[paper_id]
                        lid = label_map[lname]
                        f.write(f"{new_id},{lid},{lname}\n")

            print(f"  Saved: {labels_path} ({len(label_names)} classes: {', '.join(label_names)})")
        except Exception as e:
            print(f"  Warning: Could not export labels: {e}")
    else:
        print(f"  No cora.content found, skipping label export")

    # Save
    train_path = os.path.join(OUTPUT_DIR, "train_edges.csv")
    test_path = os.path.join(OUTPUT_DIR, "test_edges.csv")
    save_csv(train, train_path)
    save_csv(test, test_path)
    print(f"  Saved: {train_path}")
    print(f"  Saved: {test_path}")

    # Stats
    stats_path = os.path.join(OUTPUT_DIR, "dataset_stats.txt")
    all_nodes = set()
    for u, v in train + test:
        all_nodes.add(u)
        all_nodes.add(v)

    # Degree distribution
    from collections import Counter
    degree = Counter()
    for u, v in train:
        degree[u] += 1
        degree[v] += 1
    degs = sorted(degree.values())
    mean_deg = sum(degs) / len(degs) if degs else 0
    median_deg = degs[len(degs) // 2] if degs else 0

    with open(stats_path, "w") as f:
        f.write("NietzscheDB TGC Experiment - Dataset Statistics\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total nodes: {len(all_nodes)}\n")
        f.write(f"Total edges: {len(train) + len(test)}\n")
        f.write(f"Train edges: {len(train)}\n")
        f.write(f"Test edges: {len(test)}\n")
        f.write(f"Mean degree (train): {mean_deg:.2f}\n")
        f.write(f"Median degree (train): {median_deg}\n")
        f.write(f"Max degree (train): {max(degs) if degs else 0}\n")
        f.write(f"Min degree (train): {min(degs) if degs else 0}\n")

    print(f"  Saved: {stats_path}")
    print()
    print(f"  Mean degree: {mean_deg:.2f}, Median: {median_deg}")
    print()
    print("  Ready! Now run:")
    print("    cargo run --release --bin experiment_link_prediction")
    print()


if __name__ == "__main__":
    main()
