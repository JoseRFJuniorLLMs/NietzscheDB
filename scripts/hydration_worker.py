#!/usr/bin/env python3
"""
Phase 29 — Semantic Bloom: Hydration Worker (Week 1 MVP)

Reads phantom/hydration-candidate nodes from NietzscheDB, gathers neighbor
context via BFS DiffusionWalk (2 hops), builds an LLM prompt, and writes
generated content back to the database.

Usage:
  # Dry-run (no writes, just logs what WOULD happen)
  python scripts/hydration_worker.py --dry-run

  # Live run against VM via SSL
  python scripts/hydration_worker.py --host 136.111.0.47:443 --ssl

  # Live run on VM (localhost)
  python scripts/hydration_worker.py --host localhost:50051

  # Limit to specific collections
  python scripts/hydration_worker.py --collections science_galaxies,tech_galaxies

  # Custom batch size and LLM
  python scripts/hydration_worker.py --batch-size 20 --llm-provider gemini

Requirements:
  pip install grpcio grpcio-tools requests google-generativeai
"""

import argparse
import json
import logging
import math
import os
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# SDK path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'sdks', 'python'))

import grpc
from nietzschedb.proto import nietzsche_pb2 as pb
from nietzschedb.proto import nietzsche_pb2_grpc as pb_grpc
from nietzschedb import NietzscheClient
from nietzschedb.types import Node

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_HOST = "136.111.0.47:443"
DEFAULT_HTTP_HOST = "http://136.111.0.47:8080"
DEFAULT_CERT = os.path.expanduser("~/AppData/Local/Temp/eva-cert.pem")
BFS_MAX_DEPTH = 2          # DiffusionWalk hops
BFS_MAX_NODES = 50         # cap neighbors per walk
MIN_NEIGHBOR_CONTENT = 2   # need at least N neighbors with real content
ENERGY_BOOST_HYDRATED = 0.7  # energy after hydration (was ~0.1 as phantom)
MAX_RETRIES_LLM = 2

log = logging.getLogger("hydration")

# ═══════════════════════════════════════════════════════════════════════════
# DATA TYPES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Desire:
    priority: float
    sector: Dict[str, int]
    depth_range: List[float]
    suggested_query: str
    fulfilled: bool
    current_density: float

@dataclass
class HydrationCandidate:
    node_id: str
    coords: List[float]
    collection: str
    energy: float
    node_type: str
    magnitude: float = 0.0

@dataclass
class NeighborContext:
    node_id: str
    title: str
    content_snippet: str
    distance: float  # Poincaré distance from candidate
    magnitude: float

@dataclass
class HydrationResult:
    title: str
    definition: str
    tags: List[str]
    keywords: List[str]
    confidence: float
    node_label: str = ""

@dataclass
class WorkerStats:
    collections_scanned: int = 0
    candidates_found: int = 0
    candidates_hydrated: int = 0
    candidates_skipped: int = 0  # not enough neighbor context
    llm_calls: int = 0
    llm_errors: int = 0
    writes: int = 0
    write_errors: int = 0

# ═══════════════════════════════════════════════════════════════════════════
# gRPC CONNECTION
# ═══════════════════════════════════════════════════════════════════════════

def connect_grpc(host: str, ssl: bool, cert_path: str):
    """Returns (stub, channel). Uses custom cert for self-signed SSL."""
    opts = [
        ("grpc.max_send_message_length", 256 * 1024 * 1024),
        ("grpc.max_receive_message_length", 256 * 1024 * 1024),
        ("grpc.keepalive_time_ms", 30000),
        ("grpc.keepalive_timeout_ms", 10000),
    ]
    if ssl:
        cert = b""
        if cert_path and os.path.exists(cert_path):
            with open(cert_path, "rb") as f:
                cert = f.read()
            log.info("Using SSL cert: %s", cert_path)
        creds = grpc.ssl_channel_credentials(root_certificates=cert or None)
        channel = grpc.secure_channel(host, creds, options=opts)
    else:
        channel = grpc.insecure_channel(host, options=opts)

    log.info("Connecting to %s (ssl=%s) ...", host, ssl)
    grpc.channel_ready_future(channel).result(timeout=30)
    log.info("Connected.")
    stub = pb_grpc.NietzscheDBStub(channel)
    return stub, channel

# ═══════════════════════════════════════════════════════════════════════════
# DESIRES (HTTP)
# ═══════════════════════════════════════════════════════════════════════════

def fetch_desires(http_host: str, collection: str) -> List[Desire]:
    url = f"{http_host}/api/agency/desires?collection={collection}"
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        desires = []
        for d in data.get("desires", []):
            if d.get("fulfilled"):
                continue
            desires.append(Desire(
                priority=d.get("priority", 0),
                sector=d.get("sector", {}),
                depth_range=d.get("depth_range", [0.0, 1.0]),
                suggested_query=d.get("suggested_query", ""),
                fulfilled=d.get("fulfilled", False),
                current_density=d.get("current_density", 0),
            ))
        desires.sort(key=lambda x: -x.priority)
        return desires
    except Exception as e:
        log.warning("Failed to fetch desires for %s: %s", collection, e)
        return []

# ═══════════════════════════════════════════════════════════════════════════
# FIND HYDRATION CANDIDATES
# ═══════════════════════════════════════════════════════════════════════════

def find_candidates(stub: pb_grpc.NietzscheDBStub, collection: str,
                    limit: int = 200) -> List[HydrationCandidate]:
    """Find nodes with empty content or hydration_candidate=true."""
    candidates = []

    # Strategy 1: Query nodes via NQL, check content emptiness
    # NQL can't filter on metadata directly, so we fetch a batch and filter locally
    try:
        resp = stub.Query(pb.QueryRequest(
            nql=f"MATCH (n:Semantic) RETURN n LIMIT {limit}",
            collection=collection,
        ))
        for node_pb in resp.nodes:
            content = _parse_content(node_pb.content) or {}
            is_empty = _is_empty_content(content)
            is_hydration = content.get("hydration_candidate") is True if content else False

            if is_empty or is_hydration:
                coords = list(node_pb.embedding.coords) if node_pb.embedding.coords else []
                mag = math.sqrt(sum(c * c for c in coords)) if coords else 0.0
                candidates.append(HydrationCandidate(
                    node_id=node_pb.id,
                    coords=coords,
                    collection=collection,
                    energy=node_pb.energy,
                    node_type=node_pb.node_type,
                    magnitude=mag,
                ))
    except grpc.RpcError as e:
        log.warning("NQL query failed for %s: %s", collection, e)

    # Strategy 2: Also try Concept and Episodic types
    for ntype in ["Concept", "Episodic"]:
        try:
            resp = stub.Query(pb.QueryRequest(
                nql=f"MATCH (n:{ntype}) RETURN n LIMIT {limit // 2}",
                collection=collection,
            ))
            for node_pb in resp.nodes:
                content = _parse_content(node_pb.content) or {}
                if _is_empty_content(content) or (content and content.get("hydration_candidate") is True):
                    coords = list(node_pb.embedding.coords) if node_pb.embedding.coords else []
                    mag = math.sqrt(sum(c * c for c in coords)) if coords else 0.0
                    # Avoid duplicates
                    if not any(c.node_id == node_pb.id for c in candidates):
                        candidates.append(HydrationCandidate(
                            node_id=node_pb.id,
                            coords=coords,
                            collection=collection,
                            energy=node_pb.energy,
                            node_type=node_pb.node_type,
                            magnitude=mag,
                        ))
        except grpc.RpcError:
            pass

    log.info("  [%s] Found %d hydration candidates", collection, len(candidates))
    return candidates


def _parse_content(raw: bytes) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}


def _is_empty_content(content: Dict[str, Any]) -> bool:
    """A node is 'empty' if it has no meaningful content fields.

    L-System phantoms have content=null (parsed as None/{}), or just
    metadata keys like 'hydration_candidate'. Real nodes have domain
    fields like name, text, description, category, etc.
    """
    if not content:
        return True
    # Metadata-only keys that don't count as "real content"
    metadata_only = {"hydration_candidate", "hydrated_by", "hydrated_at",
                     "hydration_confidence"}
    real_keys = set(content.keys()) - metadata_only
    return len(real_keys) == 0

# ═══════════════════════════════════════════════════════════════════════════
# DIFFUSION WALK (BFS 2-hop neighbor context)
# ═══════════════════════════════════════════════════════════════════════════

def prefetch_seed_nodes(stub: pb_grpc.NietzscheDBStub, collection: str
                        ) -> List[NeighborContext]:
    """Pre-fetch all nodes with real content in a collection (seed/galaxy nodes).

    Since L-System creates massive phantom subgraphs, BFS and KNN both return
    empty nodes. Instead, we pre-load all content-bearing nodes and compute
    Poincaré distance in Python to find the closest seeds for each candidate.
    """
    seeds = []
    for ntype in ["Concept", "Semantic", "Episodic"]:
        try:
            resp = stub.Query(pb.QueryRequest(
                nql=f"MATCH (n:{ntype}) RETURN n LIMIT 500",
                collection=collection,
            ))
            for node_pb in resp.nodes:
                content = _parse_content(node_pb.content)
                if not content or _is_empty_content(content):
                    continue
                title = (content.get("title") or content.get("name") or
                         content.get("label") or content.get("concept") or "")
                snippet = (content.get("definition") or content.get("description") or
                           content.get("text") or content.get("summary") or
                           content.get("meaning") or "")
                if not title and not snippet:
                    snippet = json.dumps(content)[:200]
                    if not snippet or snippet == "{}":
                        continue
                coords = list(node_pb.embedding.coords) if node_pb.embedding.coords else []
                mag = math.sqrt(sum(c * c for c in coords)) if coords else 0.0
                seeds.append(NeighborContext(
                    node_id=node_pb.id,
                    title=str(title),
                    content_snippet=str(snippet)[:300],
                    distance=0.0,
                    magnitude=mag,
                ))
                # Store coords as extra attribute for distance calc
                seeds[-1]._coords = coords
        except grpc.RpcError:
            pass
    log.info("  Pre-fetched %d seed nodes with content", len(seeds))
    return seeds


def find_nearest_seeds(candidate: HydrationCandidate, seeds: List[NeighborContext],
                       top_k: int = 10) -> List[NeighborContext]:
    """Find the closest seed nodes to a candidate by Poincaré distance."""
    if not candidate.coords or not seeds:
        return []
    scored = []
    for s in seeds:
        s_coords = getattr(s, '_coords', [])
        dist = _poincare_distance(candidate.coords, s_coords) if s_coords else 10.0
        scored.append((dist, s))
    scored.sort(key=lambda x: x[0])
    result = []
    for dist, s in scored[:top_k]:
        result.append(NeighborContext(
            node_id=s.node_id,
            title=s.title,
            content_snippet=s.content_snippet,
            distance=dist,
            magnitude=s.magnitude,
        ))
    return result


def diffusion_walk(stub: pb_grpc.NietzscheDBStub, candidate: HydrationCandidate,
                   seeds: List[NeighborContext]) -> List[NeighborContext]:
    """Find neighbors with real content using BFS + seed proximity fallback.

    Strategy:
    1. BFS 2 hops (topology neighbors) — fast, respects graph structure
    2. If BFS finds too few real neighbors, use pre-fetched seed nodes
       and compute Poincaré distance to find closest content-bearing nodes
    """
    neighbors = []

    # Strategy 1: BFS topology walk
    try:
        resp = stub.Bfs(pb.TraversalRequest(
            start_node_id=candidate.node_id,
            max_depth=BFS_MAX_DEPTH,
            max_nodes=BFS_MAX_NODES,
            energy_min=0.0,
            collection=candidate.collection,
        ))
        visited = list(resp.visited_ids)
        for nid in visited:
            if nid == candidate.node_id:
                continue
            ctx = _fetch_neighbor_context(stub, nid, candidate)
            if ctx:
                neighbors.append(ctx)
    except grpc.RpcError as e:
        log.debug("  BFS failed for %s: %s", candidate.node_id[:8], e)

    # Strategy 2: Seed proximity fallback
    if len(neighbors) < MIN_NEIGHBOR_CONTENT:
        nearest = find_nearest_seeds(candidate, seeds)
        seen = {n.node_id for n in neighbors}
        for s in nearest:
            if s.node_id not in seen:
                neighbors.append(s)
                seen.add(s.node_id)

    # Sort by distance (closest first)
    neighbors.sort(key=lambda n: n.distance)
    return neighbors[:10]


def _fetch_neighbor_context(stub, nid: str, candidate: HydrationCandidate
                            ) -> Optional[NeighborContext]:
    """Fetch a node and return context if it has real content."""
    try:
        node_resp = stub.GetNode(pb.NodeIdRequest(
            id=nid, collection=candidate.collection
        ))
        if not node_resp.found:
            return None
        content = _parse_content(node_resp.content) or {}
        if _is_empty_content(content):
            return None

        title = (content.get("title") or content.get("name") or
                 content.get("label") or content.get("concept") or "")
        snippet = (content.get("definition") or content.get("description") or
                   content.get("text") or content.get("summary") or
                   content.get("meaning") or "")
        if not title and not snippet:
            snippet = json.dumps(content)[:200]
            if not snippet or snippet == "{}":
                return None

        coords = list(node_resp.embedding.coords) if node_resp.embedding.coords else []
        mag = math.sqrt(sum(c * c for c in coords)) if coords else 0.0
        dist = _poincare_distance(candidate.coords, coords) if coords and candidate.coords else 1.0

        return NeighborContext(
            node_id=nid,
            title=str(title),
            content_snippet=str(snippet)[:300],
            distance=dist,
            magnitude=mag,
        )
    except grpc.RpcError:
        return None


def _poincare_distance(u: List[float], v: List[float]) -> float:
    """Approximate Poincaré ball distance."""
    if not u or not v or len(u) != len(v):
        return 1.0
    diff_sq = sum((a - b) ** 2 for a, b in zip(u, v))
    nu = sum(a * a for a in u)
    nv = sum(b * b for b in v)
    denom = (1 - nu) * (1 - nv)
    if denom <= 0:
        return 10.0
    arg = 1 + 2 * diff_sq / max(denom, 1e-10)
    return math.acosh(max(arg, 1.0))

# ═══════════════════════════════════════════════════════════════════════════
# LLM CONTENT GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def build_prompt(candidate: HydrationCandidate, neighbors: List[NeighborContext],
                 desire: Optional[Desire] = None) -> str:
    """Build the LLM prompt for content generation."""
    parts = []
    parts.append("You are a knowledge graph content generator for NietzscheDB, a hyperbolic graph database.")
    parts.append("A node exists in the graph with coordinates but no content. Your job is to infer what this node represents based on its neighbors and position.")
    parts.append("")

    parts.append(f"## Node to hydrate")
    parts.append(f"- ID: {candidate.node_id[:12]}...")
    parts.append(f"- Type: {candidate.node_type}")
    parts.append(f"- Magnitude (depth in Poincaré ball): {candidate.magnitude:.4f}")
    depth_desc = "abstract/general" if candidate.magnitude < 0.3 else "mid-level" if candidate.magnitude < 0.6 else "specific/concrete"
    parts.append(f"- Depth interpretation: {depth_desc}")
    parts.append(f"- Collection: {candidate.collection}")
    parts.append("")

    if neighbors:
        parts.append(f"## Neighboring nodes ({len(neighbors)} with content)")
        for i, n in enumerate(neighbors, 1):
            parts.append(f"{i}. **{n.title or '(untitled)'}** (dist={n.distance:.3f}, mag={n.magnitude:.3f})")
            if n.content_snippet:
                parts.append(f"   {n.content_snippet[:200]}")
        parts.append("")

    if desire:
        parts.append(f"## Knowledge desire (what the graph wants)")
        parts.append(f"- Priority: {desire.priority:.2f}")
        parts.append(f"- Suggested topic: {desire.suggested_query}")
        parts.append(f"- Depth range: {desire.depth_range}")
        parts.append("")

    parts.append("## Your task")
    parts.append("Based on the neighbors and position, infer what concept this node represents.")
    parts.append("Respond ONLY with a JSON object (no markdown, no explanation):")
    parts.append("""```json
{
  "title": "short concept name",
  "definition": "1-2 sentence definition of what this concept is",
  "tags": ["tag1", "tag2", "tag3"],
  "keywords": ["keyword1", "keyword2"],
  "confidence": 0.85,
  "node_label": "specific label for this concept type"
}
```""")
    parts.append("")
    parts.append("Rules:")
    parts.append("- Title should be concise (1-5 words)")
    parts.append("- Definition should be factual and relate to the neighbors")
    parts.append("- Confidence: 0.0-1.0, how sure you are about the inference")
    parts.append("- If you cannot infer anything meaningful, set confidence < 0.3")
    parts.append("- Match the domain of the neighboring concepts")

    return "\n".join(parts)


def call_llm_gemini(prompt: str, api_key: str) -> Optional[HydrationResult]:
    """Call Gemini for content generation. Uses new google.genai SDK."""
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.0-flash-lite",
            contents=prompt,
            config={
                "temperature": 0.3,
                "max_output_tokens": 500,
                "response_mime_type": "application/json",
            },
        )
        return _parse_llm_response(response.text)
    except ImportError:
        # Fallback to old SDK
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-2.0-flash-lite")
            response = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=500,
                    response_mime_type="application/json",
                ),
            )
            return _parse_llm_response(response.text)
        except Exception as e:
            log.error("Gemini API error (old SDK): %s", e)
            return None
    except Exception as e:
        log.error("Gemini API error: %s", e)
        return None


def call_llm_anthropic(prompt: str, api_key: str) -> Optional[HydrationResult]:
    """Call Claude for content generation (fallback)."""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text
        return _parse_llm_response(text)
    except Exception as e:
        log.error("Anthropic API error: %s", e)
        return None


def _parse_llm_response(text: str) -> Optional[HydrationResult]:
    """Parse JSON from LLM response."""
    text = text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        data = json.loads(text)
        return HydrationResult(
            title=data.get("title", ""),
            definition=data.get("definition", ""),
            tags=data.get("tags", []),
            keywords=data.get("keywords", []),
            confidence=float(data.get("confidence", 0.0)),
            node_label=data.get("node_label", ""),
        )
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        log.error("Failed to parse LLM response: %s — text: %s", e, text[:200])
        return None

# ═══════════════════════════════════════════════════════════════════════════
# POSITIONAL INHERITANCE FALLBACK
# ═══════════════════════════════════════════════════════════════════════════

def positional_fallback(candidate: HydrationCandidate) -> HydrationResult:
    """When no neighbors have content, generate minimal content from position."""
    depth_desc = "abstract" if candidate.magnitude < 0.3 else "intermediate" if candidate.magnitude < 0.6 else "specific"
    return HydrationResult(
        title=f"Node-{candidate.node_id[:8]}",
        definition=f"Auto-generated placeholder for {depth_desc}-depth node in {candidate.collection} (magnitude {candidate.magnitude:.3f}). Awaiting richer neighbor context for full hydration.",
        tags=[depth_desc, candidate.collection, "auto-placeholder"],
        keywords=["placeholder", "pending-hydration"],
        confidence=0.15,
        node_label="Placeholder",
    )

# ═══════════════════════════════════════════════════════════════════════════
# WRITE BACK TO DB
# ═══════════════════════════════════════════════════════════════════════════

def write_hydration(stub: pb_grpc.NietzscheDBStub, candidate: HydrationCandidate,
                    result: HydrationResult, dry_run: bool) -> bool:
    """Write hydrated content back to NietzscheDB via MergeNode."""
    content = {
        "title": result.title,
        "definition": result.definition,
        "tags": result.tags,
        "keywords": result.keywords,
        "hydration_confidence": result.confidence,
        "hydrated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hydrated_by": "semantic-bloom-v1",
    }
    if result.node_label:
        content["node_label"] = result.node_label

    if dry_run:
        log.info("  [DRY-RUN] Would write to %s: title=%s conf=%.2f",
                 candidate.node_id[:12], result.title, result.confidence)
        return True

    try:
        # Use MergeNode with match on ID — on_match patches the content
        match_keys = json.dumps({"id": candidate.node_id}).encode()
        on_match = json.dumps(content).encode()

        # MergeNode matches by content fields, not ID. For existing nodes,
        # we use the raw stub to update: GetNode + reconstruct with new content
        node_resp = stub.GetNode(pb.NodeIdRequest(
            id=candidate.node_id, collection=candidate.collection
        ))
        if not node_resp.found:
            log.warning("  Node %s not found during write-back", candidate.node_id[:12])
            return False

        # Merge existing content with hydration content
        existing = _parse_content(node_resp.content)
        existing.update(content)
        # Remove hydration_candidate flag since we're hydrating it
        existing.pop("hydration_candidate", None)

        # Delete and re-insert with content (atomic-ish via gRPC)
        # This preserves edges since NietzscheDB edges reference by UUID
        coords = list(node_resp.embedding.coords) if node_resp.embedding.coords else candidate.coords

        stub.DeleteNode(pb.NodeIdRequest(
            id=candidate.node_id, collection=candidate.collection
        ))
        stub.InsertNode(pb.InsertNodeRequest(
            id=candidate.node_id,
            embedding=pb.PoincareVector(coords=coords),
            content=json.dumps(existing).encode(),
            node_type=candidate.node_type or "Semantic",
            energy=ENERGY_BOOST_HYDRATED,
            collection=candidate.collection,
        ))
        log.info("  Hydrated %s: title=%s conf=%.2f",
                 candidate.node_id[:12], result.title, result.confidence)
        return True

    except grpc.RpcError as e:
        log.error("  Write failed for %s: %s", candidate.node_id[:12], e)
        return False

# ═══════════════════════════════════════════════════════════════════════════
# MAIN WORKER LOOP
# ═══════════════════════════════════════════════════════════════════════════

def get_collections(stub: pb_grpc.NietzscheDBStub,
                    filter_names: Optional[List[str]] = None) -> List[str]:
    """List collections, optionally filtered."""
    resp = stub.ListCollections(pb.Empty())
    names = []
    for c in resp.collections:
        name = c.collection
        # Skip system/cache/test collections
        if any(name.startswith(p) for p in ("eva_cache", "eva_perceptions",
               "speaker_embeddings", "eva_sensory", "lobby_", "test_")):
            continue
        if filter_names and name not in filter_names:
            continue
        # Only collections with some nodes
        if c.node_count > 0:
            names.append(name)
    return sorted(names)


def run_worker(args):
    stats = WorkerStats()

    # Connect
    stub, channel = connect_grpc(args.host, args.ssl, args.cert)

    # LLM setup
    llm_provider = args.llm_provider
    api_key = args.api_key
    if not api_key:
        if llm_provider == "gemini":
            api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        elif llm_provider == "anthropic":
            api_key = os.environ.get("ANTHROPIC_API_KEY")

    if not api_key and not args.dry_run:
        log.error("No API key provided. Use --api-key or set GOOGLE_API_KEY / ANTHROPIC_API_KEY env var.")
        log.error("Or use --dry-run to see what would happen without LLM calls.")
        sys.exit(1)

    call_llm = call_llm_gemini if llm_provider == "gemini" else call_llm_anthropic

    # Get collections
    filter_names = args.collections.split(",") if args.collections else None
    collections = get_collections(stub, filter_names)
    log.info("Collections to scan: %s", collections)

    for coll in collections:
        stats.collections_scanned += 1
        log.info("═══ Scanning: %s ═══", coll)

        # Fetch desires for context
        desires = fetch_desires(args.http_host, coll)
        if desires:
            log.info("  %d unfulfilled desires (top priority=%.2f: %s)",
                     len(desires), desires[0].priority, desires[0].suggested_query[:60])

        # Pre-fetch seed nodes with content
        seeds = prefetch_seed_nodes(stub, coll)

        # Find candidates
        candidates = find_candidates(stub, coll, limit=args.batch_size)
        stats.candidates_found += len(candidates)

        if not candidates:
            log.info("  No candidates found, skipping.")
            continue

        if not seeds:
            log.warning("  No seed nodes with content found — cannot hydrate without context.")
            stats.candidates_skipped += len(candidates)
            continue

        for candidate in candidates:
            # DiffusionWalk: BFS 2 hops + seed proximity
            neighbors = diffusion_walk(stub, candidate, seeds)
            log.info("  Node %s (mag=%.3f): %d neighbors with content",
                     candidate.node_id[:12], candidate.magnitude, len(neighbors))

            if len(neighbors) < MIN_NEIGHBOR_CONTENT:
                # Not enough context — use positional fallback or skip
                if args.allow_fallback:
                    result = positional_fallback(candidate)
                    log.info("  Using positional fallback (confidence=%.2f)", result.confidence)
                else:
                    log.info("  Skipping: only %d neighbors (need %d)",
                             len(neighbors), MIN_NEIGHBOR_CONTENT)
                    stats.candidates_skipped += 1
                    continue
            else:
                # Build prompt and call LLM
                desire = desires[0] if desires else None
                prompt = build_prompt(candidate, neighbors, desire)

                if args.dry_run:
                    log.info("  [DRY-RUN] Would call LLM with %d chars prompt, %d neighbors",
                             len(prompt), len(neighbors))
                    for n in neighbors[:3]:
                        log.info("    - %s (dist=%.3f): %s", n.title[:30], n.distance, n.content_snippet[:80])
                    stats.candidates_hydrated += 1
                    continue

                stats.llm_calls += 1
                result = None
                for attempt in range(MAX_RETRIES_LLM):
                    result = call_llm(prompt, api_key)
                    if result:
                        break
                    log.warning("  LLM attempt %d failed, retrying...", attempt + 1)
                    time.sleep(1)

                if not result:
                    stats.llm_errors += 1
                    log.error("  LLM failed after %d attempts for %s",
                              MAX_RETRIES_LLM, candidate.node_id[:12])
                    continue

                # Skip low-confidence results
                if result.confidence < args.min_confidence:
                    log.info("  Skipping low confidence (%.2f < %.2f): %s",
                             result.confidence, args.min_confidence, result.title)
                    stats.candidates_skipped += 1
                    continue

            # Write back
            ok = write_hydration(stub, candidate, result, args.dry_run)
            if ok:
                stats.candidates_hydrated += 1
                stats.writes += 1
            else:
                stats.write_errors += 1

            # Rate limit: don't hammer the LLM
            if not args.dry_run:
                time.sleep(args.delay)

    # Summary
    log.info("═══════════════════════════════════════════")
    log.info("Hydration Worker Summary:")
    log.info("  Collections scanned: %d", stats.collections_scanned)
    log.info("  Candidates found:    %d", stats.candidates_found)
    log.info("  Hydrated:            %d", stats.candidates_hydrated)
    log.info("  Skipped (no context):%d", stats.candidates_skipped)
    log.info("  LLM calls:           %d", stats.llm_calls)
    log.info("  LLM errors:          %d", stats.llm_errors)
    log.info("  DB writes:           %d", stats.writes)
    log.info("  Write errors:        %d", stats.write_errors)
    log.info("═══════════════════════════════════════════")

    channel.close()
    return stats


def main():
    parser = argparse.ArgumentParser(description="Phase 29 Semantic Bloom — Hydration Worker")
    parser.add_argument("--host", default=DEFAULT_HOST, help="gRPC host:port")
    parser.add_argument("--http-host", default=DEFAULT_HTTP_HOST, help="HTTP API base URL (for desires)")
    parser.add_argument("--ssl", action="store_true", default=True, help="Use SSL (default: true)")
    parser.add_argument("--no-ssl", dest="ssl", action="store_false", help="Disable SSL")
    parser.add_argument("--cert", default=DEFAULT_CERT, help="SSL cert path")
    parser.add_argument("--collections", default="", help="Comma-separated collection names (default: all)")
    parser.add_argument("--batch-size", type=int, default=200, help="Max candidates per collection")
    parser.add_argument("--dry-run", action="store_true", help="Log what would happen, no writes or LLM calls")
    parser.add_argument("--llm-provider", choices=["gemini", "anthropic"], default="gemini",
                        help="LLM provider (default: gemini)")
    parser.add_argument("--api-key", default="", help="LLM API key (or use env var)")
    parser.add_argument("--min-confidence", type=float, default=0.3,
                        help="Minimum LLM confidence to write (default: 0.3)")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Seconds between LLM calls (default: 0.5)")
    parser.add_argument("--allow-fallback", action="store_true",
                        help="Use positional inheritance when not enough neighbors")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.dry_run:
        log.info("=== DRY-RUN MODE — no writes, no LLM calls ===")

    run_worker(args)


if __name__ == "__main__":
    main()
