# NietzscheDB Storage Layout (RocksDB)

NietzscheDB utilizes RocksDB as its primary graph storage engine, organized into **15 Column Families (CF)** to optimize I/O patterns and enable efficient secondary indexing.

## Column Family Catalog

| Name | Role | Key Format | Value Format |
|---|---|---|---|
| `nodes` | Node metadata (NodeMeta) | NodeID (UUID) | Bincode(NodeMeta) |
| `embeddings` | High-dim vectors | NodeID (UUID) | f32 Array |
| `edges` | Edge metadata | EdgeID | Bincode(Edge) |
| `adj_out` | Outgoing adjacency | NodeID | Vec<EdgeID> |
| `adj_in` | Incoming adjacency | NodeID | Vec<EdgeID> |
| `meta` | System metadata/Registry | String Key | Mixed |
| `sensory` | Sensory latent data | NodeID | Compressed Latent |
| `energy_idx` | Sorted Energy Index | Energy(f32) + ID | (Empty) |
| `meta_idx` | Generic Property Index | PropValue + ID | (Empty) |
| `lists` | Atomic lists (Redis-like) | ListKey | Vec<ValueID> |
| `sql_schema` | Relational schemas | TableName | Schema JSON |
| `sql_data` | Relational records | TableName + RowID | Record JSON |
| `cooldowns` | Code-as-Data activation | NodeID | Timestamp |
| `dsi_id` | DSI ID Mapping | NodeID | SemanticCode |
| `dsi_semantic` | Semantic Trie Index | SemanticPrefix | Vec<NodeID> |

## Performance Optimization (Split Storage)

A key architectural feature of NietzscheDB is the **Separation of Nodes and Embeddings**.
- **The Problem**: Standard graph databases store vectors inline. A recursive traversal (BFS) reads megabytes of vector data it doesn't need, thrashing the cache.
- **The Solution**: NietzscheDB stores `NodeMeta` (~100 bytes) in the `nodes` CF and vectors in `embeddings`.
- **Result**: Traversal speed is **10-25x faster** because the kernel only loads the metadata CF into memory during the search phase.

## Secondary Indexing
Indexes on fields like `energy`, `depth`, or `node_type` are maintained in `energy_idx` and `meta_idx`. These are **Sortable-Key Indexes**, allowing for O(log N) range scans directly via RocksDB iterators.
