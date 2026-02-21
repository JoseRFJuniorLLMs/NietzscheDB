//! # nietzsche-kafka
//!
//! Kafka sink connector for NietzscheDB -- streams graph mutations from
//! Kafka topics into the hyperbolic graph storage engine.
//!
//! ## Architecture
//!
//! The connector implements a **sink** pattern: it consumes serialized
//! [`GraphMutation`] messages from one or more Kafka topics and applies them
//! to a [`nietzsche_graph::GraphStorage`] instance, keeping both the
//! persistent RocksDB store and the in-memory [`nietzsche_graph::AdjacencyIndex`]
//! in sync.
//!
//! ```text
//!  Kafka Topic(s)
//!       |
//!       v
//!  +-----------+     JSON deserialization
//!  | Consumer  | --------------------------> GraphMutation
//!  +-----------+                                  |
//!                                                 v
//!                                          +-----------+
//!                                          | KafkaSink |
//!                                          +-----------+
//!                                           |         |
//!                              GraphStorage v         v AdjacencyIndex
//!                              (RocksDB)               (DashMap)
//! ```
//!
//! ## Supported Mutations
//!
//! | Variant       | Description                                  |
//! |---------------|----------------------------------------------|
//! | `InsertNode`  | Insert a node with optional embedding/energy |
//! | `DeleteNode`  | Remove a node and its adjacency entries       |
//! | `InsertEdge`  | Create a directed edge between two nodes      |
//! | `DeleteEdge`  | Remove an edge by ID                          |
//! | `SetEnergy`   | Update a node's energy level                  |
//! | `SetContent`  | Merge fields into a node's JSON content       |
//!
//! ## Batch Processing
//!
//! The [`KafkaSink::process_batch`] method applies a slice of mutations
//! atomically per message, collecting per-message errors into a
//! [`BatchResult`].  A single bad record never blocks the rest of the batch.
//!
//! ## Error Handling
//!
//! All errors are surfaced through [`KafkaError`], which covers JSON
//! deserialization failures, graph storage errors, semantically invalid
//! messages (e.g., referencing nonexistent nodes), and internal channel
//! closures.

pub mod batch;
pub mod error;
pub mod message;
pub mod sink;

pub use batch::BatchResult;
pub use error::KafkaError;
pub use message::GraphMutation;
pub use sink::{KafkaSink, SinkConfig, SinkMetrics};
