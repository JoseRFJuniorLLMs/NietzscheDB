//! NietzscheDB MCP Server — Model Context Protocol integration.
//!
//! Exposes NietzscheDB capabilities as MCP tools that AI assistants
//! (Claude, Cursor, Windsurf, etc.) can discover and invoke.
//!
//! ## Protocol
//!
//! The MCP server communicates over **JSON-RPC 2.0** on stdin/stdout.
//! Each tool corresponds to a core NietzscheDB operation:
//!
//! | Tool | Description |
//! |------|-------------|
//! | `query` | Execute NQL queries |
//! | `insert_node` | Insert a new node |
//! | `get_node` | Retrieve a node by ID |
//! | `delete_node` | Delete a node by ID |
//! | `insert_edge` | Insert a new edge |
//! | `knn_search` | Hyperbolic k-NN search |
//! | `list_collections` | List all collections |
//! | `get_stats` | Get database statistics |
//! | `run_algorithm` | Run a graph algorithm |
//! | `diffuse` | Heat kernel diffusion from a seed node |

pub mod error;
pub mod protocol;
pub mod tools;
pub mod server;

pub use error::McpError;
pub use protocol::{JsonRpcRequest, JsonRpcResponse, McpTool, McpToolResult};
pub use server::McpServer;
