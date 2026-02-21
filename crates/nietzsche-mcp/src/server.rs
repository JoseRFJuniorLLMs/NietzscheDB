//! MCP stdio server — reads JSON-RPC from stdin, writes to stdout.
//!
//! This implements the Model Context Protocol (MCP) server that AI
//! assistants use to discover and invoke NietzscheDB capabilities.

use std::sync::Arc;

use nietzsche_graph::{AdjacencyIndex, GraphStorage};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

use crate::error::McpError;
use crate::protocol::*;
use crate::tools;

/// MCP Server that communicates over stdin/stdout.
pub struct McpServer {
    storage: Arc<GraphStorage>,
    adjacency: Arc<AdjacencyIndex>,
}

impl McpServer {
    pub fn new(storage: Arc<GraphStorage>, adjacency: Arc<AdjacencyIndex>) -> Self {
        Self { storage, adjacency }
    }

    /// Run the MCP server loop (reads stdin line-by-line, writes JSON-RPC to stdout).
    pub async fn run(&self) -> Result<(), McpError> {
        let stdin = tokio::io::stdin();
        let mut stdout = tokio::io::stdout();
        let mut reader = BufReader::new(stdin);
        let mut line = String::new();

        loop {
            line.clear();
            let n = reader.read_line(&mut line).await?;
            if n == 0 { break; } // EOF

            let trimmed = line.trim();
            if trimmed.is_empty() { continue; }

            let response = self.handle_message(trimmed);
            let mut out = serde_json::to_string(&response)?;
            out.push('\n');
            stdout.write_all(out.as_bytes()).await?;
            stdout.flush().await?;
        }

        Ok(())
    }

    /// Handle a single JSON-RPC message and return the response.
    pub fn handle_message(&self, msg: &str) -> JsonRpcResponse {
        let req: JsonRpcRequest = match serde_json::from_str(msg) {
            Ok(r) => r,
            Err(e) => return JsonRpcResponse::error(None, -32700, format!("parse error: {e}")),
        };

        let id = req.id.clone();

        match req.method.as_str() {
            "initialize" => self.handle_initialize(id),
            "initialized" => {
                // Notification — no response needed, but we return one for simplicity
                JsonRpcResponse::success(id, serde_json::json!({}))
            }
            "tools/list" => self.handle_tools_list(id),
            "tools/call" => self.handle_tools_call(id, &req.params),
            "ping" => JsonRpcResponse::success(id, serde_json::json!({})),
            _ => JsonRpcResponse::error(id, -32601, format!("method not found: {}", req.method)),
        }
    }

    fn handle_initialize(&self, id: Option<serde_json::Value>) -> JsonRpcResponse {
        let result = InitializeResult {
            protocol_version: "2024-11-05".into(),
            capabilities: ServerCapabilities {
                tools: ToolCapability { list_changed: false },
            },
            server_info: ServerInfo {
                name: "nietzsche-mcp".into(),
                version: env!("CARGO_PKG_VERSION").into(),
            },
        };
        JsonRpcResponse::success(id, serde_json::to_value(&result).unwrap_or_default())
    }

    fn handle_tools_list(&self, id: Option<serde_json::Value>) -> JsonRpcResponse {
        let tool_list = tools::list_tools();
        JsonRpcResponse::success(id, serde_json::json!({ "tools": tool_list }))
    }

    fn handle_tools_call(
        &self,
        id: Option<serde_json::Value>,
        params: &serde_json::Value,
    ) -> JsonRpcResponse {
        let name = match params.get("name").and_then(|v| v.as_str()) {
            Some(n) => n,
            None => return JsonRpcResponse::error(id, -32602, "missing tool name".into()),
        };

        let args = params.get("arguments")
            .cloned()
            .unwrap_or(serde_json::json!({}));

        match tools::call_tool(name, &args, &self.storage, &self.adjacency) {
            Ok(result) => JsonRpcResponse::success(id, serde_json::to_value(&result).unwrap_or_default()),
            Err(e) => {
                let err_result = McpToolResult::error(e.to_string());
                JsonRpcResponse::success(id, serde_json::to_value(&err_result).unwrap_or_default())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_server() -> (tempfile::TempDir, McpServer) {
        let dir = tempfile::tempdir().unwrap();
        let storage = GraphStorage::open(dir.path().join("db").to_str().unwrap()).unwrap();
        let adjacency = AdjacencyIndex::new();
        let server = McpServer::new(Arc::new(storage), Arc::new(adjacency));
        (dir, server)
    }

    #[test]
    fn handle_initialize() {
        let (_dir, server) = test_server();
        let resp = server.handle_message(
            r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{}}}"#
        );
        assert!(resp.result.is_some());
        let r = resp.result.unwrap();
        assert_eq!(r["protocolVersion"], "2024-11-05");
        assert_eq!(r["serverInfo"]["name"], "nietzsche-mcp");
    }

    #[test]
    fn handle_tools_list() {
        let (_dir, server) = test_server();
        let resp = server.handle_message(
            r#"{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}"#
        );
        let r = resp.result.unwrap();
        let tools = r["tools"].as_array().unwrap();
        assert!(tools.len() >= 9);
        assert!(tools.iter().any(|t| t["name"] == "query"));
    }

    #[test]
    fn handle_tools_call_get_stats() {
        let (_dir, server) = test_server();
        let resp = server.handle_message(
            r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"get_stats","arguments":{}}}"#
        );
        let r = resp.result.unwrap();
        assert!(r["content"][0]["text"].as_str().unwrap().contains("node_count"));
    }

    #[test]
    fn handle_unknown_method() {
        let (_dir, server) = test_server();
        let resp = server.handle_message(
            r#"{"jsonrpc":"2.0","id":4,"method":"bogus","params":{}}"#
        );
        assert!(resp.error.is_some());
        assert_eq!(resp.error.unwrap().code, -32601);
    }

    #[test]
    fn handle_invalid_json() {
        let (_dir, server) = test_server();
        let resp = server.handle_message("not json");
        assert!(resp.error.is_some());
        assert_eq!(resp.error.unwrap().code, -32700);
    }

    #[test]
    fn handle_ping() {
        let (_dir, server) = test_server();
        let resp = server.handle_message(
            r#"{"jsonrpc":"2.0","id":5,"method":"ping","params":{}}"#
        );
        assert!(resp.result.is_some());
    }

    #[test]
    fn full_insert_query_roundtrip() {
        let (_dir, server) = test_server();

        // Insert a node
        let resp = server.handle_message(
            r#"{"jsonrpc":"2.0","id":10,"method":"tools/call","params":{"name":"insert_node","arguments":{"content":{"title":"mcp_test"},"energy":0.8}}}"#
        );
        let r = resp.result.unwrap();
        let text = r["content"][0]["text"].as_str().unwrap();
        let insert_resp: serde_json::Value = serde_json::from_str(text).unwrap();
        assert_eq!(insert_resp["status"], "inserted");

        // Query for it
        let resp = server.handle_message(
            r#"{"jsonrpc":"2.0","id":11,"method":"tools/call","params":{"name":"query","arguments":{"nql":"MATCH (n) WHERE n.energy > 0.5 RETURN n"}}}"#
        );
        let r = resp.result.unwrap();
        let text = r["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("mcp_test"));
    }
}
