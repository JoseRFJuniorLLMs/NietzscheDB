//! MCP JSON-RPC protocol types.
//!
//! Implements the Model Context Protocol v1 specification:
//! - `initialize` / `initialized` handshake
//! - `tools/list` — discovery
//! - `tools/call` — invocation

use serde::{Deserialize, Serialize};

/// JSON-RPC 2.0 request envelope.
#[derive(Debug, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    pub id: Option<serde_json::Value>,
    pub method: String,
    #[serde(default)]
    pub params: serde_json::Value,
}

/// JSON-RPC 2.0 response envelope.
#[derive(Debug, Serialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

#[derive(Debug, Serialize)]
pub struct JsonRpcError {
    pub code: i64,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

impl JsonRpcResponse {
    pub fn success(id: Option<serde_json::Value>, result: serde_json::Value) -> Self {
        Self { jsonrpc: "2.0".into(), id, result: Some(result), error: None }
    }

    pub fn error(id: Option<serde_json::Value>, code: i64, message: String) -> Self {
        Self {
            jsonrpc: "2.0".into(), id, result: None,
            error: Some(JsonRpcError { code, message, data: None }),
        }
    }
}

/// MCP tool descriptor for `tools/list` response.
#[derive(Debug, Clone, Serialize)]
pub struct McpTool {
    pub name: String,
    pub description: String,
    #[serde(rename = "inputSchema")]
    pub input_schema: serde_json::Value,
}

/// Result from a tool invocation.
#[derive(Debug, Serialize)]
pub struct McpToolResult {
    pub content: Vec<McpContent>,
    #[serde(rename = "isError", skip_serializing_if = "std::ops::Not::not")]
    pub is_error: bool,
}

#[derive(Debug, Serialize)]
pub struct McpContent {
    #[serde(rename = "type")]
    pub content_type: String,
    pub text: String,
}

impl McpToolResult {
    pub fn text(s: String) -> Self {
        Self {
            content: vec![McpContent { content_type: "text".into(), text: s }],
            is_error: false,
        }
    }

    pub fn error(s: String) -> Self {
        Self {
            content: vec![McpContent { content_type: "text".into(), text: s }],
            is_error: true,
        }
    }
}

/// MCP server capabilities (returned in `initialize` response).
#[derive(Debug, Serialize)]
pub struct ServerCapabilities {
    pub tools: ToolCapability,
}

#[derive(Debug, Serialize)]
pub struct ToolCapability {
    #[serde(rename = "listChanged")]
    pub list_changed: bool,
}

/// MCP server info.
#[derive(Debug, Serialize)]
pub struct ServerInfo {
    pub name: String,
    pub version: String,
}

/// Full `initialize` response result.
#[derive(Debug, Serialize)]
pub struct InitializeResult {
    #[serde(rename = "protocolVersion")]
    pub protocol_version: String,
    pub capabilities: ServerCapabilities,
    #[serde(rename = "serverInfo")]
    pub server_info: ServerInfo,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserialize_request() {
        let json = r#"{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}"#;
        let req: JsonRpcRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.method, "tools/list");
        assert_eq!(req.id, Some(serde_json::json!(1)));
    }

    #[test]
    fn serialize_success_response() {
        let resp = JsonRpcResponse::success(Some(serde_json::json!(1)), serde_json::json!({"ok": true}));
        let s = serde_json::to_string(&resp).unwrap();
        assert!(s.contains("\"result\""));
        assert!(!s.contains("\"error\""));
    }

    #[test]
    fn serialize_error_response() {
        let resp = JsonRpcResponse::error(Some(serde_json::json!(1)), -32600, "bad".into());
        let s = serde_json::to_string(&resp).unwrap();
        assert!(s.contains("\"error\""));
        assert!(!s.contains("\"result\""));
    }

    #[test]
    fn tool_result_text() {
        let r = McpToolResult::text("hello".into());
        assert!(!r.is_error);
        assert_eq!(r.content[0].text, "hello");
    }

    #[test]
    fn tool_result_error() {
        let r = McpToolResult::error("fail".into());
        assert!(r.is_error);
    }

    #[test]
    fn initialize_result_serializes() {
        let init = InitializeResult {
            protocol_version: "2024-11-05".into(),
            capabilities: ServerCapabilities {
                tools: ToolCapability { list_changed: false },
            },
            server_info: ServerInfo {
                name: "nietzsche-mcp".into(),
                version: "0.1.0".into(),
            },
        };
        let json = serde_json::to_string(&init).unwrap();
        assert!(json.contains("protocolVersion"));
        assert!(json.contains("nietzsche-mcp"));
    }
}
