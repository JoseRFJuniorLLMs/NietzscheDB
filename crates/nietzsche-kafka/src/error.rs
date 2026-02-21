use thiserror::Error;

/// Errors produced by the Kafka sink connector.
#[derive(Debug, Error)]
pub enum KafkaError {
    /// Failed to deserialize a Kafka message payload into a [`GraphMutation`].
    #[error("deserialization error: {0}")]
    Deserialization(String),

    /// An operation against the underlying [`GraphStorage`] failed.
    #[error("graph storage error: {0}")]
    Graph(#[from] nietzsche_graph::GraphError),

    /// The message payload is syntactically valid JSON but semantically invalid
    /// (e.g., missing required fields, unknown mutation type).
    #[error("invalid message: {0}")]
    InvalidMessage(String),

    /// The internal channel used for async hand-off was closed unexpectedly.
    #[error("channel closed")]
    ChannelClosed,
}

impl From<serde_json::Error> for KafkaError {
    fn from(e: serde_json::Error) -> Self {
        Self::Deserialization(e.to_string())
    }
}
