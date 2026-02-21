use thiserror::Error;

#[derive(Debug, Error)]
pub enum MediaError {
    #[error("opendal error: {0}")]
    OpenDal(#[from] opendal::Error),
    #[error("media not found: {0}")]
    NotFound(String),
    #[error("invalid media type: {0}")]
    InvalidMediaType(String),
    #[error("metadata error: {0}")]
    Metadata(String),
}
