use thiserror::Error;

#[derive(Debug, Error)]
pub enum CuGraphError {
    #[error("node not found: {0}")]
    NodeNotFound(uuid::Uuid),

    #[error("CSR build failed: {0}")]
    CsrBuild(String),

    #[error("cuGraph library not found at {path}: {source}")]
    LibraryLoad {
        path: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[error("cuGraph API error (code {code}): {message}")]
    Api { code: i32, message: String },

    #[error("CUDA device error: {0}")]
    Cuda(String),

    #[error("NVRTC compilation error: {0}")]
    KernelCompile(String),

    #[error("result extraction failed: {0}")]
    ResultExtract(String),
}
