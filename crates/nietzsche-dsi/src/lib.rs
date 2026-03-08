pub mod indexer;
pub mod id;
pub mod error;
pub mod decoder;

pub use id::SemanticId;
pub use indexer::DsiIndexer;
pub use error::DsiError;
pub use decoder::{DsiDecoderNet, DsiResult};

pub type Result<T> = std::result::Result<T, DsiError>;
