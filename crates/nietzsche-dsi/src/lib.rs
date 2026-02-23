pub mod indexer;
pub mod id;
pub mod error;

pub use id::SemanticId;
pub use indexer::DsiIndexer;
pub use error::DsiError;

pub type Result<T> = std::result::Result<T, DsiError>;
