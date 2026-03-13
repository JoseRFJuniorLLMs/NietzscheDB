// Copyright (C) 2025-2026 Jose R F Junior <web2ajax@gmail.com>
pub mod indexer;
pub mod id;
pub mod error;
pub mod decoder;
pub mod pipeline;

pub use id::SemanticId;
pub use indexer::DsiIndexer;
pub use error::DsiError;
pub use decoder::{DsiDecoderNet, DsiResult};
pub use pipeline::{DsiPipeline, init_dsi_config, is_dsi_neural_enabled};

pub type Result<T> = std::result::Result<T, DsiError>;
