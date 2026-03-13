// Copyright (C) 2025-2026 Jose R F Junior <web2ajax@gmail.com>
pub mod model;
pub mod encoder;
pub mod decoder;
pub mod error;

pub use model::{VqVae, VqVaeConfig};
pub use encoder::VqEncoder;
pub use decoder::VqDecoder;
pub use error::VqError;

pub type Result<T> = std::result::Result<T, VqError>;
