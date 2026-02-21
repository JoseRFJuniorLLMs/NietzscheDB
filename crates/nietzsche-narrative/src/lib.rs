//! # nietzsche-narrative
//!
//! **Narrative Engine** — story arc detection and generation from graph evolution.
//!
//! Compiles graph state changes, energy distributions, and structural patterns
//! into human-readable narratives. Detects emergence, conflict, decay, and
//! recurrence arcs.
//!
//! ## NQL examples
//!
//! ```text
//! NARRATE IN "memories" WINDOW 24 FORMAT json
//! NARRATE WINDOW 168 FORMAT text
//! NARRATE
//! ```

pub mod engine;
pub mod error;
pub mod model;

pub use engine::{NarrativeConfig, NarrativeEngine};
pub use error::NarrativeError;
pub use model::{NarrativeEvent, NarrativeEventType, NarrativeReport, NarrativeStats};
