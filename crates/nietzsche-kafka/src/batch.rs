use crate::error::KafkaError;

/// Result of processing a batch of [`GraphMutation`] messages.
///
/// Tracks how many mutations succeeded and failed, along with detailed
/// error information for each failure keyed by its index in the batch.
#[derive(Debug)]
pub struct BatchResult {
    /// Number of mutations that were applied successfully.
    pub succeeded: usize,
    /// Number of mutations that failed.
    pub failed: usize,
    /// Per-message errors: `(index_in_batch, error)`.
    pub errors: Vec<(usize, KafkaError)>,
}

impl BatchResult {
    /// Create an empty batch result (no messages processed yet).
    pub fn empty() -> Self {
        Self {
            succeeded: 0,
            failed: 0,
            errors: Vec::new(),
        }
    }

    /// Record a successful mutation.
    pub fn record_success(&mut self) {
        self.succeeded += 1;
    }

    /// Record a failed mutation at the given batch index.
    pub fn record_failure(&mut self, index: usize, error: KafkaError) {
        self.failed += 1;
        self.errors.push((index, error));
    }

    /// Returns true if all messages in the batch were processed successfully.
    pub fn is_all_ok(&self) -> bool {
        self.failed == 0
    }

    /// Total number of messages processed (succeeded + failed).
    pub fn total(&self) -> usize {
        self.succeeded + self.failed
    }
}
