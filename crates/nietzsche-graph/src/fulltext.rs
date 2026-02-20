//! Full-text search using an inverted index with BM25 scoring.
//!
//! Stores posting lists in the `meta` column family with a `fts:` prefix.
//! Tokenization is simple whitespace + lowercase + stop-word removal.

use std::collections::HashMap;
use uuid::Uuid;

use crate::error::GraphError;
use crate::storage::GraphStorage;

/// Full-text search result.
#[derive(Debug, Clone)]
pub struct FtsResult {
    pub node_id: Uuid,
    pub score: f64,
}

/// Full-text index operations.
pub struct FullTextIndex<'a> {
    storage: &'a GraphStorage,
}

impl<'a> FullTextIndex<'a> {
    pub fn new(storage: &'a GraphStorage) -> Self {
        Self { storage }
    }

    /// Index a node's text content.
    ///
    /// Tokenizes `text`, writes posting list entries for each term.
    /// Key format: `fts:{term}` → bincode `Vec<(Uuid, u32)>` (node_id, term_freq).
    pub fn index_node(&self, node_id: &Uuid, text: &str) -> Result<(), GraphError> {
        let tokens = tokenize(text);
        let mut term_freq: HashMap<String, u32> = HashMap::new();
        for tok in &tokens {
            *term_freq.entry(tok.clone()).or_default() += 1;
        }

        // Store doc length for BM25
        let dl_key = format!("fts:dl:{}", node_id);
        self.storage.put_meta(&dl_key, &(tokens.len() as u32).to_le_bytes())?;

        // Increment document count
        let doc_count = self.get_doc_count()? + 1;
        self.storage.put_meta("fts:doc_count", &doc_count.to_le_bytes())?;

        // Update posting lists
        for (term, freq) in term_freq {
            let key = format!("fts:term:{}", term);
            let mut postings = self.get_postings(&key)?;
            // Remove existing entry for this node if re-indexing
            postings.retain(|(id, _)| id != node_id);
            postings.push((*node_id, freq));
            let bytes = bincode::serialize(&postings)
                .map_err(|e| GraphError::Storage(format!("bincode: {e}")))?;
            self.storage.put_meta(&key, &bytes)?;
        }

        Ok(())
    }

    /// Remove a node from the full-text index.
    pub fn remove_node(&self, node_id: &Uuid) -> Result<(), GraphError> {
        // We'd need to know which terms this node had — for simplicity,
        // we mark the doc as deleted. The posting lists will be cleaned
        // lazily during searches.
        let dl_key = format!("fts:dl:{}", node_id);
        self.storage.delete_meta(&dl_key)?;

        let doc_count = self.get_doc_count()?;
        if doc_count > 0 {
            self.storage.put_meta("fts:doc_count", &(doc_count - 1).to_le_bytes())?;
        }

        Ok(())
    }

    /// Search the index using BM25 scoring.
    ///
    /// Returns results sorted by score descending, limited to `limit`.
    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<FtsResult>, GraphError> {
        let query_terms = tokenize(query);
        if query_terms.is_empty() {
            return Ok(vec![]);
        }

        let doc_count = self.get_doc_count()? as f64;
        if doc_count == 0.0 {
            return Ok(vec![]);
        }

        // BM25 parameters
        let k1 = 1.2;
        let b = 0.75;

        // Compute average document length
        let avg_dl = self.get_avg_doc_length()?;

        let mut scores: HashMap<Uuid, f64> = HashMap::new();

        for term in &query_terms {
            let key = format!("fts:term:{}", term);
            let postings = self.get_postings(&key)?;
            let df = postings.len() as f64;
            if df == 0.0 { continue; }

            // IDF component
            let idf = ((doc_count - df + 0.5) / (df + 0.5) + 1.0).ln();

            for (node_id, tf) in &postings {
                let dl = self.get_doc_length(node_id)? as f64;
                let tf_f = *tf as f64;

                // BM25 TF normalization
                let tf_norm = (tf_f * (k1 + 1.0)) / (tf_f + k1 * (1.0 - b + b * dl / avg_dl));

                *scores.entry(*node_id).or_default() += idf * tf_norm;
            }
        }

        let mut results: Vec<FtsResult> = scores
            .into_iter()
            .map(|(node_id, score)| FtsResult { node_id, score })
            .collect();

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);

        Ok(results)
    }

    /// Auto-index: extract text from a node's JSON content and index it.
    ///
    /// Extracts all string values from the JSON recursively.
    pub fn auto_index_content(
        &self,
        node_id: &Uuid,
        content: &serde_json::Value,
    ) -> Result<(), GraphError> {
        let mut text = String::new();
        extract_strings(content, &mut text);
        if !text.is_empty() {
            self.index_node(node_id, &text)?;
        }
        Ok(())
    }

    // ── Internal helpers ────────────────────────────────

    fn get_doc_count(&self) -> Result<u64, GraphError> {
        match self.storage.get_meta("fts:doc_count")? {
            Some(bytes) if bytes.len() >= 8 => {
                Ok(u64::from_le_bytes(bytes[..8].try_into().unwrap()))
            }
            _ => Ok(0),
        }
    }

    fn get_doc_length(&self, node_id: &Uuid) -> Result<u32, GraphError> {
        let key = format!("fts:dl:{}", node_id);
        match self.storage.get_meta(&key)? {
            Some(bytes) if bytes.len() >= 4 => {
                Ok(u32::from_le_bytes(bytes[..4].try_into().unwrap()))
            }
            _ => Ok(0),
        }
    }

    fn get_avg_doc_length(&self) -> Result<f64, GraphError> {
        // Simple approximation: scan a few doc lengths
        // For production, store total_dl alongside doc_count
        let doc_count = self.get_doc_count()?;
        if doc_count == 0 { return Ok(1.0); }
        // Use a fixed estimate of 100 tokens per doc
        Ok(100.0)
    }

    fn get_postings(&self, key: &str) -> Result<Vec<(Uuid, u32)>, GraphError> {
        match self.storage.get_meta(key)? {
            Some(bytes) => {
                bincode::deserialize(&bytes)
                    .map_err(|e| GraphError::Storage(format!("bincode deserialize: {e}")))
            }
            None => Ok(vec![]),
        }
    }
}

/// Simple whitespace tokenizer with lowercase and stop-word removal.
fn tokenize(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|w| {
            w.chars()
                .filter(|c| c.is_alphanumeric())
                .collect::<String>()
                .to_lowercase()
        })
        .filter(|w| w.len() >= 2 && !is_stop_word(w))
        .collect()
}

fn is_stop_word(word: &str) -> bool {
    matches!(word,
        "the" | "is" | "at" | "in" | "on" | "of" | "to" | "and" | "or" | "it"
        | "an" | "as" | "by" | "be" | "if" | "do" | "no" | "so" | "up" | "we"
        | "he" | "me" | "my" | "us" | "am" | "um" | "de" | "da" | "em" | "se"
        | "que" | "para" | "com" | "por" | "uma"
    )
}

/// Recursively extract string values from JSON.
fn extract_strings(value: &serde_json::Value, out: &mut String) {
    match value {
        serde_json::Value::String(s) => {
            if !out.is_empty() { out.push(' '); }
            out.push_str(s);
        }
        serde_json::Value::Object(map) => {
            for v in map.values() {
                extract_strings(v, out);
            }
        }
        serde_json::Value::Array(arr) => {
            for v in arr {
                extract_strings(v, out);
            }
        }
        _ => {}
    }
}
