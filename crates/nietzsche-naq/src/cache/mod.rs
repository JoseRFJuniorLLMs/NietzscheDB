// Copyright (C) 2025-2026 Jose R F Junior <web2ajax@gmail.com>
//! # NAQ AST Cache
//!
//! LRU cache for parsed NQL queries. Parse once, execute forever.
//!
//! ## Problem
//!
//! `code_as_data` action nodes store NQL as strings and re-parse them
//! every time they fire. The same 10 action nodes firing every tick =
//! 10 PEG parses per tick, forever.
//!
//! ## Solution
//!
//! ```rust,no_run
//! use nietzsche_naq::cache::NaqCache;
//!
//! let mut cache = NaqCache::new(256);
//!
//! // First call: parses NQL (slow)
//! let ast = cache.get_or_parse("MATCH (n) WHERE n.energy > 0.5 RETURN n").unwrap();
//!
//! // Second call: instant LRU hit (zero parsing)
//! let ast = cache.get_or_parse("MATCH (n) WHERE n.energy > 0.5 RETURN n").unwrap();
//! ```

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;

use lru::LruCache;
use nietzsche_query::{Query, QueryError, parse};

/// LRU cache for parsed NQL queries.
///
/// Keyed by a fast hash of the NQL string. Caches up to `capacity`
/// parsed AST nodes. Thread-safety: wrap in `Mutex` or `RwLock` if
/// accessed from multiple threads.
pub struct NaqCache {
    cache: LruCache<u64, Query>,
    hits:  u64,
    misses: u64,
}

impl NaqCache {
    /// Create a new cache with the given capacity.
    pub fn new(capacity: usize) -> Self {
        let cap = NonZeroUsize::new(capacity.max(1)).unwrap();
        Self {
            cache:  LruCache::new(cap),
            hits:   0,
            misses: 0,
        }
    }

    /// Get a cached AST or parse the NQL string and cache the result.
    ///
    /// Returns a clone of the cached query — cheap because AST nodes
    /// are small value types.
    pub fn get_or_parse(&mut self, nql: &str) -> Result<Query, QueryError> {
        let hash = Self::hash_nql(nql);

        if let Some(cached) = self.cache.get(&hash) {
            self.hits += 1;
            return Ok(cached.clone());
        }

        self.misses += 1;
        let ast = parse(nql)?;
        self.cache.put(hash, ast.clone());
        Ok(ast)
    }

    /// Check if a query is already cached (without parsing).
    pub fn contains(&self, nql: &str) -> bool {
        let hash = Self::hash_nql(nql);
        self.cache.contains(&hash)
    }

    /// Pre-populate the cache with a known NQL string.
    /// Useful at startup to warm the cache with frequently-used action queries.
    pub fn warm(&mut self, nql: &str) -> Result<(), QueryError> {
        let hash = Self::hash_nql(nql);
        if !self.cache.contains(&hash) {
            let ast = parse(nql)?;
            self.cache.put(hash, ast);
        }
        Ok(())
    }

    /// Insert a pre-built AST into the cache under an NQL key.
    /// Useful when migrating from NQL strings to Builder API — cache the
    /// builder result so that code still using the NQL string path benefits.
    pub fn insert(&mut self, nql: &str, query: Query) {
        let hash = Self::hash_nql(nql);
        self.cache.put(hash, query);
    }

    /// Number of cached entries.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Returns `true` if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Cache hit count since creation.
    pub fn hits(&self) -> u64 {
        self.hits
    }

    /// Cache miss count since creation.
    pub fn misses(&self) -> u64 {
        self.misses
    }

    /// Hit ratio (0.0 – 1.0). Returns 0.0 if no queries yet.
    pub fn hit_ratio(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 { 0.0 } else { self.hits as f64 / total as f64 }
    }

    /// Clear the cache.
    pub fn clear(&mut self) {
        self.cache.clear();
        self.hits = 0;
        self.misses = 0;
    }

    fn hash_nql(nql: &str) -> u64 {
        let mut hasher = DefaultHasher::new();
        nql.hash(&mut hasher);
        hasher.finish()
    }
}

// ─────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_hit_miss() {
        let mut cache = NaqCache::new(16);

        let nql = "MATCH (n) RETURN n LIMIT 5";

        // Miss
        let q1 = cache.get_or_parse(nql).unwrap();
        assert_eq!(cache.hits(), 0);
        assert_eq!(cache.misses(), 1);
        assert_eq!(cache.len(), 1);

        // Hit
        let q2 = cache.get_or_parse(nql).unwrap();
        assert_eq!(cache.hits(), 1);
        assert_eq!(cache.misses(), 1);

        // Both should produce Match queries
        match (&q1, &q2) {
            (Query::Match(_), Query::Match(_)) => {}
            _ => panic!("expected Match"),
        }
    }

    #[test]
    fn cache_warm() {
        let mut cache = NaqCache::new(16);
        let nql = "MATCH (n) WHERE n.energy > 0.5 RETURN n";

        cache.warm(nql).unwrap();
        assert!(cache.contains(nql));
        assert_eq!(cache.len(), 1);

        // get_or_parse should hit cache
        let _q = cache.get_or_parse(nql).unwrap();
        assert_eq!(cache.hits(), 1);
        assert_eq!(cache.misses(), 0);
    }

    #[test]
    fn cache_eviction() {
        let mut cache = NaqCache::new(2);

        cache.get_or_parse("MATCH (n) RETURN n LIMIT 1").unwrap();
        cache.get_or_parse("MATCH (n) RETURN n LIMIT 2").unwrap();
        cache.get_or_parse("MATCH (n) RETURN n LIMIT 3").unwrap();

        // Capacity 2 — oldest (LIMIT 1) should be evicted
        assert_eq!(cache.len(), 2);
        assert!(!cache.contains("MATCH (n) RETURN n LIMIT 1"));
        assert!(cache.contains("MATCH (n) RETURN n LIMIT 3"));
    }

    #[test]
    fn cache_insert_builder_result() {
        use crate::builder::Naq;

        let mut cache = NaqCache::new(16);
        let nql = "MATCH (n) RETURN n LIMIT 10";
        let builder_ast = Naq::match_all(10);

        cache.insert(nql, builder_ast);
        assert!(cache.contains(nql));

        let q = cache.get_or_parse(nql).unwrap();
        match q {
            Query::Match(m) => assert_eq!(m.ret.limit, Some(10)),
            _ => panic!("expected Match"),
        }
        assert_eq!(cache.hits(), 1); // served from cache, no parsing
    }

    #[test]
    fn hit_ratio() {
        let mut cache = NaqCache::new(16);
        assert_eq!(cache.hit_ratio(), 0.0);

        let nql = "MATCH (n) RETURN n";
        cache.get_or_parse(nql).unwrap(); // miss
        cache.get_or_parse(nql).unwrap(); // hit
        cache.get_or_parse(nql).unwrap(); // hit
        cache.get_or_parse(nql).unwrap(); // hit

        // 3 hits / 4 total = 0.75
        assert!((cache.hit_ratio() - 0.75).abs() < f64::EPSILON);
    }
}
