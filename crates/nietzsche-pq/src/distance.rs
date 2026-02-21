use crate::encoder::PQCode;
use crate::Codebook;

/// Pre-computed distance table for asymmetric distance computation.
/// table[m][k] = distance from query sub-vector m to centroid k.
pub struct DistanceTable {
    table: Vec<Vec<f32>>,
}

impl DistanceTable {
    /// Build a distance table for a query vector.
    pub fn build(query: &[f32], codebook: &Codebook) -> Self {
        let sub_dim = codebook.sub_dim;
        let mut table = Vec::with_capacity(codebook.config.m);

        for m in 0..codebook.config.m {
            let q_sub = &query[m * sub_dim..(m + 1) * sub_dim];
            let mut dists = Vec::with_capacity(codebook.config.k);

            for k in 0..codebook.config.k {
                let centroid = codebook.centroid(m, k);
                let d: f32 = q_sub
                    .iter()
                    .zip(centroid.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum();
                dists.push(d);
            }
            table.push(dists);
        }

        Self { table }
    }

    /// Number of sub-quantizer distance vectors in this table (equals M).
    pub fn num_sub_quantizers(&self) -> usize {
        self.table.len()
    }

    /// Number of centroid distances for a given sub-quantizer (equals K).
    pub fn num_centroids(&self, m: usize) -> usize {
        self.table[m].len()
    }

    /// Get the distance from query sub-vector `m` to centroid `k`.
    pub fn get(&self, m: usize, k: usize) -> f32 {
        self.table[m][k]
    }

    /// Compute approximate distance from query to a PQ code.
    pub fn distance(&self, code: &PQCode) -> f32 {
        code.codes
            .iter()
            .enumerate()
            .map(|(m, &k)| self.table[m][k as usize])
            .sum()
    }
}

/// Compute asymmetric distance between a raw query vector and a PQ code.
pub fn asymmetric_distance(query: &[f32], code: &PQCode, codebook: &Codebook) -> f32 {
    let table = DistanceTable::build(query, codebook);
    table.distance(code)
}
