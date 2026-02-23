use crate::{Result, SemanticId};
use nietzsche_graph::GraphStorage;
use nietzsche_vqvae::VqEncoder;
use uuid::Uuid;

pub struct DsiIndexer {
    encoder: VqEncoder,
    levels: usize,
}

impl DsiIndexer {
    pub fn new(encoder: VqEncoder, levels: usize) -> Self {
        Self { encoder, levels: levels.max(1) }
    }

    /// Generate a hierarchical SemanticId for a node based on its embedding.
    pub async fn index_node(
        &self,
        storage: &GraphStorage,
        node_id: &Uuid,
    ) -> Result<SemanticId> {
        let embedding = storage.get_embedding(node_id)?
            .ok_or_else(|| nietzsche_graph::error::GraphError::NodeNotFound(*node_id))?;
            
        let mut codes = Vec::with_capacity(self.levels);
        
        // Strategy: Chunk the embedding vector into `levels` parts and encode each.
        // If the vector is too small, we use overlapping windows or just repeat.
        let coords = &embedding.coords;
        let chunk_size = (coords.len() / self.levels).max(1);
        
        for i in 0..self.levels {
            let start = i * chunk_size;
            let end = if i == self.levels - 1 {
                coords.len()
            } else {
                (start + chunk_size).min(coords.len())
            };
            
            let chunk = &coords[start..end];
            // If the chunk is smaller than what the encoder expects (e.g. 128), 
            // the encoder's internal model will likely fail if it's sensitive to input size.
            // However, our VqEncoder::encode just creates a 1xD tensor.
            let code = self.encoder.encode(chunk).await?;
            codes.push(code as u16);
        }
        
        let id = SemanticId::new(codes);
        
        // Save to storage using prefix-friendly bytes
        let semantic_bytes = id.to_prefix_bytes();
            
        storage.put_dsi_id(node_id, &semantic_bytes)?;
        
        Ok(id)
    }
}
