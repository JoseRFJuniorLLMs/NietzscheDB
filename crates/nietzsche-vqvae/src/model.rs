// No imports needed here

#[derive(Debug, Clone)]
pub struct VqVaeConfig {
    pub model_name: String,
    pub embedding_dim: usize,
    pub num_embeddings: usize,
}

impl Default for VqVaeConfig {
    fn default() -> Self {
        Self {
            model_name: "vqvae_graph_v1".to_string(),
            embedding_dim: 128,
            num_embeddings: 512,
        }
    }
}

pub struct VqVae {
    config: VqVaeConfig,
}

impl VqVae {
    pub fn new(config: VqVaeConfig) -> Self {
        Self { config }
    }

    pub fn config(&self) -> &VqVaeConfig {
        &self.config
    }
}
