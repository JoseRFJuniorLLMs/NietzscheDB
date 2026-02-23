use crate::{Result, VqVaeConfig};
use nietzsche_neural::REGISTRY;
// No imports from ndarray needed for this specific implementation form

pub struct VqEncoder {
    config: VqVaeConfig,
}

impl VqEncoder {
    pub fn new(config: VqVaeConfig) -> Self {
        Self { config }
    }

    /// Encode a continuous embedding into a discrete latent index.
    pub async fn encode(&self, embedding: &[f32]) -> Result<usize> {
        let session_arc = REGISTRY.get_session(&self.config.model_name)?;
        let mut session = session_arc.lock().map_err(|e| crate::VqError::Internal(e.to_string()))?;
        
        let shape = vec![1, embedding.len()];
        let input_value = ort::value::Value::from_array((shape, embedding.to_vec()))?;
        
        // Use the encoder-specific output (often named "indices" or similar)
        let outputs = session.run(ort::inputs![input_value])?;
        
        // We look for discrete indices
        let indices_value = outputs.iter().next()
            .map(|(_, v)| v)
            .ok_or_else(|| crate::VqError::Internal("No indices output from model".to_string()))?;
            
        let (_, data) = indices_value.try_extract_tensor::<i64>()?;
        
        if data.is_empty() {
            return Err(crate::VqError::Internal("Empty indices output".to_string()));
        }

        Ok(data[0] as usize)
    }
}
