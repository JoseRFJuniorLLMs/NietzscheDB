use crate::{Result, VqVaeConfig};
use nietzsche_neural::REGISTRY;
// No imports from ndarray needed for this specific implementation form

pub struct VqDecoder {
    config: VqVaeConfig,
}

impl VqDecoder {
    pub fn new(config: VqVaeConfig) -> Self {
        Self { config }
    }

    /// Decode a discrete latent index back into a continuous embedding.
    pub async fn decode(&self, index: usize) -> Result<Vec<f32>> {
        let session_arc = REGISTRY.get_session(&self.config.model_name)?;
        let mut session = session_arc.lock().map_err(|e| crate::VqError::Internal(e.to_string()))?;
        
        let shape = vec![1];
        let input_value = ort::value::Value::from_array((shape, vec![index as i64]))?;
        
        let outputs = session.run(ort::inputs![input_value])?;
        
        let embedding_value = outputs.iter().next()
            .map(|(_, v)| v)
            .ok_or_else(|| crate::VqError::Internal("No embedding output from model".to_string()))?;
            
        let (_, data) = embedding_value.try_extract_tensor::<f32>()?;
        
        Ok(data.to_vec())
    }
}
