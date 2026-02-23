use crate::error::DreamError;
use nietzsche_graph::PoincareVector;
use nietzsche_neural::REGISTRY;
use ort::inputs;

pub struct WorldModel {
    model_name: String,
}

impl WorldModel {
    pub fn new(model_name: String) -> Self {
        Self { model_name }
    }

    /// Predict the next latent state (embedding) given current state and action.
    pub async fn predict_next(
        &self,
        current_embedding: &PoincareVector,
        action_id: u8,
    ) -> Result<PoincareVector, DreamError> {
        let session_arc = REGISTRY.get_session(&self.model_name)?;
        let mut session = session_arc.lock().map_err(|e| DreamError::Internal(e.to_string()))?;

        // 1. Prepare inputs
        // Shape: [batch=1, dim]
        let current_vec = current_embedding.coords.to_vec();
        let input_shape = vec![1, current_vec.len()];
        let current_value = ort::value::Value::from_array((input_shape, current_vec))?;

        // Action input: [batch=1, 1]
        let action_value = ort::value::Value::from_array((vec![1, 1], vec![action_id as f32]))?;

        // 2. Run session
        let outputs = session.run(inputs![current_value, action_value])?;

        // 3. Extract output (predicted next embedding)
        let output_value = outputs.iter().next()
            .map(|(_, v)| v)
            .ok_or_else(|| DreamError::Internal("No output from world model".into()))?;

        let (shape, data) = output_value.try_extract_tensor::<f32>()?;
        
        if data.is_empty() {
             return Err(DreamError::Internal("Empty world model output".into()));
        }

        Ok(PoincareVector::new(data.to_vec()))
    }
}
