use crate::{Result, GrowthAction, GrowthState, GrowthEnv};
use nietzsche_neural::REGISTRY;
use ndarray::Array2;

#[derive(Debug, Clone)]
pub struct PpoConfig {
    pub model_name: String,
    pub epsilon: f32, // for clipping, if doing training
}

impl Default for PpoConfig {
    fn default() -> Self {
        Self {
            model_name: "ppo_growth_v1".to_string(),
            epsilon: 0.2,
        }
    }
}

pub struct PpoEngine {
    config: PpoConfig,
}

impl PpoEngine {
    pub fn new(config: PpoConfig) -> Self {
        Self {
            config,
        }
    }

    /// Suggest the best growth action based on current state.
    pub async fn suggest_action(&self, state: &GrowthState) -> Result<GrowthAction> {
        // Run inference
        let session_arc = REGISTRY.get_session(&self.config.model_name)?;
        let mut session = session_arc.lock().map_err(|e| crate::RlError::Internal(e.to_string()))?;
        
        // Use (shape, vec) to avoid ndarray version conflicts in ort traits
        let features_vec = state.features.to_vec();
        let input_shape = vec![1, features_vec.len()];
        let input_value = ort::value::Value::from_array((input_shape, features_vec))?;
        
        let outputs = session.run(ort::inputs![input_value])?;
        
        // Extract logits from the first output using iterator
        let logits_value = outputs.iter().next()
            .map(|(_, v)| v)
            .ok_or_else(|| crate::RlError::Internal("No output from model".to_string()))?;
            
        let (shape, data) = logits_value.try_extract_tensor::<f32>()?;
        
        let shape_vec: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        if shape_vec.len() < 2 || *shape_vec.last().unwrap() < 4 {
            return Err(crate::RlError::Internal(format!("Invalid output shape: {:?}", shape_vec)));
        }

        let first_dim = shape_vec[0];
        let last_dim = *shape_vec.last().unwrap();
        
        let logits = Array2::from_shape_vec((first_dim, last_dim), data.to_vec())
            .map_err(|e| crate::RlError::Internal(e.to_string()))?;
        
        // Pick action with highest logit
        let mut best_action = 0;
        let mut max_logit = f32::NEG_INFINITY;
        
        for i in 0..4 {
            let val = logits[[0, i]];
            if val > max_logit {
                max_logit = val;
                best_action = i;
            }
        }

        GrowthEnv::map_action(best_action)
    }
}
