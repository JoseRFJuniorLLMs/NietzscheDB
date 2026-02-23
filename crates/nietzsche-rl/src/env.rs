use ndarray::Array1;
use crate::Result;

/// RL state representation derived from graph health.
#[derive(Debug, Clone)]
pub struct GrowthState {
    pub features: Array1<f32>,
}

impl GrowthState {
    pub fn new(features: Vec<f32>) -> Self {
        Self {
            features: Array1::from_vec(features),
        }
    }
}

/// Action produced by the RL policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrowthAction {
    Balanced = 0,
    FavorGrowth = 1,
    FavorPruning = 2,
    Consolidate = 3,
}

pub struct GrowthEnv;

impl GrowthEnv {
    pub fn map_action(action_idx: usize) -> Result<GrowthAction> {
        match action_idx {
            0 => Ok(GrowthAction::Balanced),
            1 => Ok(GrowthAction::FavorGrowth),
            2 => Ok(GrowthAction::FavorPruning),
            3 => Ok(GrowthAction::Consolidate),
            _ => Err(crate::RlError::InvalidState(format!("Invalid action index: {}", action_idx))),
        }
    }
}
