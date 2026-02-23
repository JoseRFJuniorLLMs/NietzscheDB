use ndarray::ArrayD;
use crate::{Result, NeuralError};
use ort::value::DynValue;

pub fn poincare_constraint(mut tensor: ArrayD<f32>) -> ArrayD<f32> {
    tensor.mapv_inplace(|x| {
        // Simple element-wise constraint is not enough for Poincare ball,
        // we need to constrain the norm of the vector.
        // This function assumes the last dimension is the embedding dimension.
        x
    });
    
    // For each vector in the tensor, project it to the Poincare ball.
    // Assuming tensor shape is [..., dim]
    let shape = tensor.shape().to_vec();
    let dim = *shape.last().unwrap();
    let flat_len = tensor.len();
    let num_vectors = flat_len / dim;
    
    let (mut data, _) = tensor.into_raw_vec_and_offset();
    for i in 0..num_vectors {
        let start = i * dim;
        let end = start + dim;
        let vec = &mut data[start..end];
        
        let norm_sq: f32 = vec.iter().map(|x| x * x).sum();
        let norm = norm_sq.sqrt();
        
        if norm >= 1.0 {
            let scale = 0.999 / norm;
            for val in vec.iter_mut() {
                *val *= scale;
            }
        }
    }
    
    ArrayD::from_shape_vec(shape.to_vec(), data).unwrap()
}

pub fn to_ndarray(tensor: &DynValue) -> Result<ArrayD<f32>> {
    let view = tensor.try_extract_tensor::<f32>()?;
    let shape: Vec<usize> = view.0.iter().map(|&d| d as usize).collect();
    let data = view.1.to_vec();
    Ok(ArrayD::from_shape_vec(shape, data).unwrap())
}

pub fn from_ndarray(array: ArrayD<f32>) -> Result<ort::value::Value> {
    let shape = array.shape().to_vec();
    let (data, _) = array.into_raw_vec_and_offset();
    ort::value::Value::from_array((shape, data))
        .map(|v| v.into_dyn())
        .map_err(|e| NeuralError::OrtError(e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_tensor_roundtrip() {
        let arr = array![[1.0f32, 2.0], [3.0, 4.0]].into_dyn();
        let val = from_ndarray(arr.clone()).unwrap();
        let back = to_ndarray(&val.into_dyn()).unwrap();
        assert_eq!(arr, back);
    }

    #[test]
    fn test_poincare_constraint() {
        let arr = array![[2.0f32, 0.0], [0.0, 0.5]].into_dyn();
        let constrained = poincare_constraint(arr.clone());
        
        // First vector [2.0, 0.0] has norm 2.0, should be scaled to 0.999
        let v1 = constrained.slice(ndarray::s![0, ..]);
        assert!((v1[[0usize]] - 0.999).abs() < 1e-6);
        assert_eq!(v1[[1usize]], 0.0);
        
        // Second vector [0.0, 0.5] has norm 0.5, should stay same
        let v2 = constrained.slice(ndarray::s![1, ..]);
        assert_eq!(v2[[0usize]], 0.0);
        assert_eq!(v2[[1usize]], 0.5);
    }
}
