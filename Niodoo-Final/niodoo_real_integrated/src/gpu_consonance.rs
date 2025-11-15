//! GPU-Accelerated Consonance Calculations for RTX 5090
//!
//! Vectorized consonance computations using GPU for maximum throughput

#[cfg(feature = "gpu")]
use candle_core::{DType, Device, Result, Tensor};
#[cfg(feature = "gpu")]
use tokio::task;
#[cfg(feature = "gpu")]
use tracing::{debug, info};

/// GPU-accelerated consonance calculator
#[cfg(feature = "gpu")]
pub struct GpuConsonanceCalculator {
    device: Device,
}

#[cfg(feature = "gpu")]
impl GpuConsonanceCalculator {
    /// Create GPU consonance calculator
    pub fn new(device: Device) -> Self {
        Self { device }
    }

    /// Batch compute PAD variance for multiple states
    pub async fn batch_pad_variance(&self, pad_states: &[[f64; 7]]) -> Result<Vec<f64>> {
        if pad_states.is_empty() {
            return Ok(Vec::new());
        }

        let device = self.device.clone();
        let pad_states_vec = pad_states.to_vec();

        task::spawn_blocking(move || {
            let batch_size = pad_states_vec.len();
            let mut pad_values = Vec::with_capacity(batch_size * 3);

            for state in &pad_states_vec {
                pad_values.push(state[0] as f32);
                pad_values.push(state[1] as f32);
                pad_values.push(state[2] as f32);
            }

            // Reshape to [batch, 3]
            let pad_tensor = Tensor::from_vec(pad_values, (batch_size, 3), &device)?;

            // Compute means: [batch, 1]
            let means = pad_tensor
                .sum_keepdim(1)?
                .broadcast_div(&Tensor::new(&[3.0f32], &device)?)?;

            // Compute variances: [batch, 1]
            let centered = pad_tensor.sub(&means.expand((batch_size, 3))?)?;
            let variances = centered
                .sqr()?
                .sum_keepdim(1)?
                .broadcast_div(&Tensor::new(&[3.0f32], &device)?)?;

            // Compute std dev: sqrt of variance
            let std_devs = variances.sqrt()?;

            // Convert to CPU
            let std_vec = std_devs.to_vec1::<f32>()?;
            Ok(std_vec.into_iter().map(|v| v as f64).collect())
        })
        .await
        .map_err(|e| anyhow::anyhow!("GPU task panicked: {}", e))?
    }

    /// Batch compute weighted consonance scores
    pub async fn batch_weighted_consonance(
        &self,
        source_scores: &[[f64; 5]], // [batch, 5 sources]
        weights: &[f64; 5],
    ) -> Result<Vec<f64>> {
        if source_scores.is_empty() {
            return Ok(Vec::new());
        }

        let device = self.device.clone();
        let source_scores_vec = source_scores.to_vec();
        let weights_vec = weights.to_vec();

        task::spawn_blocking(move || {
            let batch_size = source_scores_vec.len();
            let mut flat_scores = Vec::with_capacity(batch_size * 5);

            for scores in &source_scores_vec {
                for score in scores {
                    flat_scores.push(*score as f32);
                }
            }

            // Reshape to [batch, 5]
            let scores_tensor = Tensor::from_vec(flat_scores, (batch_size, 5), &device)?;

            // Weight matrix: [5, 1]
            let weight_vec: Vec<f32> = weights_vec.iter().map(|w| *w as f32).collect();
            let weight_matrix = Tensor::new(weight_vec.as_slice(), (5, 1), &device)?;

            // Single matmul: [batch, 5] @ [5, 1] -> [batch, 1]
            let weighted = scores_tensor.matmul(&weight_matrix)?;

            // Clamp to [0, 1]
            let zero = Tensor::zeros((batch_size, 1), DType::F32, &device)?;
            let one = Tensor::ones((batch_size, 1), DType::F32, &device)?;
            let clamped = weighted.maximum(&zero)?.minimum(&one)?;

            // Convert to CPU
            let result_vec = clamped.reshape((batch_size,))?.to_vec1::<f32>()?;
            Ok(result_vec.into_iter().map(|v| v as f64).collect())
        })
        .await
        .map_err(|e| anyhow::anyhow!("GPU task panicked: {}", e))?
    }

    /// Batch compute cosine similarities
    pub async fn batch_cosine_similarity(
        &self,
        vectors_a: &[Vec<f32>],
        vectors_b: &[Vec<f32>],
    ) -> Result<Vec<f32>> {
        if vectors_a.len() != vectors_b.len() || vectors_a.is_empty() {
            return Ok(Vec::new());
        }

        let device = self.device.clone();
        let vectors_a_vec = vectors_a.to_vec();
        let vectors_b_vec = vectors_b.to_vec();

        task::spawn_blocking(move || {
            let batch_size = vectors_a_vec.len();
            let dim = vectors_a_vec[0].len();

            // Flatten and stack
            let mut flat_a = Vec::with_capacity(batch_size * dim);
            let mut flat_b = Vec::with_capacity(batch_size * dim);

            for vec in &vectors_a_vec {
                flat_a.extend(vec.iter());
            }
            for vec in &vectors_b_vec {
                flat_b.extend(vec.iter());
            }

            let a_tensor = Tensor::from_vec(flat_a, (batch_size, dim), &device)?;
            let b_tensor = Tensor::from_vec(flat_b, (batch_size, dim), &device)?;

            // Compute dot products: [batch]
            let dots = a_tensor.broadcast_mul(&b_tensor)?.sum_keepdim(1)?;

            // Compute norms
            let norms_a = a_tensor.sqr()?.sum_keepdim(1)?.sqrt()?;
            let norms_b = b_tensor.sqr()?.sum_keepdim(1)?.sqrt()?;

            // Cosine similarity: dot / (norm_a * norm_b)
            let norms_product = norms_a.broadcast_mul(&norms_b)?;
            let eps = Tensor::new(&[1e-8f32], &device)?;
            let norms_safe = norms_product.maximum(&eps)?;
            let similarities = dots.broadcast_div(&norms_safe)?;

            // Convert to CPU
            similarities.reshape((batch_size,))?.to_vec1::<f32>()
        })
        .await
        .map_err(|e| anyhow::anyhow!("GPU task panicked: {}", e))?
    }
}

#[cfg(test)]
#[cfg(feature = "gpu")]
mod tests {
    use super::*;

    #[test]
    fn test_batch_pad_variance() {
        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
        let calculator = GpuConsonanceCalculator::new(device);

        let pad_states = vec![[0.5, 0.8, 0.3, 0.0, 0.0, 0.0, 0.0]];
        let variances = calculator.batch_pad_variance(&pad_states).unwrap();
        assert_eq!(variances.len(), 1);
    }
}
