//! GPU-accelerated fitness calculation for WeightedEpisodicMem
//!
//! Provides batch fitness calculation with optional GPU acceleration.
//! Falls back to CPU if GPU is unavailable or feature is disabled.

#[cfg(feature = "gpu")]
use crate::gpu_fusion::GpuTensorFusion;
use crate::torus::PadGhostState;
use crate::weighted_episodic_mem::{
    calculate_fitness_score, calculate_pad_salience, calculate_temporal_decay,
    normalized_retrieval_weight, TemporalDecayConfig,
};
use anyhow::Result;
#[cfg(feature = "gpu")]
use candle_core::{DType, Device, Tensor};
use ndarray::Array1;
use tracing::info;
#[cfg(feature = "gpu")]
use tracing::warn;

/// GPU fitness calculator with CPU fallback
pub struct GPUMemoryFitnessCalculator {
    /// Device preference ("cuda", "cpu", or "auto")
    device: String,
    /// Whether GPU is available
    gpu_available: bool,
    /// GPU memory pool for tensor reuse (RTX 5090 optimization)
    #[cfg(feature = "gpu")]
    memory_pool: Option<crate::gpu_memory_pool::GpuMemoryPool>,
}

impl GPUMemoryFitnessCalculator {
    /// Create new calculator with device preference
    pub fn new(device: &str) -> Self {
        let gpu_available = Self::check_gpu_available();
        info!(
            "GPU fitness calculator initialized: device={}, gpu_available={}",
            device, gpu_available
        );

        // Phase 5.2: Record GPU availability metric
        crate::metrics::gpu_fitness_metrics().set_gpu_available(gpu_available);

        // RTX 5090 OPTIMIZATION: Initialize GPU memory pool for tensor reuse
        #[cfg(feature = "gpu")]
        let memory_pool = if gpu_available && device == "cuda" {
            if let Ok(cuda_device) = Device::cuda_if_available(0) {
                Some(crate::gpu_memory_pool::GpuMemoryPool::new(cuda_device, 200))
            // 200 tensors per bucket
            } else {
                None
            }
        } else {
            None
        };

        Self {
            device: device.to_string(),
            gpu_available,
            #[cfg(feature = "gpu")]
            memory_pool,
        }
    }

    /// Check if GPU is available
    #[cfg(feature = "gpu")]
    fn check_gpu_available() -> bool {
        Device::cuda_if_available(0).is_ok()
    }

    #[cfg(not(feature = "gpu"))]
    fn check_gpu_available() -> bool {
        false
    }

    /// Batch calculate fitness scores for multiple memories
    ///
    /// Processes memories in parallel batches for maximum throughput.
    /// Falls back to CPU if GPU unavailable.
    pub fn batch_fitness(
        &self,
        memories: &[(PadGhostState, f64, u32, f32, f32, f32)], // (pad_state, age_days, retrieval_count, beta1, consonance, consolidation_level)
        weights: &[f32; 6],
        temporal_config: &TemporalDecayConfig,
    ) -> Vec<f32> {
        let start = std::time::Instant::now();
        let batch_size = memories.len();

        let result = if self.gpu_available && self.device == "cuda" {
            let res = self._batch_fitness_gpu(memories, weights, temporal_config);
            let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
            crate::metrics::gpu_fitness_metrics().record_gpu_calculation(batch_size, latency_ms);
            res
        } else {
            let res = self._batch_fitness_cpu(memories, weights, temporal_config);
            let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
            crate::metrics::gpu_fitness_metrics().record_cpu_fallback(batch_size, latency_ms);
            res
        };

        result
    }

    /// CPU-based batch fitness calculation
    fn _batch_fitness_cpu(
        &self,
        memories: &[(PadGhostState, f64, u32, f32, f32, f32)],
        weights: &[f32; 6],
        temporal_config: &TemporalDecayConfig,
    ) -> Vec<f32> {
        use rayon::prelude::*;

        memories
            .par_iter()
            .map(
                |(pad_state, age_days, retrieval_count, beta1, consonance, consolidation_level)| {
                    calculate_fitness_score(
                        *age_days,
                        pad_state,
                        *retrieval_count,
                        *beta1,
                        *consonance,
                        *consolidation_level,
                        weights,
                        temporal_config,
                        None, // resource_availability
                        None, // persistence_entropy
                        None, // h1_trust_score
                        None, // h2_anomaly_score
                    )
                },
            )
            .collect()
    }

    /// GPU-based batch fitness calculation (placeholder for future implementation)
    #[cfg(feature = "gpu")]
    fn _batch_fitness_gpu(
        &self,
        memories: &[(PadGhostState, f64, u32, f32, f32, f32)],
        weights: &[f32; 6],
        temporal_config: &TemporalDecayConfig,
    ) -> Vec<f32> {
        match self.batch_fitness_gpu_impl(memories, weights, temporal_config) {
            Ok(scores) => scores,
            Err(error) => {
                warn!(?error, "GPU fitness calculation failed; using CPU fallback");
                self._batch_fitness_cpu(memories, weights, temporal_config)
            }
        }
    }

    #[cfg(not(feature = "gpu"))]
    fn _batch_fitness_gpu(
        &self,
        memories: &[(PadGhostState, f64, u32, f32, f32, f32)],
        weights: &[f32; 6],
        temporal_config: &TemporalDecayConfig,
    ) -> Vec<f32> {
        self._batch_fitness_cpu(memories, weights, temporal_config)
    }

    #[cfg(feature = "gpu")]
    fn batch_fitness_gpu_impl(
        &self,
        memories: &[(PadGhostState, f64, u32, f32, f32, f32)],
        weights: &[f32; 6],
        temporal_config: &TemporalDecayConfig,
    ) -> candle_core::Result<Vec<f32>> {
        if memories.is_empty() {
            return Ok(Vec::new());
        }

        let device = Device::cuda_if_available(0)?;
        let batch = memories.len();

        let mut temporal = Vec::with_capacity(batch);
        let mut pad_salience = Vec::with_capacity(batch);
        let mut retrieval = Vec::with_capacity(batch);
        let mut beta1_norm = Vec::with_capacity(batch);
        let mut consonance_norm = Vec::with_capacity(batch);

        for (pad_state, age_days, retrieval_count, beta1, consonance, consolidation_level) in
            memories
        {
            temporal.push(calculate_temporal_decay(
                *age_days,
                *consolidation_level,
                temporal_config,
            ));
            pad_salience.push(calculate_pad_salience(pad_state));
            retrieval.push(normalized_retrieval_weight(*retrieval_count));
            beta1_norm.push(beta1.clamp(0.0, 1.0));
            consonance_norm.push(consonance.clamp(0.0, 1.0));
        }

        let temporal_t = Tensor::from_vec(temporal, (batch,), &device)?;
        let pad_t = Tensor::from_vec(pad_salience, (batch,), &device)?;
        let retrieval_t = Tensor::from_vec(retrieval, (batch,), &device)?;
        let beta1_t = Tensor::from_vec(beta1_norm, (batch,), &device)?;
        let consonance_t = Tensor::from_vec(consonance_norm, (batch,), &device)?;

        // RTX 5090 OPTIMIZATION: Use fused tensor operations for maximum GPU utilization
        let fusion = GpuTensorFusion::new(device);
        let fitness = fusion.fused_fitness_calculation(
            &temporal_t,
            &pad_t,
            &retrieval_t,
            &beta1_t,
            &consonance_t,
            weights,
        )?;

        // Stay on GPU as long as possible - only move to CPU for final result extraction
        // This maximizes GPU utilization on RTX 5090
        let scores = fitness.to_vec1::<f32>()?;
        Ok(scores)
    }

    /// Batch calculate fitness from embeddings and metadata
    ///
    /// Higher-level interface that takes embeddings and metadata arrays
    pub fn batch_fitness_from_arrays(
        &self,
        pad_states: &[PadGhostState],
        ages: &[f64],
        retrieval_counts: &[u32],
        beta1_scores: &[f32],
        consonance_scores: &[f32],
        consolidation_levels: &[f32],
        weights: &[f32; 6],
        temporal_config: &TemporalDecayConfig,
    ) -> Vec<f32> {
        if pad_states.len() != ages.len()
            || ages.len() != retrieval_counts.len()
            || retrieval_counts.len() != beta1_scores.len()
            || beta1_scores.len() != consonance_scores.len()
            || consonance_scores.len() != consolidation_levels.len()
        {
            return Vec::new(); // Return empty on mismatch
        }

        let memories: Vec<_> = pad_states
            .iter()
            .zip(ages.iter())
            .zip(retrieval_counts.iter())
            .zip(beta1_scores.iter())
            .zip(consonance_scores.iter())
            .zip(consolidation_levels.iter())
            .map(|(((((pad, age), ret), beta1), cons), cons_level)| {
                (pad.clone(), *age, *ret, *beta1, *cons, *cons_level)
            })
            .collect();

        self.batch_fitness(&memories, weights, temporal_config)
    }

    /// Optimized batch calculation using ndarray for vectorization
    pub fn batch_fitness_ndarray(
        &self,
        pad_states: &[PadGhostState],
        ages: &Array1<f64>,
        retrieval_counts: &Array1<u32>,
        beta1_scores: &Array1<f32>,
        consonance_scores: &Array1<f32>,
        consolidation_levels: &Array1<f32>,
        weights: &[f32; 6],
        temporal_config: &TemporalDecayConfig,
    ) -> Array1<f32> {
        if pad_states.len() != ages.len()
            || ages.len() != retrieval_counts.len()
            || retrieval_counts.len() != beta1_scores.len()
            || beta1_scores.len() != consonance_scores.len()
            || consonance_scores.len() != consolidation_levels.len()
        {
            return Array1::from_vec(vec![]); // Return empty on mismatch
        }

        // Convert to vectors for processing
        let ages_vec: Vec<f64> = ages.iter().copied().collect();
        let retrieval_vec: Vec<u32> = retrieval_counts.iter().copied().collect();
        let beta1_vec: Vec<f32> = beta1_scores.iter().copied().collect();
        let consonance_vec: Vec<f32> = consonance_scores.iter().copied().collect();
        let cons_level_vec: Vec<f32> = consolidation_levels.iter().copied().collect();

        let memories: Vec<_> = pad_states
            .iter()
            .zip(ages_vec.iter())
            .zip(retrieval_vec.iter())
            .zip(beta1_vec.iter())
            .zip(consonance_vec.iter())
            .zip(cons_level_vec.iter())
            .map(|(((((pad, age), ret), beta1), cons), cons_level)| {
                (pad.clone(), *age, *ret, *beta1, *cons, *cons_level)
            })
            .collect();

        let fitness_scores = self.batch_fitness(&memories, weights, temporal_config);
        Array1::from_vec(fitness_scores)
    }

    /// Refresh metrics and emit telemetry about GPU availability
    pub async fn refresh_metrics(&self) -> Result<()> {
        if self.gpu_available && self.device == "cuda" {
            info!(device = %self.device, "GPU fitness metrics refreshed: CUDA available");
        } else if self.gpu_available {
            info!(device = %self.device, "GPU fitness metrics refreshed: GPU detected, using fallback device");
        } else {
            info!(device = %self.device, "GPU fitness metrics refreshed: CPU fallback active");
        }
        Ok(())
    }
}

impl Default for GPUMemoryFitnessCalculator {
    fn default() -> Self {
        Self::new("cpu")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::torus::PadGhostState;
    use crate::weighted_episodic_mem::DEFAULT_FITNESS_WEIGHTS;

    fn create_test_pad_state() -> PadGhostState {
        PadGhostState {
            pad: [0.5, 0.8, 0.3, 0.0, 0.0, 0.0, 0.0],
            entropy: 0.5,
            mu: [0.5, 0.8, 0.3, 0.0, 0.0, 0.0, 0.0],
            sigma: [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        }
    }

    #[test]
    fn test_batch_fitness_cpu() {
        let calculator = GPUMemoryFitnessCalculator::new("cpu");
        let pad_state = create_test_pad_state();
        let memories = vec![(pad_state.clone(), 1.0, 5, 0.6, 0.7, 0.2)];
        let weights = DEFAULT_FITNESS_WEIGHTS;
        let config = TemporalDecayConfig::default();

        let scores = calculator.batch_fitness(&memories, &weights, &config);
        assert_eq!(scores.len(), 1);
        assert!(scores[0] >= 0.0 && scores[0] <= 1.0);
    }

    #[test]
    fn test_batch_fitness_multiple() {
        let calculator = GPUMemoryFitnessCalculator::new("cpu");
        let pad_state = create_test_pad_state();
        let memories = vec![
            (pad_state.clone(), 1.0, 5, 0.6, 0.7, 0.2),
            (pad_state.clone(), 2.0, 10, 0.8, 0.9, 0.5),
        ];
        let weights = DEFAULT_FITNESS_WEIGHTS;
        let config = TemporalDecayConfig::default();

        let scores = calculator.batch_fitness(&memories, &weights, &config);
        assert_eq!(scores.len(), 2);
    }
}
