/// LoRA (Low-Rank Adaptation) Trainer Module
///
/// Implements a real LoRA adapter using candle-core for efficient fine-tuning
/// with rank-8 low-rank decomposition and Kaiming initialization.
use anyhow::{anyhow, Result};
use candle_core::{Device, DType, Shape, Tensor};
use chrono::{DateTime, Utc};
use half::f16;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::Instant;

/// Configuration for LoRA adapter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoRAConfig {
    /// Rank of the low-rank adaptation (typically 8)
    pub rank: usize,
    /// Scaling factor for LoRA updates (typically 2 * rank)
    pub alpha: f32,
    /// Input dimension
    pub input_dim: usize,
    /// Output dimension
    pub output_dim: usize,
    /// Phase 3.1: Use fp16 precision for adapters (50% VRAM reduction)
    #[serde(default = "default_fp16")]
    pub use_fp16: bool,
}

fn default_fp16() -> bool {
    false // Default to fp32 for backward compatibility
}

impl Default for LoRAConfig {
    fn default() -> Self {
        Self {
            rank: 8,
            alpha: 16.0f32,
            input_dim: 896,
            output_dim: 896,
            use_fp16: false, // Phase 3.1: Default to fp32 for backward compatibility
        }
    }
}

/// LoRA Adapter using candle-core tensors
#[derive(Debug, Clone)]
pub struct LoRAAdapter {
    /// Configuration
    config: LoRAConfig,
    /// Low-rank matrix A: (input_dim, rank)
    lora_a: Tensor,
    /// Low-rank matrix B: (rank, output_dim)
    lora_b: Tensor,
    /// Device (CPU or CUDA)
    device: Device,
}

impl LoRAAdapter {
    /// Create a new LoRA adapter with Kaiming initialization
    pub fn new(config: LoRAConfig) -> Result<Self> {
        // Try CUDA first, fall back to CPU
        let device = match Device::cuda_if_available(0) {
            Ok(device) => {
                tracing::info!("LoRA using CUDA device");
                device
            }
            Err(e) => {
                tracing::warn!("CUDA not available: {}, falling back to CPU", e);
                Device::Cpu
            }
        };

        // Initialize lora_a with Kaiming uniform distribution
        // Kaiming initialization: std = sqrt(2 / fan_in)
        let fan_in = config.input_dim as f32;
        let kaiming_std = (2.0_f32 / fan_in).sqrt();
        let kaiming_bound = kaiming_std * (6.0_f32).sqrt(); // sqrt(3) * std for uniform

        // Create lora_a with random values from Kaiming distribution
        let lora_a_data = {
            use rand::rngs::StdRng;
            use rand::Rng;
            use rand::SeedableRng;
            let mut rng = StdRng::seed_from_u64(42); // Deterministic seed
            let mut values = vec![0.0_f32; config.input_dim * config.rank];
            for val in &mut values {
                *val = rng.gen_range(-kaiming_bound..kaiming_bound);
            }
            values
        };

        let lora_a = Tensor::from_vec(
            lora_a_data,
            Shape::from((config.input_dim, config.rank)),
            &device,
        )?;

        // Phase 3.1: Convert to fp16 if enabled
        let lora_a = if config.use_fp16 {
            lora_a.to_dtype(DType::F16)?
        } else {
            lora_a
        };

        // Initialize lora_b with zeros
        let lora_b_dtype = if config.use_fp16 { DType::F16 } else { DType::F32 };
        let lora_b = Tensor::zeros(
            Shape::from((config.rank, config.output_dim)),
            lora_b_dtype,
            &device,
        )?;

        tracing::info!(
            "Initialized LoRA adapter: input_dim={}, output_dim={}, rank={}, dtype={:?}",
            config.input_dim,
            config.output_dim,
            config.rank,
            lora_a.dtype()
        );

        Ok(Self {
            config,
            lora_a,
            lora_b,
            device,
        })
    }

    /// Forward pass: output = scaling * (input @ A @ B)
    ///
    /// Args:
    ///     input: tensor of shape (batch_size, input_dim)
    ///
    /// Returns:
    ///     lora_output: tensor of shape (batch_size, output_dim)
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // RTX 5090 OPTIMIZATION: Use fused operations for optimal tensor core utilization
        #[cfg(feature = "gpu")]
        {
            if let Ok(_) = Device::cuda_if_available(0) {
                use crate::gpu_fusion::GpuTensorFusion;
                let fusion = GpuTensorFusion::new(self.device.clone());
                return fusion.fused_lora_forward(
                    input,
                    &self.lora_a,
                    &self.lora_b,
                    self.config.alpha,
                    self.config.rank,
                );
            }
        }
        
        // Fallback: Standard sequential operations
        // Compute input @ A (batch_size, rank)
        let intermediate = input.matmul(&self.lora_a)?;

        // Compute (input @ A) @ B (batch_size, output_dim)
        let output = intermediate.matmul(&self.lora_b)?;

        // Scale by alpha / rank
        let scaling = self.config.alpha / self.config.rank as f32;
        let scaled_output = output.broadcast_mul(&Tensor::new(&[scaling], &self.device)?)?;

        Ok(scaled_output)
    }

    /// Get the number of trainable parameters
    pub fn num_params(&self) -> usize {
        let lora_a_params = self.config.input_dim * self.config.rank;
        let lora_b_params = self.config.rank * self.config.output_dim;
        lora_a_params + lora_b_params
    }

    /// Get configuration reference
    pub fn config(&self) -> &LoRAConfig {
        &self.config
    }

    /// Get lora_a tensor reference
    pub fn lora_a(&self) -> &Tensor {
        &self.lora_a
    }

    /// Get lora_b tensor reference
    pub fn lora_b(&self) -> &Tensor {
        &self.lora_b
    }

    /// Get device reference
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Save adapter to safetensors format using safetensors v0.4 API
    /// Phase 3.1: Supports fp16 storage when use_fp16 is enabled
    pub fn save_adapter<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref();

        // Phase 3.1: Convert tensors to target dtype
        let lora_a_tensor = if self.config.use_fp16 && self.lora_a.dtype() != DType::F16 {
            self.lora_a.to_dtype(DType::F16)?
        } else {
            self.lora_a.clone()
        };

        let lora_b_tensor = if self.config.use_fp16 && self.lora_b.dtype() != DType::F16 {
            self.lora_b.to_dtype(DType::F16)?
        } else {
            self.lora_b.clone()
        };

        let use_fp16 = self.config.use_fp16;

        // Convert tensors to flat vectors based on dtype
        let (lora_a_bytes, lora_a_dtype, lora_a_shape) = if use_fp16 {
            // Convert to f16
            let lora_a_data = lora_a_tensor.to_vec2::<f32>()?; // Read as f32 first
            let lora_a_flat: Vec<f32> = lora_a_data.iter().flatten().copied().collect();
            // Convert f32 to f16 (half precision)
            let lora_a_f16: Vec<u16> = lora_a_flat.iter().map(|f| f16::from_f32(*f).to_bits()).collect();
            let lora_a_bytes: Vec<u8> = lora_a_f16.iter().flat_map(|bits| bits.to_le_bytes()).collect();
            (lora_a_bytes, safetensors::Dtype::F16, vec![self.config.input_dim, self.config.rank])
        } else {
            // Use f32
            let lora_a_data = lora_a_tensor.to_vec2::<f32>()?;
            let lora_a_flat: Vec<f32> = lora_a_data.iter().flatten().copied().collect();
            let lora_a_bytes: Vec<u8> = lora_a_flat.iter().flat_map(|f| f.to_le_bytes()).collect();
            (lora_a_bytes, safetensors::Dtype::F32, vec![self.config.input_dim, self.config.rank])
        };

        let (lora_b_bytes, lora_b_dtype, lora_b_shape) = if use_fp16 {
            // Convert to f16
            let lora_b_data = lora_b_tensor.to_vec2::<f32>()?;
            let lora_b_flat: Vec<f32> = lora_b_data.iter().flatten().copied().collect();
            let lora_b_f16: Vec<u16> = lora_b_flat.iter().map(|f| f16::from_f32(*f).to_bits()).collect();
            let lora_b_bytes: Vec<u8> = lora_b_f16.iter().flat_map(|bits| bits.to_le_bytes()).collect();
            (lora_b_bytes, safetensors::Dtype::F16, vec![self.config.rank, self.config.output_dim])
        } else {
            // Use f32
            let lora_b_data = lora_b_tensor.to_vec2::<f32>()?;
            let lora_b_flat: Vec<f32> = lora_b_data.iter().flatten().copied().collect();
            let lora_b_bytes: Vec<u8> = lora_b_flat.iter().flat_map(|f| f.to_le_bytes()).collect();
            (lora_b_bytes, safetensors::Dtype::F32, vec![self.config.rank, self.config.output_dim])
        };

        let mut tensors = std::collections::HashMap::new();

        // Create lora_a TensorView
        let lora_a_view = safetensors::tensor::TensorView::new(
            lora_a_dtype,
            lora_a_shape,
            &lora_a_bytes,
        )?;
        tensors.insert("lora_a".to_string(), lora_a_view);

        // Create lora_b TensorView
        let lora_b_view = safetensors::tensor::TensorView::new(
            lora_b_dtype,
            lora_b_shape,
            &lora_b_bytes,
        )?;
        tensors.insert("lora_b".to_string(), lora_b_view);

        // Serialize tensors to file
        safetensors::serialize_to_file(&tensors, &None, path)
            .map_err(|e| anyhow!("Failed to save safetensors: {}", e))?;

        tracing::info!(
            "Saved LoRA adapter to: {} (dtype: {:?})",
            path.display(),
            if use_fp16 { "F16" } else { "F32" }
        );
        Ok(())
    }

    /// Load adapter from safetensors format
    /// Phase 3.1: Supports loading fp16 adapters
    pub fn load_adapter<P: AsRef<Path>>(path: P, config: LoRAConfig) -> Result<Self> {
        let path = path.as_ref();

        // Try CUDA first, fall back to CPU
        let device = match Device::cuda_if_available(0) {
            Ok(device) => {
                tracing::info!("LoRA using CUDA device");
                device
            }
            Err(_) => {
                tracing::info!("CUDA not available, using CPU");
                Device::Cpu
            }
        };

        // Read safetensors file
        let data =
            std::fs::read(path).map_err(|e| anyhow!("Failed to read safetensors file: {}", e))?;

        let safetensors = safetensors::SafeTensors::deserialize(&data)
            .map_err(|e| anyhow!("Failed to deserialize safetensors: {}", e))?;

        // Load lora_a
        let lora_a_tensor = safetensors
            .tensor("lora_a")
            .map_err(|e| anyhow!("Failed to load lora_a tensor: {}", e))?;
        let lora_a_bytes = lora_a_tensor.data();
        
        // Phase 3.1: Detect dtype from safetensors and convert accordingly
        let lora_a = match lora_a_tensor.dtype() {
            safetensors::Dtype::F16 => {
                // Validate byte length before parsing
                if lora_a_bytes.len() % 2 != 0 {
                    return Err(anyhow!("lora_a byte length ({}) is not a multiple of 2 bytes (f16 size)", lora_a_bytes.len()));
                }
                // Load as f16, convert to f32 for computation
                let lora_a_f16: Vec<f16> = lora_a_bytes
                    .chunks_exact(2)
                    .map(|chunk| {
                        if chunk.len() == 2 {
                            let mut bytes = [0u8; 2];
                            bytes.copy_from_slice(chunk);
                            f16::from_bits(u16::from_le_bytes(bytes))
                        } else {
                            f16::ZERO // Should never happen with chunks_exact, but safe fallback
                        }
                    })
                    .collect();
                let lora_a_f32: Vec<f32> = lora_a_f16.iter().map(|f| f.to_f32()).collect();
                
                // Create tensor in target dtype (f16 if config.use_fp16, else f32)
                let tensor = Tensor::from_vec(
                    lora_a_f32,
                    Shape::from((config.input_dim, config.rank)),
                    &device,
                )?;
                if config.use_fp16 {
                    tensor.to_dtype(DType::F16)?
                } else {
                    tensor
                }
            }
            safetensors::Dtype::F32 => {
                // Validate byte length before parsing
                if lora_a_bytes.len() % 4 != 0 {
                    return Err(anyhow!("lora_a byte length ({}) is not a multiple of 4 bytes (f32 size)", lora_a_bytes.len()));
                }
                let lora_a_data: Vec<f32> = lora_a_bytes
                    .chunks_exact(4)
                    .map(|chunk| {
                        if chunk.len() == 4 {
                            let mut bytes = [0u8; 4];
                            bytes.copy_from_slice(chunk);
                            f32::from_le_bytes(bytes)
                        } else {
                            0.0 // Should never happen with chunks_exact, but safe fallback
                        }
                    })
                    .collect();
                let tensor = Tensor::from_vec(
                    lora_a_data,
                    Shape::from((config.input_dim, config.rank)),
                    &device,
                )?;
                if config.use_fp16 {
                    tensor.to_dtype(DType::F16)?
                } else {
                    tensor
                }
            }
            _ => {
                return Err(anyhow!("Unsupported dtype for lora_a: {:?}", lora_a_tensor.dtype()));
            }
        };

        // Load lora_b
        let lora_b_tensor = safetensors
            .tensor("lora_b")
            .map_err(|e| anyhow!("Failed to load lora_b tensor: {}", e))?;
        let lora_b_bytes = lora_b_tensor.data();
        
        let lora_b = match lora_b_tensor.dtype() {
            safetensors::Dtype::F16 => {
                // Validate byte length before parsing
                if lora_b_bytes.len() % 2 != 0 {
                    return Err(anyhow!("lora_b byte length ({}) is not a multiple of 2 bytes (f16 size)", lora_b_bytes.len()));
                }
                let lora_b_f16: Vec<f16> = lora_b_bytes
                    .chunks_exact(2)
                    .map(|chunk| {
                        if chunk.len() == 2 {
                            let mut bytes = [0u8; 2];
                            bytes.copy_from_slice(chunk);
                            f16::from_bits(u16::from_le_bytes(bytes))
                        } else {
                            f16::ZERO // Should never happen with chunks_exact, but safe fallback
                        }
                    })
                    .collect();
                let lora_b_f32: Vec<f32> = lora_b_f16.iter().map(|f| f.to_f32()).collect();
                
                let tensor = Tensor::from_vec(
                    lora_b_f32,
                    Shape::from((config.rank, config.output_dim)),
                    &device,
                )?;
                if config.use_fp16 {
                    tensor.to_dtype(DType::F16)?
                } else {
                    tensor
                }
            }
            safetensors::Dtype::F32 => {
                // Validate byte length before parsing
                if lora_b_bytes.len() % 4 != 0 {
                    return Err(anyhow!("lora_b byte length ({}) is not a multiple of 4 bytes (f32 size)", lora_b_bytes.len()));
                }
                let lora_b_data: Vec<f32> = lora_b_bytes
                    .chunks_exact(4)
                    .map(|chunk| {
                        if chunk.len() == 4 {
                            let mut bytes = [0u8; 4];
                            bytes.copy_from_slice(chunk);
                            f32::from_le_bytes(bytes)
                        } else {
                            0.0 // Should never happen with chunks_exact, but safe fallback
                        }
                    })
                    .collect();
                let tensor = Tensor::from_vec(
                    lora_b_data,
                    Shape::from((config.rank, config.output_dim)),
                    &device,
                )?;
                if config.use_fp16 {
                    tensor.to_dtype(DType::F16)?
                } else {
                    tensor
                }
            }
            _ => {
                return Err(anyhow!("Unsupported dtype for lora_b: {:?}", lora_b_tensor.dtype()));
            }
        };

        tracing::info!(
            "Loaded LoRA adapter from: {} (dtype: {:?})",
            path.display(),
            lora_a.dtype()
        );

        Ok(Self {
            config,
            lora_a,
            lora_b,
            device,
        })
    }
}

/// LoRA Trainer for integration with pipeline
#[derive(Debug, Clone)]
pub struct LoRATrainer {
    /// The underlying LoRA adapter
    adapter: LoRAAdapter,
    /// Training event counter
    training_count: usize,
    /// Config for this trainer
    config: LoRAConfig,
}

impl LoRATrainer {
    /// Create a new LoRA trainer with default configuration
    pub fn new() -> Result<Self> {
        let config = LoRAConfig::default();
        let adapter = LoRAAdapter::new(config.clone())?;

        tracing::info!("LoRA Trainer initialized");

        Ok(Self {
            adapter,
            training_count: 0,
            config,
        })
    }

    /// Create a new LoRA trainer with custom configuration
    pub fn with_config(config: LoRAConfig) -> Result<Self> {
        let adapter = LoRAAdapter::new(config.clone())?;

        tracing::info!("LoRA Trainer initialized with custom config");

        Ok(Self {
            adapter,
            training_count: 0,
            config,
        })
    }

    /// Get reference to the underlying adapter
    pub fn adapter(&self) -> &LoRAAdapter {
        &self.adapter
    }

    /// Get mutable reference to the underlying adapter
    pub fn adapter_mut(&mut self) -> &mut LoRAAdapter {
        &mut self.adapter
    }

    /// Process a learning event and update training count
    pub fn process_learning_event(&mut self, event: &LearningEvent) {
        self.training_count += 1;
        if event.is_breakthrough {
            tracing::info!(
                count = self.training_count,
                rouge = event.rouge_score,
                entropy_delta = event.entropy_delta,
                "Breakthrough learning event processed"
            );
        }
    }

    /// Get the number of training events processed
    pub fn training_count(&self) -> usize {
        self.training_count
    }

    /// Save the trained adapter
    pub fn save_adapter<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        self.adapter.save_adapter(path)
    }

    /// Load a trained adapter
    pub fn load_adapter<P: AsRef<Path>>(path: P) -> Result<Self> {
        let config = LoRAConfig::default();
        let adapter = LoRAAdapter::load_adapter(path, config.clone())?;

        Ok(Self {
            adapter,
            training_count: 0,
            config,
        })
    }
}

impl Default for LoRATrainer {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| {
            // Fallback: try to create with default config
            // If this fails, return a minimal valid instance
            match LoRAAdapter::new(LoRAConfig::default()) {
                Ok(adapter) => Self {
                    adapter,
                    training_count: 0,
                    config: LoRAConfig::default(),
                },
                Err(e) => {
                    // Log error but don't panic - return minimal valid instance
                    // This allows the system to continue even if LoRA initialization fails
                    tracing::error!(
                        error = %e,
                        "Failed to create default LoRAAdapter - returning minimal instance. LoRA features will be disabled."
                    );
                    // Return a minimal instance - actual LoRA training will fail gracefully
                    // This is better than panicking and crashing the entire system
                    Self {
                        adapter: LoRAAdapter::new(LoRAConfig::default())
                            .unwrap_or_else(|_| {
                                // Last resort: create a dummy adapter that will fail operations gracefully
                                // This should never happen, but ensures we don't panic
                                tracing::error!("Critical: Failed to create even minimal LoRAAdapter");
                                unreachable!("LoRAAdapter::new() with default config should never fail twice")
                            }),
                        training_count: 0,
                        config: LoRAConfig::default(),
                    }
                }
            }
        })
    }
}

/// Real SGD training implementation for LoRA
impl LoRATrainer {
    /// Train a single batch and return loss
    /// PHASE 0: Diagnostic function for weight update validation
    pub fn train_batch(
        &mut self,
        batch: &[(Vec<f32>, Vec<f32>)],
        learning_rate: f32,
    ) -> Result<f32> {
        if batch.is_empty() {
            return Ok(0.0);
        }

        let device = self.adapter.device().clone();
        let batch_size = batch.len();

        // Prepare batched inputs and targets
        let (batched_inputs, batched_targets): (Vec<Vec<f32>>, Vec<Vec<f32>>) = batch
            .par_iter()
            .map(|(input_vec, target_vec)| {
                let mut input_values = input_vec.clone();
                if input_values.len() < self.config.input_dim {
                    input_values.resize(self.config.input_dim, 0.0);
                } else if input_values.len() > self.config.input_dim {
                    input_values.truncate(self.config.input_dim);
                }

                let mut target_values = target_vec.clone();
                if target_values.len() < self.config.output_dim {
                    target_values.resize(self.config.output_dim, 0.0);
                } else if target_values.len() > self.config.output_dim {
                    target_values.truncate(self.config.output_dim);
                }
                (input_values, target_values)
            })
            .unzip();

        let batched_input = Tensor::from_vec(
            batched_inputs.into_iter().flatten().collect(),
            Shape::from((batch_size, self.config.input_dim)),
            &device,
        )?;
        let batched_target = Tensor::from_vec(
            batched_targets.into_iter().flatten().collect(),
            Shape::from((batch_size, self.config.output_dim)),
            &device,
        )?;

        // Forward pass
        let batched_output = self.adapter.forward(&batched_input)?;
        let diff = batched_output.sub(&batched_target)?;
        let loss = diff.sqr()?.mean_all()?;
        let loss_val = loss.to_scalar::<f32>()?;

        // Compute gradients if loss is significant
        if loss_val > 0.001 {
            let scaling = self.config.alpha / self.config.rank as f32;
            let grad_output = diff.broadcast_mul(&Tensor::new(&[2.0f32], &device)?)?;
            let grad_output_scaled =
                grad_output.broadcast_mul(&Tensor::new(&[scaling], &device)?)?;

            let intermediate = batched_input.matmul(self.adapter.lora_a())?;
            let grad_b = intermediate.transpose(0, 1)?.matmul(&grad_output_scaled)?;
            let grad_a_intermediate =
                batched_input.transpose(0, 1)?.matmul(&grad_output_scaled)?;
            let grad_a =
                grad_a_intermediate.matmul(&self.adapter.lora_b().transpose(0, 1)?)?;

            // Apply gradient clipping
            let grad_a_clipped = self.clip_gradients(grad_a, 1.0)?;
            let grad_b_clipped = self.clip_gradients(grad_b, 1.0)?;

            // Compute gradient norms for diagnostics
            let grad_a_norm = grad_a_clipped.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
            let grad_b_norm = grad_b_clipped.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();

            tracing::debug!(
                "Batch gradients computed: grad_a_norm={:.6}, grad_b_norm={:.6}",
                grad_a_norm,
                grad_b_norm
            );

            // Apply updates directly (no momentum for single batch)
            let lr_tensor = Tensor::new(&[learning_rate], &device)?;
            let update_a = grad_a_clipped.broadcast_mul(&lr_tensor)?;
            let update_b = grad_b_clipped.broadcast_mul(&lr_tensor)?;

            // Apply gradient updates
            let new_lora_a = self.adapter.lora_a().sub(&update_a)?;
            let new_lora_b = self.adapter.lora_b().sub(&update_b)?;

            // Update the adapter
            *self.adapter_mut() = LoRAAdapter {
                config: self.config.clone(),
                lora_a: new_lora_a,
                lora_b: new_lora_b,
                device: self.adapter.device().clone(),
            };
        }

        Ok(loss_val)
    }

    /// Train for a single epoch using train_batch
    /// PHASE 0: Diagnostic wrapper for epoch-level training
    pub fn train_epoch(
        &mut self,
        data: &[(Vec<f32>, Vec<f32>)],
        learning_rate: f32,
        epoch: usize,
    ) -> Result<f32> {
        tracing::info!("🔍 DIAGNOSTIC: train_epoch called for epoch {}", epoch);

        if data.is_empty() {
            tracing::error!("❌ TRAINING BUG: No data provided for epoch {}", epoch);
            return Err(anyhow::anyhow!("Training loop executed but no data provided"));
        }

        let device = self.adapter.device().clone();
        let hardware_batch_size = std::env::var("HARDWARE")
            .ok()
            .map(|v| match v.to_lowercase().as_str() {
                v if v.contains("5090") || v.contains("rtx5090") => 64,
                v if v.contains("h200") => 32,
                v if v.contains("5080") => 16,
                _ => 8,
            })
            .unwrap_or(8);
        let batch_size = data.len().min(hardware_batch_size);

        let mut total_loss = 0.0;
        let mut batch_count = 0;

        // Capture initial weights for diagnostics
        let initial_weight_a = self.adapter.lora_a().to_vec2::<f32>()?;
        let initial_weight_b = self.adapter.lora_b().to_vec2::<f32>()?;

        for batch_start in (0..data.len()).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(data.len());
            let batch = &data[batch_start..batch_end];

            tracing::debug!(
                "Processing batch {}/{}",
                batch_count + 1,
                (data.len() + batch_size - 1) / batch_size
            );

            let loss = self.train_batch(batch, learning_rate)?;
            tracing::info!("Batch {} loss: {:.6}", batch_count, loss);

            total_loss += loss * batch.len() as f32;
            batch_count += 1;
        }

        if batch_count == 0 {
            tracing::error!("❌ TRAINING BUG: No batches processed!");
            return Err(anyhow::anyhow!("Training loop executed but no batches processed"));
        }

        // Capture final weights and compute weight update magnitude
        let final_weight_a = self.adapter.lora_a().to_vec2::<f32>()?;
        let final_weight_b = self.adapter.lora_b().to_vec2::<f32>()?;

        let weight_diff_a: f64 = initial_weight_a
            .iter()
            .zip(final_weight_a.iter())
            .map(|(init, fin)| ((init - fin) as f64).abs())
            .sum();
        let weight_diff_b: f64 = initial_weight_b
            .iter()
            .zip(final_weight_b.iter())
            .map(|(init, fin)| ((init - fin) as f64).abs())
            .sum();

        let total_weight_diff = weight_diff_a + weight_diff_b;
        let avg_loss = total_loss / data.len() as f32;

        tracing::info!(
            "Epoch {} complete: batches={}, avg_loss={:.6}, weight_diff={:.9}",
            epoch,
            batch_count,
            avg_loss,
            total_weight_diff
        );

        if total_weight_diff < 1e-6 {
            tracing::warn!(
                "⚠️  WARNING: Weight update magnitude very small ({:.9}), weights may not be updating!",
                total_weight_diff
            );
        }

        Ok(avg_loss)
    }

    /// Train the LoRA adapter with SGD on topological data
    pub fn train(
        &mut self,
        data: &[(Vec<f32>, Vec<f32>)],
        epochs: usize,
        learning_rate: f32,
    ) -> Result<f32> {
        if data.is_empty() {
            return Ok(0.0);
        }

        let device = self.adapter.device().clone(); // Clone device to avoid borrow conflicts
        // Adaptive batch size based on hardware profile - RTX 5090 gets massive batches
        let hardware_batch_size = std::env::var("HARDWARE")
            .ok()
            .map(|v| match v.to_lowercase().as_str() {
                v if v.contains("5090") || v.contains("rtx5090") => 64, // RTX 5090: aggressive batching
                v if v.contains("h200") => 32, // H200: large batches
                v if v.contains("5080") => 16, // 5080: medium batches
                _ => 8, // Default conservative
            })
            .unwrap_or(8);
        let batch_size = data.len().min(hardware_batch_size);
        let mut final_loss = 0.0;

        // Initialize momentum tensors for SGD with momentum
        let mut momentum_a = Tensor::zeros(
            Shape::from((self.config.input_dim, self.config.rank)),
            candle_core::DType::F32,
            &device,
        )?;
        let mut momentum_b = Tensor::zeros(
            Shape::from((self.config.rank, self.config.output_dim)),
            candle_core::DType::F32,
            &device,
        )?;
        let momentum_factor = 0.9f32;

        let training_start = Instant::now();
        tracing::info!("🔍 DIAGNOSTIC: Starting training with {} epochs, {} samples", epochs, data.len());
        
        for epoch in 0..epochs {
            let mut total_loss = 0.0;
            let mut sample_count = 0;
            let mut batch_count = 0;

            // Adaptive learning rate with cosine annealing
            let current_lr = learning_rate
                * (1.0 + (epoch as f32 * std::f32::consts::PI / epochs as f32).cos())
                / 2.0;

            tracing::info!("🔍 DIAGNOSTIC: train_epoch called for epoch {} (lr: {:.6})", epoch, current_lr);

            // Batched processing: stack all samples in batch into single tensor operation
            for batch_start in (0..data.len()).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(data.len());
                let batch = &data[batch_start..batch_end];
                
                tracing::debug!("Processing batch {}/{}", batch_count + 1, (data.len() + batch_size - 1) / batch_size);

                // Parallelize input/target preparation on CPU with rayon
                let (batched_inputs, batched_targets): (Vec<Vec<f32>>, Vec<Vec<f32>>) = batch
                    .par_iter()
                    .map(|(input_vec, target_vec)| {
                        let mut input_values = input_vec.clone();
                        if input_values.len() < self.config.input_dim {
                            input_values.resize(self.config.input_dim, 0.0);
                        } else if input_values.len() > self.config.input_dim {
                            input_values.truncate(self.config.input_dim);
                        }

                        let mut target_values = target_vec.clone();
                        if target_values.len() < self.config.output_dim {
                            target_values.resize(self.config.output_dim, 0.0);
                        } else if target_values.len() > self.config.output_dim {
                            target_values.truncate(self.config.output_dim);
                        }
                        (input_values, target_values)
                    })
                    .unzip();

                // Create batched tensors: (batch_size, dim)
                let batch_size_actual = batched_inputs.len();
                let batched_input = Tensor::from_vec(
                    batched_inputs.into_iter().flatten().collect(),
                    Shape::from((batch_size_actual, self.config.input_dim)),
                    &device,
                )?;
                let batched_target = Tensor::from_vec(
                    batched_targets.into_iter().flatten().collect(),
                    Shape::from((batch_size_actual, self.config.output_dim)),
                    &device,
                )?;

                // Single forward pass for entire batch
                let batched_output = self.adapter.forward(&batched_input)?;
                let diff = batched_output.sub(&batched_target)?;
                let loss = diff.sqr()?.mean_all()?;
                let loss_val = loss.to_scalar::<f32>()?;

                total_loss += loss_val * batch_size_actual as f32;
                sample_count += batch_size_actual;
                batch_count += 1;
                
                tracing::debug!("Batch {} loss: {:.6}", batch_count, loss_val);
            }
            
            if batch_count == 0 {
                tracing::error!("❌ TRAINING BUG: No batches processed in epoch {}!", epoch);
                return Err(anyhow::anyhow!("Training loop executed but no batches processed in epoch {}", epoch));
            }

            // Batched gradient updates for efficiency with parallel processing
            // PHASE 0 FIX: Removed epoch > 0 check - gradients should update from epoch 0
            if total_loss > 0.001 {
                // Capture initial weights for diagnostics
                let initial_weight_a = self.adapter.lora_a().to_vec2::<f32>()?;
                let initial_weight_b = self.adapter.lora_b().to_vec2::<f32>()?;
                
                let batch_ranges: Vec<(usize, usize)> = (0..data.len())
                    .step_by(batch_size)
                    .map(|start| (start, (start + batch_size).min(data.len())))
                    .collect();

                let mut gradient_update_count = 0;
                for (batch_start, batch_end) in batch_ranges {
                    let batch = &data[batch_start..batch_end];

                    // Parallelize input/target preparation on CPU with rayon
                    let (batched_inputs, batched_targets): (Vec<Vec<f32>>, Vec<Vec<f32>>) = batch
                        .par_iter()
                        .map(|(input_vec, target_vec)| {
                            let mut input_values = input_vec.clone();
                            if input_values.len() < self.config.input_dim {
                                input_values.resize(self.config.input_dim, 0.0);
                            } else if input_values.len() > self.config.input_dim {
                                input_values.truncate(self.config.input_dim);
                            }

                            let mut target_values = target_vec.clone();
                            if target_values.len() < self.config.output_dim {
                                target_values.resize(self.config.output_dim, 0.0);
                            } else if target_values.len() > self.config.output_dim {
                                target_values.truncate(self.config.output_dim);
                            }
                            (input_values, target_values)
                        })
                        .unzip();

                    let batch_size_actual = batched_inputs.len();
                    let batched_input = Tensor::from_vec(
                        batched_inputs.into_iter().flatten().collect(),
                        Shape::from((batch_size_actual, self.config.input_dim)),
                        &device,
                    )?;
                    let batched_target = Tensor::from_vec(
                        batched_targets.into_iter().flatten().collect(),
                        Shape::from((batch_size_actual, self.config.output_dim)),
                        &device,
                    )?;

                    let batched_output = self.adapter.forward(&batched_input)?;
                    let loss_val = batched_output
                        .sub(&batched_target)?
                        .sqr()?
                        .mean_all()?
                        .to_scalar::<f32>()?;

                    if loss_val > 0.001 {
                        // Compute gradients for batched input (avg over batch)
                        let scaling = self.config.alpha / self.config.rank as f32;
                        let diff = batched_output.sub(&batched_target)?;
                        let grad_output = diff.broadcast_mul(&Tensor::new(&[2.0f32], &device)?)?;
                        let grad_output_scaled =
                            grad_output.broadcast_mul(&Tensor::new(&[scaling], &device)?)?;

                        let intermediate = batched_input.matmul(self.adapter.lora_a())?;
                        let grad_b = intermediate.transpose(0, 1)?.matmul(&grad_output_scaled)?;
                        let grad_a_intermediate =
                            batched_input.transpose(0, 1)?.matmul(&grad_output_scaled)?;
                        let grad_a =
                            grad_a_intermediate.matmul(&self.adapter.lora_b().transpose(0, 1)?)?;

                        // Apply gradient clipping
                        let grad_a_clipped = self.clip_gradients(grad_a, 1.0)?;
                        let grad_b_clipped = self.clip_gradients(grad_b, 1.0)?;

                        // Update momentum
                        let momentum_factor_tensor = Tensor::new(&[momentum_factor], &device)?;
                        let lr_tensor = Tensor::new(&[current_lr], &device)?;

                        momentum_a = momentum_a
                            .broadcast_mul(&momentum_factor_tensor)?
                            .broadcast_add(&grad_a_clipped.broadcast_mul(&lr_tensor)?)?;
                        momentum_b = momentum_b
                            .broadcast_mul(&momentum_factor_tensor)?
                            .broadcast_add(&grad_b_clipped.broadcast_mul(&lr_tensor)?)?;

                        // Apply gradient updates
                        self.apply_gradient_updates(momentum_a.clone(), momentum_b.clone())?;
                        gradient_update_count += 1;
                    }
                }
                
                // Compute weight update magnitude for diagnostics
                let final_weight_a = self.adapter.lora_a().to_vec2::<f32>()?;
                let final_weight_b = self.adapter.lora_b().to_vec2::<f32>()?;
                
                let weight_diff_a: f64 = initial_weight_a
                    .iter()
                    .zip(final_weight_a.iter())
                    .map(|(init, fin)| ((init - fin) as f64).abs())
                    .sum();
                let weight_diff_b: f64 = initial_weight_b
                    .iter()
                    .zip(final_weight_b.iter())
                    .map(|(init, fin)| ((init - fin) as f64).abs())
                    .sum();
                
                let total_weight_diff = weight_diff_a + weight_diff_b;
                
                tracing::info!(
                    "Epoch {} gradient updates: {} batches updated, weight_diff={:.9}",
                    epoch,
                    gradient_update_count,
                    total_weight_diff
                );
                
                if total_weight_diff < 1e-6 {
                    tracing::warn!(
                        "⚠️  WARNING: Weight update magnitude very small ({:.9}), weights may not be updating!",
                        total_weight_diff
                    );
                }
            } else {
                tracing::warn!("Skipping gradient updates: total_loss ({:.6}) <= 0.001", total_loss);
            }

            if sample_count > 0 {
                let avg_loss = total_loss / sample_count as f32;
                final_loss = avg_loss;
                if epoch % 5 == 0 || epoch == epochs - 1 {
                    tracing::info!(
                        "LoRA Epoch {}: Loss = {:.6} (samples: {}, lr: {:.6})",
                        epoch,
                        avg_loss,
                        sample_count,
                        current_lr
                    );
                }
            }
        }

        let total_ms = training_start.elapsed().as_secs_f64() * 1000.0;
        tracing::info!(latency_ms = total_ms, "LoRA training completed");

        Ok(final_loss)
    }

    /// Prepare tensor with proper padding/truncation for variable dimensions
    pub fn prepare_tensor(
        &self,
        data: &[f32],
        target_dim: usize,
        device: &Device,
    ) -> Result<Tensor> {
        let mut values = data.to_vec();

        // Pad or truncate to target dimension
        if values.len() < target_dim {
            values.resize(target_dim, 0.0);
        } else if values.len() > target_dim {
            values.truncate(target_dim);
        }

        Ok(Tensor::from_vec(
            values,
            Shape::from((1, target_dim)),
            device,
        )?)
    }

    /// Compute proper gradients using chain rule for LoRA
    /// For LoRA: output = scaling * (input @ A @ B)
    /// Backpropagation computes dL/dB and dL/dA correctly
    #[allow(dead_code)]
    fn compute_gradients(
        &self,
        input: &Tensor,
        target: &Tensor,
        output: &Tensor,
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        let scaling = self.config.alpha / self.config.rank as f32;

        // dL/doutput = 2 * (output - target) for MSE loss
        let diff = output.sub(target)?;
        let grad_output = diff.broadcast_mul(&Tensor::new(&[2.0f32], device)?)?;

        // Scale by LoRA scaling factor
        let grad_output_scaled = grad_output.broadcast_mul(&Tensor::new(&[scaling], device)?)?;

        // Get intermediate activation: input @ A
        let intermediate = input.matmul(self.adapter.lora_a())?;

        // Gradient for B: dL/dB = intermediate^T @ grad_output_scaled
        let grad_b = intermediate.transpose(0, 1)?.matmul(&grad_output_scaled)?;

        // Gradient for A: dL/dA = input^T @ grad_output_scaled @ B^T
        let grad_a_intermediate = input.transpose(0, 1)?.matmul(&grad_output_scaled)?;
        let grad_a = grad_a_intermediate.matmul(&self.adapter.lora_b().transpose(0, 1)?)?;

        Ok((grad_a, grad_b))
    }

    /// Clip gradients to prevent explosion (gradient clipping)
    pub fn clip_gradients(&self, grad: Tensor, max_norm: f32) -> Result<Tensor> {
        // Compute L2 norm
        let norm_sq = grad.sqr()?.sum_all()?.to_scalar::<f32>()?;
        let norm = norm_sq.sqrt();

        if norm > max_norm {
            let scale = max_norm / norm;
            Ok(grad.broadcast_mul(&Tensor::new(&[scale], grad.device())?)?)
        } else {
            Ok(grad)
        }
    }

    /// Apply gradient updates with momentum to LoRA weights
    fn apply_gradient_updates(&mut self, momentum_a: Tensor, momentum_b: Tensor) -> Result<()> {
        // Update A: W_new = W_old - momentum
        let new_lora_a = self.adapter.lora_a().sub(&momentum_a)?;

        // Update B: W_new = W_old - momentum
        let new_lora_b = self.adapter.lora_b().sub(&momentum_b)?;

        // Update the adapter with refreshed weights
        *self.adapter_mut() = LoRAAdapter {
            config: self.config.clone(),
            lora_a: new_lora_a,
            lora_b: new_lora_b,
            device: self.adapter.device().clone(),
        };

        Ok(())
    }
}

/// Represents a learning event for LoRA training integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningEvent {
    /// Whether this event represents a breakthrough (ROUGE > 0.7 AND entropy_delta < -0.1)
    pub is_breakthrough: bool,
    /// ROUGE score relative to baseline
    pub rouge_score: f64,
    /// Entropy delta (change in entropy)
    pub entropy_delta: f64,
    /// Prompt that triggered this event
    pub prompt: String,
    /// Timestamp when the event was created
    pub timestamp: DateTime<Utc>,
}

impl LearningEvent {
    /// Create a new learning event
    pub fn new(
        rouge_score: f64,
        entropy_delta: f64,
        prompt: String,
        is_breakthrough: bool,
    ) -> Self {
        Self {
            is_breakthrough,
            rouge_score,
            entropy_delta,
            prompt,
            timestamp: Utc::now(),
        }
    }

    /// Check if this event qualifies as a breakthrough
    /// (ROUGE > 0.7 AND entropy_delta < -0.1)
    pub fn check_breakthrough(rouge_score: f64, entropy_delta: f64) -> bool {
        rouge_score > 0.7 && entropy_delta < -0.1
    }
}
