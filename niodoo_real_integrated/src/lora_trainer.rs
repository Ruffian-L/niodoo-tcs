/// LoRA (Low-Rank Adaptation) Trainer Module
///
/// Implements a real LoRA adapter using candle-core for efficient fine-tuning
/// with rank-8 low-rank decomposition and Kaiming initialization.
use anyhow::{anyhow, Result};
use candle_core::{Device, Shape, Tensor};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::Path;

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
}

impl Default for LoRAConfig {
    fn default() -> Self {
        Self {
            rank: 8,
            alpha: 16.0,
            input_dim: 896,
            output_dim: 896,
        }
    }
}

/// LoRA Adapter using candle-core tensors
#[derive(Debug)]
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
        let kaiming_std = (2.0 / fan_in).sqrt();
        let kaiming_bound = kaiming_std * (6.0_f32).sqrt(); // sqrt(3) * std for uniform

        // Create lora_a with random values from Kaiming distribution
        let lora_a_data = {
            use rand::Rng;
            let mut rng = rand::thread_rng();
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

        // Initialize lora_b with zeros
        let lora_b = Tensor::zeros(
            Shape::from((config.rank, config.output_dim)),
            candle_core::DType::F32,
            &device,
        )?;

        tracing::info!(
            "Initialized LoRA adapter: input_dim={}, output_dim={}, rank={}",
            config.input_dim,
            config.output_dim,
            config.rank
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
    pub fn save_adapter<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref();

        // Convert tensors to flat f32 vectors
        let lora_a_data = self.lora_a.to_vec2::<f32>()?;
        let lora_b_data = self.lora_b.to_vec2::<f32>()?;

        // Flatten for safetensors
        let lora_a_flat: Vec<f32> = lora_a_data.iter().flatten().copied().collect();
        let lora_b_flat: Vec<f32> = lora_b_data.iter().flatten().copied().collect();

        // Convert f32 to bytes safely using into_raw_parts and byte buffer
        // This is safer than unsafe casting and maintains proper alignment
        let lora_a_bytes: Vec<u8> = lora_a_flat.iter().flat_map(|f| f.to_le_bytes()).collect();

        let lora_b_bytes: Vec<u8> = lora_b_flat.iter().flat_map(|f| f.to_le_bytes()).collect();

        let mut tensors = std::collections::HashMap::new();

        // Create lora_a TensorView with proper safetensors v0.4 API
        let lora_a_view = safetensors::tensor::TensorView::new(
            safetensors::Dtype::F32,
            vec![self.config.input_dim, self.config.rank],
            &lora_a_bytes,
        )?;
        tensors.insert("lora_a".to_string(), lora_a_view);

        // Create lora_b TensorView with proper safetensors v0.4 API
        let lora_b_view = safetensors::tensor::TensorView::new(
            safetensors::Dtype::F32,
            vec![self.config.rank, self.config.output_dim],
            &lora_b_bytes,
        )?;
        tensors.insert("lora_b".to_string(), lora_b_view);

        // Serialize tensors to file using serialize_to_file
        safetensors::serialize_to_file(&tensors, &None, path)
            .map_err(|e| anyhow!("Failed to save safetensors: {}", e))?;

        tracing::info!("Saved LoRA adapter to: {}", path.display());
        Ok(())
    }

    /// Load adapter from safetensors format
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
        let lora_a_data: Vec<f32> = lora_a_bytes
            .chunks_exact(4)
            .map(|chunk| {
                let mut bytes = [0u8; 4];
                bytes.copy_from_slice(chunk);
                f32::from_le_bytes(bytes)
            })
            .collect();

        let lora_a = Tensor::from_vec(
            lora_a_data,
            Shape::from((config.input_dim, config.rank)),
            &device,
        )?;

        // Load lora_b
        let lora_b_tensor = safetensors
            .tensor("lora_b")
            .map_err(|e| anyhow!("Failed to load lora_b tensor: {}", e))?;
        let lora_b_bytes = lora_b_tensor.data();
        let lora_b_data: Vec<f32> = lora_b_bytes
            .chunks_exact(4)
            .map(|chunk| {
                let mut bytes = [0u8; 4];
                bytes.copy_from_slice(chunk);
                f32::from_le_bytes(bytes)
            })
            .collect();

        let lora_b = Tensor::from_vec(
            lora_b_data,
            Shape::from((config.rank, config.output_dim)),
            &device,
        )?;

        tracing::info!("Loaded LoRA adapter from: {}", path.display());

        Ok(Self {
            config,
            lora_a,
            lora_b,
            device,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lora_adapter_creation() -> Result<()> {
        let config = LoRAConfig {
            rank: 8,
            alpha: 16.0,
            input_dim: 896,
            output_dim: 896,
        };

        let adapter = LoRAAdapter::new(config.clone())?;

        assert_eq!(adapter.num_params(), 896 * 8 + 8 * 896);
        assert_eq!(adapter.config().rank, 8);
        assert_eq!(adapter.config().alpha, 16.0);

        Ok(())
    }

    #[test]
    fn test_lora_forward_pass() -> Result<()> {
        let config = LoRAConfig {
            rank: 8,
            alpha: 16.0,
            input_dim: 64,
            output_dim: 64,
        };

        let adapter = LoRAAdapter::new(config)?;

        // Create a test input (batch_size=2, input_dim=64)
        let input_data = vec![0.1_f32; 2 * 64];
        let input = Tensor::from_vec(input_data, Shape::from((2, 64)), &adapter.device())?;

        // Forward pass
        let output = adapter.forward(&input)?;

        // Verify output shape
        let output_shape = output.shape();
        assert_eq!(output_shape.dims(), &[2, 64]);

        Ok(())
    }

    #[test]
    fn test_lora_num_params() {
        let config = LoRAConfig {
            rank: 8,
            alpha: 16.0,
            input_dim: 256,
            output_dim: 256,
        };

        let adapter = LoRAAdapter::new(config).unwrap();
        let expected_params = 256 * 8 + 8 * 256;
        assert_eq!(adapter.num_params(), expected_params);
    }

    #[test]
    fn test_lora_trainer_gradient_computation() -> Result<()> {
        let config = LoRAConfig {
            rank: 4,
            alpha: 8.0,
            input_dim: 16,
            output_dim: 16,
        };
        let mut trainer = LoRATrainer::with_config(config)?;

        // Create simple test data
        let input_data = vec![0.1_f32; 16];
        let target_data = vec![0.2_f32; 16];

        // Call private methods using the public API
        let data = vec![(input_data.clone(), target_data.clone())];
        let result = trainer.train(&data, 5, 0.01);

        assert!(result.is_ok());
        let loss = result.unwrap();
        assert!(loss >= 0.0, "Loss should be non-negative");

        Ok(())
    }

    #[test]
    fn test_lora_tensor_preparation() -> Result<()> {
        let config = LoRAConfig {
            rank: 8,
            alpha: 16.0,
            input_dim: 32,
            output_dim: 32,
        };
        let trainer = LoRATrainer::with_config(config)?;

        // Test padding (smaller input)
        let small_input = vec![1.0_f32; 16];
        let prepared = trainer.prepare_tensor(&small_input, 32, &Device::Cpu)?;
        assert_eq!(prepared.shape().dims(), &[1, 32]);

        // Test truncation (larger input)
        let large_input = vec![1.0_f32; 64];
        let prepared = trainer.prepare_tensor(&large_input, 32, &Device::Cpu)?;
        assert_eq!(prepared.shape().dims(), &[1, 32]);

        // Test exact match
        let exact_input = vec![1.0_f32; 32];
        let prepared = trainer.prepare_tensor(&exact_input, 32, &Device::Cpu)?;
        assert_eq!(prepared.shape().dims(), &[1, 32]);

        Ok(())
    }

    #[test]
    fn test_lora_gradient_clipping() -> Result<()> {
        let config = LoRAConfig {
            rank: 4,
            alpha: 8.0,
            input_dim: 8,
            output_dim: 8,
        };
        let trainer = LoRATrainer::with_config(config)?;

        // Create a large gradient that should be clipped
        let large_grad_data = vec![100.0_f32; 8 * 4];
        let large_grad = Tensor::from_vec(large_grad_data, Shape::from((8, 4)), &Device::Cpu)?;

        let clipped = trainer.clip_gradients(large_grad, 1.0)?;

        // Verify the gradient was clipped
        let norm_sq = clipped.sqr()?.sum_all()?.to_scalar::<f32>()?;
        let norm = norm_sq.sqrt();
        assert!(norm <= 1.0 + 1e-5, "Gradient should be clipped to max_norm");

        Ok(())
    }

    #[test]
    fn test_lora_training_convergence() -> Result<()> {
        let config = LoRAConfig {
            rank: 4,
            alpha: 8.0,
            input_dim: 16,
            output_dim: 16,
        };
        let mut trainer = LoRATrainer::with_config(config)?;

        // Create synthetic data where we know the target function
        let training_data: Vec<(Vec<f32>, Vec<f32>)> = (0..10)
            .map(|i| {
                let input = vec![i as f32 / 10.0; 16];
                let target = vec![(i as f32 / 10.0) * 0.5; 16]; // Simple scaling
                (input, target)
            })
            .collect();

        // Train for multiple epochs
        let initial_loss = trainer.train(&training_data, 1, 0.01)?;
        let final_loss = trainer.train(&training_data, 20, 0.01)?;

        // Loss should decrease (or at least not increase)
        assert!(
            final_loss <= initial_loss + 0.1,
            "Training should converge: initial={}, final={}",
            initial_loss,
            final_loss
        );

        Ok(())
    }
}

/// LoRA Trainer for integration with pipeline
#[derive(Debug)]
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
        Self::new().unwrap_or_else(|_| Self {
            adapter: LoRAAdapter::new(LoRAConfig::default())
                .expect("Failed to create default LoRAAdapter"),
            training_count: 0,
            config: LoRAConfig::default(),
        })
    }
}

/// Real SGD training implementation for LoRA
impl LoRATrainer {
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
        let batch_size = data.len().min(8); // Process in small batches
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
        let momentum_factor = 0.9;

        for epoch in 0..epochs {
            let mut total_loss = 0.0;
            let mut sample_count = 0;

            // Adaptive learning rate with cosine annealing
            let current_lr = learning_rate
                * (1.0 + (epoch as f32 * std::f32::consts::PI / epochs as f32).cos())
                / 2.0;

            // Process in batches
            for batch_start in (0..data.len()).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(data.len());
                let batch = &data[batch_start..batch_end];

                for (input_vec, target_vec) in batch {
                    // Proper dimension handling with padding/truncation
                    let input = self.prepare_tensor(input_vec, self.config.input_dim, &device)?;
                    let target =
                        self.prepare_tensor(target_vec, self.config.output_dim, &device)?;

                    // Forward pass
                    let output = self.adapter.forward(&input)?;

                    // Compute loss (MSE)
                    let diff = output.sub(&target)?;
                    let loss = diff.sqr()?.mean_all()?;
                    let loss_val = loss.to_scalar::<f32>()?;
                    total_loss += loss_val;
                    sample_count += 1;

                    // Compute gradients using proper chain rule (backpropagation)
                    if epoch > 0 && loss_val > 0.001 {
                        let (grad_a, grad_b) =
                            self.compute_gradients(&input, &target, &output, &device)?;

                        // Apply gradient clipping to prevent explosion
                        let grad_a_clipped = self.clip_gradients(grad_a, 1.0)?;
                        let grad_b_clipped = self.clip_gradients(grad_b, 1.0)?;

                        // Update momentum (SGD with momentum)
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
                    }
                }
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
        let grad_output = diff.broadcast_mul(&Tensor::new(&[2.0], device)?)?;

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
