/// Hardware configuration for distributed GPU inference
/// Optimized for RTX Quadro 6000 (beelink) and RTX 5080-Q (laptop)

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareConfig {
    pub beelink_gpu: GPUConfig,
    pub laptop_gpu: GPUConfig,
    pub inference_optimization: InferenceSettings,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GPUConfig {
    pub name: String,
    pub vram_gb: usize,
    pub cuda_version: String,
    pub max_batch_size: usize,
    pub optimal_quant: String,
    pub power_limit: Option<usize>,
    pub thermal_throttle_temp: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceSettings {
    pub use_fp8: bool,  // 20% VRAM reduction
    pub enable_flash_attention: bool,
    pub kv_cache_dtype: String,
    pub max_context_window: usize,
    pub tensor_parallel_size: usize,
}

impl Default for HardwareConfig {
    fn default() -> Self {
        Self {
            beelink_gpu: GPUConfig {
                name: "RTX Quadro 6000".to_string(),
                vram_gb: 24,
                cuda_version: "11.8".to_string(),
                max_batch_size: 32,
                optimal_quant: "Q4_K_M".to_string(),  // 60 tokens/s for 7B
                power_limit: Some(295),
                thermal_throttle_temp: 88,
            },
            laptop_gpu: GPUConfig {
                name: "RTX 5080-Q".to_string(),
                vram_gb: 16,
                cuda_version: "12.8".to_string(),
                max_batch_size: 16,
                optimal_quant: "Q5_K_S".to_string(),  // 150+ tokens/s for 4B
                power_limit: Some(360),  // Plugged in mode
                thermal_throttle_temp: 88,
            },
            inference_optimization: InferenceSettings {
                use_fp8: true,  // Enable for 20% VRAM savings
                enable_flash_attention: true,
                kv_cache_dtype: "fp8".to_string(),
                max_context_window: 32768,  // Safe for both GPUs
                tensor_parallel_size: 1,  // Single GPU per model
            },
        }
    }
}

impl HardwareConfig {
    /// Get recommended settings for vLLM based on GPU
    pub fn get_vllm_args(&self, use_beelink: bool) -> Vec<String> {
        let gpu = if use_beelink { &self.beelink_gpu } else { &self.laptop_gpu };
        
        let mut args = vec![
            format!("--max-model-len={}", self.inference_optimization.max_context_window),
            format!("--gpu-memory-utilization=0.95"),  // Max out VRAM
            format!("--max-num-seqs={}", gpu.max_batch_size),
        ];
        
        if self.inference_optimization.use_fp8 {
            args.push("--dtype=fp8".to_string());
            args.push(format!("--kv-cache-dtype={}", self.inference_optimization.kv_cache_dtype));
        }
        
        if self.inference_optimization.enable_flash_attention {
            args.push("--enable-flash-attn".to_string());
        }
        
        // Thermal protection
        if gpu.vram_gb >= 24 {
            // Quadro 6000 can handle sustained loads
            args.push("--disable-cuda-graph".to_string());  // More stable for long runs
        } else {
            // 5080-Q needs thermal management
            args.push("--enforce-eager".to_string());  // Prevent OOM on laptop
        }
        
        args
    }
    
    /// Check if a model fits in GPU memory
    pub fn can_fit_model(&self, model_size_gb: f32, use_beelink: bool) -> bool {
        let gpu = if use_beelink { &self.beelink_gpu } else { &self.laptop_gpu };
        let available_vram = gpu.vram_gb as f32 * 0.9;  // Leave 10% headroom
        
        let size_with_overhead = if self.inference_optimization.use_fp8 {
            model_size_gb * 0.8  // FP8 saves 20%
        } else {
            model_size_gb
        };
        
        size_with_overhead < available_vram
    }
}