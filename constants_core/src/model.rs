// Copyright 2025 Jason Van Pham (Niodoo)
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of Niodoo TCS.
// For commercial licensing, see LICENSE-COMMERCIAL.md

// Model-specific constants - No hardcoding allowed!
// All model-related magic numbers must be defined here

use once_cell::sync::Lazy;
use std::collections::HashMap;

// ============================================================================
// Qwen Model Token IDs
// ============================================================================

/// Qwen 2.5 End-of-Sequence token IDs
pub const QWEN_25_EOS_TOKEN_PRIMARY: u32 = 151643;
pub const QWEN_25_EOS_TOKEN_SECONDARY: u32 = 151644;
pub const QWEN_25_EOS_TOKEN_TERTIARY: u32 = 151645;

/// Array of all Qwen 2.5 EOS tokens for convenience
pub const QWEN_25_EOS_TOKENS: [u32; 3] = [
    QWEN_25_EOS_TOKEN_PRIMARY,
    QWEN_25_EOS_TOKEN_SECONDARY,
    QWEN_25_EOS_TOKEN_TERTIARY,
];

/// Qwen 3.0 End-of-Sequence token IDs (if different from 2.5)
pub const QWEN_30_EOS_TOKEN_PRIMARY: u32 = 151645;

// ============================================================================
// Model Configuration Constants
// ============================================================================

/// Qwen 2.5 7B model configuration
pub const QWEN_25_7B_VOCAB_SIZE: usize = 152064;
pub const QWEN_25_7B_HIDDEN_SIZE: usize = 3584;
pub const QWEN_25_7B_INTERMEDIATE_SIZE: usize = 18944;
pub const QWEN_25_7B_NUM_HIDDEN_LAYERS: usize = 28;
pub const QWEN_25_7B_NUM_ATTENTION_HEADS: usize = 28;
pub const QWEN_25_7B_NUM_KEY_VALUE_HEADS: usize = 4;
pub const QWEN_25_7B_MAX_POSITION_EMBEDDINGS: usize = 32768;
pub const QWEN_25_7B_ROPE_THETA: f64 = 1000000.0;
pub const QWEN_25_7B_RMS_NORM_EPS: f64 = 1e-6;

/// Qwen 30B model configuration
pub const QWEN_30B_VOCAB_SIZE: usize = 152064;
pub const QWEN_30B_HIDDEN_SIZE: usize = 5120;
pub const QWEN_30B_NUM_HIDDEN_LAYERS: usize = 40;

// ============================================================================
// Model Inference Parameters
// ============================================================================

/// Default temperature for model sampling
pub const DEFAULT_MODEL_TEMPERATURE: f32 = 0.7;

/// Default top-p for nucleus sampling
pub const DEFAULT_MODEL_TOP_P: f32 = 0.9;

/// Default top-k for sampling
pub const DEFAULT_MODEL_TOP_K: usize = 40;

/// Maximum tokens to generate
pub const DEFAULT_MAX_TOKENS: usize = 2048;

/// Default repetition penalty
pub const DEFAULT_REPETITION_PENALTY: f32 = 1.1;

// ============================================================================
// Model Paths and Files
// ============================================================================

/// Default model directory relative to home
pub const DEFAULT_MODEL_DIR: &str = "models";

/// Qwen model file patterns
pub const QWEN_MODEL_SAFETENSORS: &str = "model.safetensors";
pub const QWEN_MODEL_CONFIG: &str = "config.json";
pub const QWEN_TOKENIZER_FILE: &str = "tokenizer.json";
pub const QWEN_TOKENIZER_CONFIG: &str = "tokenizer_config.json";

/// GGUF model file extension
pub const GGUF_FILE_EXTENSION: &str = "gguf";

// ============================================================================
// Model Performance Targets
// ============================================================================

/// Target time-to-first-token in milliseconds
pub const TARGET_TTFT_MS: f32 = 50.0;

/// Target time-per-output-token in milliseconds
pub const TARGET_TPOT_MS: f32 = 20.0;

/// Target throughput in tokens per second
pub const TARGET_THROUGHPUT_TPS: f32 = 50.0;

/// Maximum model loading time in seconds
pub const MAX_MODEL_LOAD_TIME_SECS: u64 = 30;

// ============================================================================
// CUDA Configuration
// ============================================================================

/// CUDA compute capability for RTX 5090/5080
pub const CUDA_COMPUTE_CAP_SM120: &str = "120";
pub const CUDA_ARCH_SM120: &str = "sm_120";

/// CUDA compute capability for RTX 4090
pub const CUDA_COMPUTE_CAP_SM89: &str = "89";
pub const CUDA_ARCH_SM89: &str = "sm_89";

/// CUDA compute capability for RTX A6000
pub const CUDA_COMPUTE_CAP_SM86: &str = "86";
pub const CUDA_ARCH_SM86: &str = "sm_86";

// ============================================================================
// Model Type Identifiers
// ============================================================================

/// Model type strings for telemetry
pub const MODEL_TYPE_QWEN_25_7B: &str = "qwen2.5-7b";
pub const MODEL_TYPE_QWEN_25_7B_AWQ: &str = "qwen2.5-7b-awq";
pub const MODEL_TYPE_QWEN_30B_AWQ: &str = "qwen3-30b-awq";
pub const MODEL_TYPE_QWEN_7B_GGUF: &str = "qwen-7b-gguf";

// ============================================================================
// Dynamic Model Registry
// ============================================================================

/// Registry of model configurations (populated at runtime)
pub static MODEL_REGISTRY: Lazy<HashMap<&'static str, ModelConfig>> = Lazy::new(|| {
    let mut registry = HashMap::new();

    registry.insert(
        MODEL_TYPE_QWEN_25_7B,
        ModelConfig {
            vocab_size: QWEN_25_7B_VOCAB_SIZE,
            hidden_size: QWEN_25_7B_HIDDEN_SIZE,
            num_layers: QWEN_25_7B_NUM_HIDDEN_LAYERS,
            eos_tokens: &QWEN_25_EOS_TOKENS,
        },
    );

    registry.insert(
        MODEL_TYPE_QWEN_30B_AWQ,
        ModelConfig {
            vocab_size: QWEN_30B_VOCAB_SIZE,
            hidden_size: QWEN_30B_HIDDEN_SIZE,
            num_layers: QWEN_30B_NUM_HIDDEN_LAYERS,
            eos_tokens: &[QWEN_30_EOS_TOKEN_PRIMARY],
        },
    );

    registry
});

/// Model configuration structure
pub struct ModelConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub eos_tokens: &'static [u32],
}
