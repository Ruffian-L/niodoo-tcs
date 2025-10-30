// Copyright 2025 Jason Van Pham (Niodoo)
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of Niodoo TCS.
// For commercial licensing, see LICENSE-COMMERCIAL.md

//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

use anyhow::Error as AnyError;
use ndarray::ShapeError;
#[cfg(feature = "onnx")]
use ort::OrtError;
use thiserror::Error;
#[cfg(feature = "tokenizers")]
use tokenizers::Error as TokenizerError;

/// Result alias for operations performed by the Qwen embedder.
pub type QwenResult<T> = Result<T, QwenError>;

/// Error variants emitted by the Qwen embedder pipeline.
#[derive(Debug, Error)]
pub enum QwenError {
    #[cfg(feature = "tokenizers")]
    #[error("tokenization failed")]
    Tokenizer {
        #[source]
        source: TokenizerError,
    },
    #[error("configuration validation failed")]
    ConfigValidation {
        #[source]
        source: AnyError,
    },
    #[error("prompt produced no tokens")]
    EmptyPrompt,
    #[error("inference step received no tokens")]
    EmptyInferenceStep,
    #[error("attention mask length {mask_len} did not match token count {token_len}")]
    AttentionMaskMismatch { mask_len: usize, token_len: usize },
    #[error("sequence too long: total context {total_seq_len} would exceed max {max_seq_len}")]
    SequenceTooLong {
        total_seq_len: usize,
        max_seq_len: usize,
    },
    #[error("failed to build tensor {name}")]
    TensorBuild {
        name: &'static str,
        #[source]
        source: ShapeError,
    },
    #[cfg(feature = "onnx")]
    #[error("ONNX inference failed")]
    OnnxInference {
        #[source]
        source: OrtError,
    },
    #[error("ONNX model produced no outputs")]
    NoOutputs,
    #[error("unexpected output count: expected at least {expected}, got {actual}")]
    UnexpectedOutputCount { expected: usize, actual: usize },
    #[error("{name}: tensor rank mismatch (expected 4D, got {:?})", shape)]
    TensorRankMismatch { name: String, shape: Vec<usize> },
    #[error("{name}: failed to materialise tensor")]
    TensorMaterialise {
        name: String,
        #[source]
        source: ShapeError,
    },
    #[error("{name}: tensor is neither f32 nor f16")]
    UnsupportedTensorType { name: String },
    #[error("{name}: failed to concatenate KV cache")]
    KvConcat {
        name: String,
        #[source]
        source: ShapeError,
    },
    #[error(
        "unexpected KV cache lengths for {name}: previous={previous}, new_tokens={new_tokens}, present={present}"
    )]
    InvalidKvShape {
        name: String,
        previous: usize,
        new_tokens: usize,
        present: usize,
    },
    #[error("invalid logits shape: total={total_elements}, vocab={vocab_size}")]
    InvalidLogitsShape {
        total_elements: usize,
        vocab_size: usize,
    },
    #[error("insufficient logits to extract embedding")]
    InsufficientLogits,
}

// From implementations for automatic error conversions with ?
impl From<AnyError> for QwenError {
    fn from(source: AnyError) -> Self {
        QwenError::ConfigValidation { source }
    }
}

#[cfg(feature = "onnx")]
impl From<OrtError> for QwenError {
    fn from(source: OrtError) -> Self {
        QwenError::OnnxInference { source }
    }
}

impl From<ShapeError> for QwenError {
    fn from(source: ShapeError) -> Self {
        // Default mapping for shape errors - can be specialized by the caller
        QwenError::TensorBuild {
            name: "unknown",
            source,
        }
    }
}

#[cfg(feature = "tokenizers")]
impl From<TokenizerError> for QwenError {
    fn from(source: TokenizerError) -> Self {
        QwenError::Tokenizer { source }
    }
}
