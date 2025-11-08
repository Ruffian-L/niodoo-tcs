//! Stub implementation of token_manager when niodoo-core is not available
//! Provides minimal functionality to allow compilation without niodoo-core dependency

use std::path::Path;
use std::sync::{Arc, Mutex};
use anyhow::Result;
use tracing::instrument;

use crate::erag::CollapseResult;
use crate::torus::PadGhostState;
use crate::tokenizer::{PromotedToken, TokenizerOutput as BaseTokenizerOutput};

// Re-export TokenizerOutput with additional fields for compatibility
#[derive(Debug, Clone)]
pub struct TokenizerOutput {
    pub tokens: Vec<u32>,
    pub augmented_prompt: String,
    pub promoted_tokens: Vec<PromotedToken>,
    pub vocab_size: usize,
    pub oov_rate: f64,
    pub failure_type: Option<String>,
    pub failure_details: Option<String>,
}

impl From<BaseTokenizerOutput> for TokenizerOutput {
    fn from(base: BaseTokenizerOutput) -> Self {
        Self {
            tokens: base.tokens,
            augmented_prompt: base.augmented_prompt,
            promoted_tokens: base.promoted_tokens,
            vocab_size: base.vocab_size,
            oov_rate: base.oov_rate,
            failure_type: None,
            failure_details: None,
        }
    }
}

#[derive(Clone)]
pub struct DynamicTokenizerManager {
    tokenizer: Arc<std::sync::Mutex<crate::tokenizer::TokenizerEngine>>,
}

impl DynamicTokenizerManager {
    #[instrument]
    pub async fn initialise(
        tokenizer_path: &Path,
        _node_id: String,
        _promotion_interval: u64,
    ) -> Result<Self> {
        let tokenizer = crate::tokenizer::TokenizerEngine::new(tokenizer_path, 0.1)?;
        Ok(Self {
            tokenizer: Arc::new(std::sync::Mutex::new(tokenizer)),
        })
    }

    #[instrument(skip(self, prompt, collapse, pad_state))]
    pub async fn process(
        &self,
        prompt: &str,
        collapse: &str,
        pad_state: &PadGhostState,
    ) -> Result<TokenizerOutput> {
        // Create a minimal CollapseResult from the collapse string
        let collapse_result = CollapseResult {
            aggregated_context: collapse.to_string(),
            top_hits: vec![],
            average_similarity: 0.0,
            curator_quality: None,
        };

        // Use the basic tokenizer - lock the mutex to get mutable access
        let mut tokenizer = self.tokenizer.lock().unwrap();
        let base_output = tokenizer.process(
            prompt,
            &collapse_result,
            pad_state,
            0.0, // entropy_mean
        )?;

        Ok(TokenizerOutput::from(base_output))
    }

    #[instrument(skip(self, prompt, collapse, pad_state, _memories))]
    pub async fn process_with_memories(
        &self,
        prompt: &str,
        collapse: &CollapseResult,
        pad_state: &PadGhostState,
        _memories: &[crate::erag::EragMemory],
    ) -> Result<TokenizerOutput> {
        // For stub, convert CollapseResult to string and use regular process
        let collapse_str = &collapse.aggregated_context;
        self.process(prompt, collapse_str, pad_state).await
    }

    pub async fn promoted_tokens(&self) -> Vec<PromotedToken> {
        vec![] // Stub returns empty
    }

    pub async fn spawn_maintenance(&self) -> Result<()> {
        Ok(()) // Stub does nothing
    }
}

