//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use crate::memory::guessing_spheres::GuessingMemorySystem;

use super::consensus::ConsensusVote;
use super::dynamic_tokenizer::TokenizerStats;
use super::pattern_discovery::PatternDiscoveryEngine;
use super::{ConsensusEngine, DynamicTokenizer, PromotedToken, TokenCandidate};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PromotionConfig {
    pub min_promotion_score: f64,
    pub max_candidates_per_cycle: usize,
    pub consensus_threshold: f64,
    pub pruning_min_usage: u64,
}

impl Default for PromotionConfig {
    fn default() -> Self {
        Self {
            min_promotion_score: 0.7,
            max_candidates_per_cycle: 10,
            consensus_threshold: 0.66,
            pruning_min_usage: 10,
        }
    }
}

pub struct TokenPromotionEngine {
    pattern_discovery: Arc<PatternDiscoveryEngine>,
    consensus: Arc<ConsensusEngine>,
    tokenizer: Arc<RwLock<DynamicTokenizer>>,
    embedding_generator: Arc<MockEmbeddingGenerator>,
    config: PromotionConfig,
}

impl TokenPromotionEngine {
    pub fn new(
        pattern_discovery: Arc<PatternDiscoveryEngine>,
        consensus: Arc<ConsensusEngine>,
        tokenizer: Arc<RwLock<DynamicTokenizer>>,
    ) -> Self {
        Self {
            pattern_discovery,
            consensus,
            tokenizer,
            embedding_generator: Arc::new(MockEmbeddingGenerator::default()),
            config: PromotionConfig::default(),
        }
    }

    pub fn with_config(mut self, config: PromotionConfig) -> Self {
        self.config = config;
        self
    }

    /// Encode text using the current dynamic vocabulary, tracking usage statistics.
    pub async fn encode_with_dynamic_vocab(&self, text: &str) -> Result<Vec<u32>> {
        let mut tokenizer = self.tokenizer.write().await;
        tokenizer.encode_extended(text)
    }

    /// Compute a promotion score for an arbitrary byte sequence against the current memory system.
    pub async fn score_candidate(
        &self,
        byte_seq: &[u8],
        memory_system: &GuessingMemorySystem,
    ) -> Result<f64> {
        let candidates = self
            .pattern_discovery
            .discover_candidates(memory_system)
            .await?;

        let score = candidates
            .into_iter()
            .find(|candidate| candidate.bytes.as_slice() == byte_seq)
            .map(|candidate| candidate.promotion_score())
            .unwrap_or(0.0);

        Ok(score)
    }

    pub async fn run_promotion_cycle(
        &self,
        memory_system: &GuessingMemorySystem,
    ) -> Result<PromotionCycleResult> {
        let start = Instant::now();
        tracing::info!("starting token promotion cycle");

        self.pattern_discovery
            .rebuild_spatial_index(memory_system)
            .await;

        let mut candidates = self
            .pattern_discovery
            .discover_candidates(memory_system)
            .await?;
        tracing::info!(
            candidate_count = candidates.len(),
            "pattern discovery complete"
        );

        candidates
            .retain(|candidate| candidate.promotion_score() >= self.config.min_promotion_score);
        if candidates.len() > self.config.max_candidates_per_cycle {
            candidates.truncate(self.config.max_candidates_per_cycle);
        }

        let mut promoted_tokens = Vec::new();
        let mut rejected_candidates = Vec::new();

        for candidate in candidates {
            let vote = self.consensus.propose_token(&candidate).await?;
            if vote.approved {
                let token = self.promote_candidate(candidate, vote).await?;
                promoted_tokens.push(token);
            } else {
                rejected_candidates.push(candidate);
            }
        }

        let pruned = self
            .tokenizer
            .write()
            .await
            .prune_unused(self.config.pruning_min_usage);
        let duration = start.elapsed();

        Ok(PromotionCycleResult {
            promoted: promoted_tokens,
            rejected: rejected_candidates,
            pruned,
            duration,
        })
    }

    async fn promote_candidate(
        &self,
        candidate: TokenCandidate,
        vote: ConsensusVote,
    ) -> Result<PromotedToken> {
        let embedding = self
            .embedding_generator
            .generate_embedding(&candidate.bytes)
            .await?;

        let token_id = {
            let tokenizer = self.tokenizer.read().await;
            tokenizer.next_token_id()
        };

        let promoted = PromotedToken {
            token_id,
            bytes: candidate.bytes.clone(),
            embedding,
            promotion_score: candidate.promotion_score(),
            promoted_at: SystemTime::now(),
        };

        {
            let mut tokenizer = self.tokenizer.write().await;
            tokenizer.add_promoted_token(&promoted)?;
        }

        tracing::info!(
            token_id = token_id,
            score = promoted.promotion_score,
            votes_for = vote.votes_for,
            votes_against = vote.votes_against,
            "promoted token"
        );

        Ok(promoted)
    }

    /// Get tokenizer statistics
    pub async fn tokenizer_stats(&self) -> TokenizerStats {
        let tokenizer = self.tokenizer.read().await;
        tokenizer.stats()
    }
}

#[derive(Debug)]
pub struct PromotionCycleResult {
    pub promoted: Vec<PromotedToken>,
    pub rejected: Vec<TokenCandidate>,
    pub pruned: usize,
    pub duration: Duration,
}

#[derive(Default)]
struct MockEmbeddingGenerator;

impl MockEmbeddingGenerator {
    async fn generate_embedding(&self, bytes: &[u8]) -> Result<Vec<f32>> {
        let mut embedding = vec![0.0_f32; 768];
        if bytes.is_empty() {
            return Ok(embedding);
        }

        let mut checksum = 0_u32;
        for (idx, byte) in bytes.iter().enumerate() {
            checksum = checksum.wrapping_add((*byte as u32) * (idx as u32 + 1));
        }

        let base = (checksum % 997) as f32 / 997.0;
        for (idx, value) in embedding.iter_mut().enumerate() {
            let source = bytes[idx % bytes.len()] as f32 / 255.0;
            *value = (source + base).fract();
        }

        Ok(embedding)
    }
}
