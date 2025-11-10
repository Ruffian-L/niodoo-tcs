use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use futures::future::{abortable, AbortHandle};
use niodoo_core::config::ConsciousnessConfig;
use niodoo_core::memory::guessing_spheres::{EmotionalVector as CoreEmotion, GuessingMemorySystem, SphereId};

// Use local tokenizer types
use crate::tokenizer::PromotedToken;

// Stub types for token_promotion (not available in niodoo-core)
#[derive(Debug, Clone)]
pub struct DynamicTokenizer {
    _private: (),
}

#[derive(Debug, Clone)]
pub struct TokenPromotionEngine {
    _private: (),
}

#[derive(Debug, Clone)]
pub struct TokenizerStats {
    pub base_vocab_size: usize,
    pub extended_vocab_size: usize,
    pub total_usage: u64,
    pub active_extended_tokens: usize,
}

impl TokenizerStats {
    pub fn vocab_size(&self) -> usize {
        self.base_vocab_size + self.extended_vocab_size
    }
    
    pub fn oov_rate(&self) -> f64 {
        if self.base_vocab_size == 0 {
            0.0
        } else {
            self.extended_vocab_size as f64 / self.base_vocab_size as f64
        }
    }
}

#[derive(Debug, Clone)]
pub struct MergeStats {
    pub added: usize,
    pub conflicts_resolved: usize,
    pub usage_updated: usize,
}

#[derive(Debug, Clone)]
pub struct PromotionConfig {
    pub min_promotion_score: f64,
    pub max_candidates_per_cycle: usize,
    pub consensus_threshold: f64,
    pub pruning_min_usage: u64,
}

#[derive(Debug, Clone)]
pub struct PromotionCycleResult {

#[derive(Debug, Clone)]
pub struct RemoteVocabulary {
    _private: (),
}
    pub promoted: Vec<PromotedToken>,
    pub pruned: usize,
    pub duration: std::time::Duration,
}

impl TokenPromotionEngine {
    pub fn new(_pattern: (), _consensus: (), _tokenizer: std::sync::Arc<tokio::sync::RwLock<DynamicTokenizer>>) -> Self {
        Self { _private: () }
    }
    
    pub fn with_config(self, _config: PromotionConfig) -> Self {
        self
    }
}

impl DynamicTokenizer {
    pub fn new(_tokenizer: ()) -> Self {
        Self { _private: () }
    }
}
use tokio::{fs, sync::RwLock, time::interval};
use tracing::{debug, info, instrument, warn};

use crate::erag::{CollapseResult, EragMemory};
use crate::metrics::tokenizer_metrics;
use crate::torus::PadGhostState;

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

impl Drop for DynamicTokenizerManager {
    fn drop(&mut self) {
        // Cooperative shutdown for background maintenance loop
        self.shutdown.store(true, Ordering::Relaxed);
        // Use try_lock in Drop to avoid panicking if mutex is poisoned
        if let Ok(mut guard) = self.abort_handle.lock() {
            if let Some(handle) = guard.take() {
                handle.abort();
            }
        }
    }
}

#[derive(Clone)]
pub struct DynamicTokenizerManager {
    tokenizer: Arc<RwLock<DynamicTokenizer>>,
    promotion_engine: Arc<RwLock<TokenPromotionEngine>>,
    memories: Arc<RwLock<GuessingMemorySystem>>,
    metrics: Arc<TokenizationMetrics>,
    promotion_interval: u64,
    state_path: PathBuf,
    shutdown: Arc<AtomicBool>,
    promotion_active: Arc<AtomicBool>,
    abort_handle: Arc<Mutex<Option<AbortHandle>>>,
}

struct TokenizationMetrics {
    last_promotion: Mutex<Option<Instant>>,
    vocab_stats: Mutex<Option<TokenizerStats>>,
    merge_stats: Mutex<Option<MergeStats>>,
    promoted: Mutex<Vec<PromotedToken>>,
}

impl TokenizationMetrics {
    fn new() -> Self {
        Self {
            last_promotion: Mutex::new(None),
            vocab_stats: Mutex::new(None),
            merge_stats: Mutex::new(None),
            promoted: Mutex::new(Vec::new()),
        }
    }

    fn record_promotion(&self, result: &PromotionCycleResult) {
        tokenizer_metrics().record_promotion(
            result.promoted.len(),
            result.pruned,
            result.duration.as_secs_f64() * 1000.0,
        );
        // Handle mutex errors gracefully - log but don't crash
        if let Ok(mut guard) = self.last_promotion.lock() {
            *guard = Some(Instant::now());
        }
        if let Ok(mut promoted) = self.promoted.lock() {
            promoted.clear();
            promoted.extend(result.promoted.clone());
        }
    }

    fn record_stats(&self, stats: &TokenizerStats) {
        tokenizer_metrics().record(stats.vocab_size() as f64, stats.oov_rate());
    }
}

impl DynamicTokenizerManager {
    #[instrument]
    pub async fn initialise(
        tokenizer_path: &Path,
        _node_id: String,
        promotion_interval: u64,
    ) -> Result<Self> {
        // Stub implementation - token promotion types not fully available
        let tokenizer = Arc::new(RwLock::new(DynamicTokenizer::new(())));
        
        let promotion_engine = Arc::new(RwLock::new(
            TokenPromotionEngine::new((), (), tokenizer.clone()).with_config(
                PromotionConfig {
                    min_promotion_score: 0.7,
                    max_candidates_per_cycle: 10,
                    consensus_threshold: 0.66,
                    pruning_min_usage: 10,
                },
            )
        ));

        let state_path = Self::state_path(tokenizer_path);

        let manager = Self {
            tokenizer,
            promotion_engine: Arc::new(RwLock::new(promotion_engine)),
            memories: Arc::new(RwLock::new(GuessingMemorySystem::new())),
            metrics: Arc::new(TokenizationMetrics::new()),
            promotion_interval,
            state_path,
            shutdown: Arc::new(AtomicBool::new(false)),
            promotion_active: Arc::new(AtomicBool::new(false)),
            abort_handle: Arc::new(Mutex::new(None)),
        };

        manager.load_persisted_vocabulary().await?;

        Ok(manager)
    }

    fn state_path(tokenizer_path: &Path) -> PathBuf {
        match tokenizer_path.extension().and_then(|ext| ext.to_str()) {
            Some("json") => tokenizer_path.with_extension("dynamic_state.json"),
            _ => tokenizer_path.with_extension("tokenizer_state.json"),
        }
    }

    pub async fn spawn_maintenance(self: &Arc<Self>) {
        if self.promotion_interval == 0 {
            return;
        }

        if self
            .promotion_active
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return;
        }

        let manager = Arc::clone(self);
        let manager_for_abort = Arc::clone(self);

        let task = async move {
            let mut ticker = interval(Duration::from_secs(manager.promotion_interval));
            let shutdown = manager.shutdown.clone();

            loop {
                if shutdown.load(Ordering::Relaxed) {
                    break;
                }

                tokio::select! {
                    _ = ticker.tick() => {
                        if shutdown.load(Ordering::Relaxed) { break; }
                        if let Err(err) = manager.run_promotion_cycle().await {
                            warn!(%err, "token promotion cycle failed");
                        }
                    }
                }
            }

            manager.promotion_active.store(false, Ordering::Relaxed);
        };
        let (abortable_task, abort_handle) = abortable(task);
        {
            let mut slot = manager_for_abort
                .abort_handle
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            *slot = Some(abort_handle);
        }
        let _ = tokio::spawn(abortable_task);
    }

    pub fn request_shutdown(&self) {
        self.shutdown.store(true, Ordering::Relaxed);
    }

    #[instrument(skip(self, prompt, collapse, pad_state))]
    pub async fn process(
        &self,
        prompt: &str,
        collapse: &str,
        pad_state: &PadGhostState,
    ) -> Result<TokenizerOutput> {
        // Opportunistically run promotion each call when no background interval is configured.
        if self.promotion_interval == 0 {
            if let Err(err) = self.run_promotion_cycle().await {
                warn!(%err, "token promotion during process failed");
            }
        }

        let augmented_prompt = format!(
            "Prompt: {}\nContext: {}\nEntropy: {:.3}",
            Self::snippet(prompt, 96),
            Self::snippet(collapse, 160),
            pad_state.entropy
        );

        let mut tokenizer = self.tokenizer.write().await;
        let tokens = tokenizer.encode_extended(&augmented_prompt)?;
        let stats = tokenizer.stats();
        let vocab_size = stats.vocab_size();
        let oov_rate = stats.oov_rate();
        {
            let mut slot = self
                .metrics
                .vocab_stats
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            *slot = Some(stats.clone());
        }
        self.metrics.record_stats(&stats);

        info!(
            tokens = tokens.len(),
            vocab_size, oov_rate, "tokenized prompt"
        );

        let promoted_snapshot = self
            .metrics
            .promoted
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();

        Ok(TokenizerOutput {
            tokens,
            augmented_prompt,
            promoted_tokens: promoted_snapshot,
            vocab_size,
            oov_rate,
            failure_type: None,
            failure_details: None,
        })
    }

    #[instrument(skip(self, prompt, collapse, pad_state, memories))]
    pub async fn process_with_memories(
        &self,
        prompt: &str,
        collapse: &CollapseResult,
        pad_state: &PadGhostState,
        memories: Vec<EragMemory>,
    ) -> Result<TokenizerOutput> {
        {
            let mut guess = self.memories.write().await;
            guess.clear();
            for memory in &memories {
                let emotion = CoreEmotion::new(
                    memory.emotional_vector.joy,
                    memory.emotional_vector.sadness,
                    memory.emotional_vector.anger,
                    memory.emotional_vector.fear,
                    memory.emotional_vector.surprise,
                );
                guess.store_memory(
                    SphereId(format!("erag-{}", memory.timestamp)),
                    memory.input.clone(),
                    [0.0, 0.0, 0.0],
                    emotion,
                    memory.erag_context.join("\n"),
                );
            }
        }

        self.process(prompt, &collapse.aggregated_context, pad_state)
            .await
    }

    #[instrument(skip(self))]
    pub async fn run_promotion_cycle(&self) -> Result<()> {
        if std::env::var("TOKEN_PROMOTION_DISABLED")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
        {
            return Ok(());
        }

        if self
            .promotion_active
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            debug!("token promotion cycle already running; skipping");
            return Ok(());
        }

        struct PromotionGuard<'a>(&'a AtomicBool);

        impl<'a> Drop for PromotionGuard<'a> {
            fn drop(&mut self) {
                self.0.store(false, Ordering::Release);
            }
        }

        let _guard = PromotionGuard(&self.promotion_active);

        let promotion_engine = self.promotion_engine.clone();
        let memories = self.memories.clone();

        let start = Instant::now();
        let promotion_result = {
            let engine = promotion_engine.read().await;
            let memories = memories.read().await;
            engine.run_promotion_cycle(&memories).await
        }?;
        let elapsed = start.elapsed();

        self.persist_vocabulary().await?;
        self.metrics.record_promotion(&promotion_result);

        debug!(
            cycle_ms = elapsed.as_millis(),
            promoted = promotion_result.promoted.len(),
            "promotion cycle completed"
        );

        Ok(())
    }

    #[instrument(skip(self, remote))]
    pub async fn merge_remote_vocabulary(&self, remote: RemoteVocabulary) -> Result<()> {
        let mut tokenizer = self.tokenizer.write().await;
        let stats = tokenizer.merge_remote_vocabulary(&remote)?;
        drop(tokenizer);
        *self
            .metrics
            .merge_stats
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(stats);
        self.persist_vocabulary().await?;
        Ok(())
    }

    pub async fn export_vocabulary(&self) -> Result<RemoteVocabulary> {
        let tokenizer = self.tokenizer.read().await;
        Ok(tokenizer.export_vocabulary())
    }

    pub async fn vocab_stats(&self) -> Option<TokenizerStats> {
        let guard = self
            .metrics
            .vocab_stats
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        guard.clone()
    }

    pub async fn promoted_tokens(&self) -> Vec<PromotedToken> {
        self.metrics
            .promoted
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    async fn load_persisted_vocabulary(&self) -> Result<()> {
        if fs::metadata(&self.state_path).await.is_err() {
            return Ok(());
        }
        let data = fs::read(&self.state_path).await?;
        let remote: RemoteVocabulary = serde_json::from_slice(&data)?;
        self.merge_remote_vocabulary(remote).await
    }

    async fn persist_vocabulary(&self) -> Result<()> {
        let tokens = {
            let promoted = self
                .metrics
                .promoted
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            promoted.clone()
        };

        if tokens.is_empty() {
            return Ok(());
        }

        let vocab = self.export_vocabulary().await?;
        let data = serde_json::to_vec_pretty(&vocab)?;

        if let Some(parent) = self.state_path.parent() {
            fs::create_dir_all(parent).await?;
        }

        let mut tmp_path = self.state_path.clone();
        tmp_path.set_extension("tmp");
        fs::write(&tmp_path, &data).await?;
        fs::rename(&tmp_path, &self.state_path).await?;
        Ok(())
    }

    fn snippet(text: &str, limit: usize) -> String {
        if text.is_empty() {
            return "∅".to_string();
        }
        let mut out = String::with_capacity(limit + 1);
        let mut count = 0;
        for ch in text.chars() {
            if count >= limit {
                out.push('…');
                break;
            }
            out.push(ch);
            count += 1;
        }
        out.trim().to_string()
    }
}
