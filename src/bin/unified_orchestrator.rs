/// Unified Orchestrator - Master Integration Coordinator
///
/// This is the main entry point for the Niodoo consciousness system.
/// It coordinates all components (TDA, Knot Analysis, TQFT, Learning)
/// into a coherent pipeline for consciousness processing.
///
/// Architecture (follows finalREADME):
/// 1. State Extraction → receives consciousness states
/// 2. TDA Pipeline → analyzes topological structure
/// 3. Knot Analyzer → detects and classifies cognitive patterns
/// 4. TQFT Reasoning → applies mathematical reasoning
/// 5. RL Agent → learns and optimizes responses
/// 6. Consensus → evolves vocabulary

use anyhow::{Result, Context};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, error, instrument};

/// Configuration for the unified orchestrator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrchestratorConfig {
    pub name: String,
    pub enable_tda: bool,
    pub enable_knot_analysis: bool,
    pub enable_tqft: bool,
    pub enable_learning: bool,
    pub state_buffer_size: usize,
    pub process_batch_size: usize,
    pub log_level: String,
}

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            name: "Niodoo-Orchestrator".to_string(),
            enable_tda: true,
            enable_knot_analysis: true,
            enable_tqft: true,
            enable_learning: true,
            state_buffer_size: 1000,
            process_batch_size: 100,
            log_level: "info".to_string(),
        }
    }
}

/// Result of consciousness processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessProcessingResult {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub state_id: String,
    pub topological_features: TopologicalAnalysis,
    pub detected_patterns: Vec<CognitivePattern>,
    pub reasoning_applied: Vec<String>,
    pub learning_updates: usize,
}

/// Topological analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologicalAnalysis {
    pub betti_numbers: [usize; 3],
    pub persistent_features: usize,
    pub complexity_score: f32,
}

impl Default for TopologicalAnalysis {
    fn default() -> Self {
        Self {
            betti_numbers: [1, 0, 0],
            persistent_features: 0,
            complexity_score: 0.0,
        }
    }
}

/// Detected cognitive pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitivePattern {
    pub pattern_type: String,
    pub confidence: f32,
    pub persistence: f32,
    pub complexity: f32,
}

/// The main orchestrator
pub struct UnifiedOrchestrator {
    config: Arc<RwLock<OrchestratorConfig>>,
    state_counter: Arc<RwLock<usize>>,
    results_buffer: Arc<RwLock<Vec<ConsciousnessProcessingResult>>>,
}

impl UnifiedOrchestrator {
    /// Create a new orchestrator with given configuration
    pub fn new(config: OrchestratorConfig) -> Self {
        Self {
            config: Arc::new(RwLock::new(config)),
            state_counter: Arc::new(RwLock::new(0)),
            results_buffer: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Process a consciousness state through the full pipeline
    #[instrument(skip(self), name = "process_consciousness_state")]
    pub async fn process_state(&self, state_data: Vec<f32>) -> Result<ConsciousnessProcessingResult> {
        let config = self.config.read().await;
        let mut counter = self.state_counter.write().await;
        *counter += 1;
        let state_id = format!("state_{:06}", counter);
        drop(counter);

        info!("Processing consciousness state: {}", state_id);

        let mut result = ConsciousnessProcessingResult {
            timestamp: chrono::Utc::now(),
            state_id: state_id.clone(),
            topological_features: TopologicalAnalysis::default(),
            detected_patterns: Vec::new(),
            reasoning_applied: Vec::new(),
            learning_updates: 0,
        };

        // Stage 1: Topological Data Analysis
        if config.enable_tda {
            info!("Stage 1: Running TDA pipeline");
            result.topological_features = self.analyze_topology(&state_data).await?;
            result.reasoning_applied.push("TDA".to_string());
        }

        // Stage 2: Knot Analysis
        if config.enable_knot_analysis {
            info!("Stage 2: Running knot analysis");
            result.detected_patterns = self.detect_patterns(&state_data).await?;
            result.reasoning_applied.push("KnotAnalysis".to_string());
        }

        // Stage 3: TQFT Reasoning
        if config.enable_tqft {
            info!("Stage 3: Running TQFT reasoning");
            self.apply_tqft_reasoning(&mut result).await?;
            result.reasoning_applied.push("TQFT".to_string());
        }

        // Stage 4: Learning
        if config.enable_learning {
            info!("Stage 4: Applying learning updates");
            result.learning_updates = self.update_learning(&result).await?;
            result.reasoning_applied.push("Learning".to_string());
        }

        info!(
            "State {} processed successfully with {} stages",
            state_id,
            result.reasoning_applied.len()
        );

        // Store result
        let mut buffer = self.results_buffer.write().await;
        buffer.push(result.clone());

        // Keep only recent results
        if buffer.len() > config.state_buffer_size {
            buffer.drain(0..buffer.len() - config.state_buffer_size);
        }

        Ok(result)
    }

    /// Stage 1: Topological Data Analysis
    #[instrument(skip(self))]
    async fn analyze_topology(&self, state_data: &[f32]) -> Result<TopologicalAnalysis> {
        // Simulate TDA pipeline
        // In real system, this would use Ripser engine from src/tcs/

        // Compute simple features from state vector
        let complexity_score = self.compute_complexity(state_data);

        // Estimate Betti numbers based on data characteristics
        let mean = state_data.iter().sum::<f32>() / state_data.len() as f32;
        let variance = state_data
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>()
            / state_data.len() as f32;

        let b0 = 1; // Always connected
        let b1 = if variance > 0.5 { 1 } else { 0 }; // Has loop if varied
        let b2 = 0; // 2D systems don't have voids in basic analysis

        Ok(TopologicalAnalysis {
            betti_numbers: [b0, b1, b2],
            persistent_features: if b1 > 0 { 1 } else { 0 },
            complexity_score,
        })
    }

    /// Stage 2: Knot Analysis - Detect Cognitive Patterns
    #[instrument(skip(self))]
    async fn detect_patterns(&self, state_data: &[f32]) -> Result<Vec<CognitivePattern>> {
        // Simulate knot/pattern detection
        // In real system, this would use KnotAnalyzer from src/tcs/

        let mut patterns = Vec::new();

        // Detect high-variance pattern
        let mean = state_data.iter().sum::<f32>() / state_data.len() as f32;
        let variance = state_data
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>()
            / state_data.len() as f32;

        if variance > 0.5 {
            patterns.push(CognitivePattern {
                pattern_type: "HighVariance".to_string(),
                confidence: variance.min(1.0),
                persistence: 1.0,
                complexity: variance,
            });
        }

        // Detect oscillatory pattern
        if state_data.len() >= 4 {
            let diffs: Vec<f32> = state_data.windows(2).map(|w| w[1] - w[0]).collect();
            let sign_changes = diffs
                .windows(2)
                .filter(|w| w[0] * w[1] < 0.0)
                .count();

            if sign_changes > 2 {
                patterns.push(CognitivePattern {
                    pattern_type: "Oscillatory".to_string(),
                    confidence: 0.7,
                    persistence: 0.8,
                    complexity: sign_changes as f32 / state_data.len() as f32,
                });
            }
        }

        Ok(patterns)
    }

    /// Stage 3: TQFT Reasoning - Apply topological transformations
    #[instrument(skip(self))]
    async fn apply_tqft_reasoning(&self, result: &mut ConsciousnessProcessingResult) -> Result<()> {
        // Use TQFT engine to infer topological transitions
        use niodoo_feeling::tqft::TQFTEngine;

        let engine = TQFTEngine::new(3)
            .context("Failed to create TQFT engine")?;

        // Infer cobordism from Betti numbers
        let betti_before = [1, 0, 0]; // Baseline
        let betti_after = result.topological_features.betti_numbers;

        if let Some(cobordism) =
            TQFTEngine::infer_cobordism_from_betti(&betti_before, &betti_after)
        {
            result
                .reasoning_applied
                .push(format!("Cobordism::{:?}", cobordism));
        }

        Ok(())
    }

    /// Stage 4: Learning - Update internal models
    #[instrument(skip(self))]
    async fn update_learning(&self, result: &ConsciousnessProcessingResult) -> Result<usize> {
        // Simulate learning updates
        // In real system, would use UntryingAgent from src/evolution/

        let mut updates = 0;

        // Update based on complexity
        if result.topological_features.complexity_score > 0.7 {
            updates += 1; // Complex pattern learned
        }

        // Update based on patterns
        updates += result.detected_patterns.len();

        Ok(updates)
    }

    /// Helper: compute complexity score
    fn compute_complexity(&self, data: &[f32]) -> f32 {
        if data.is_empty() {
            return 0.0;
        }

        let mean = data.iter().sum::<f32>() / data.len() as f32;
        let variance = data
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>()
            / data.len() as f32;

        let range = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
            - data.iter().cloned().fold(f32::INFINITY, f32::min);

        // Combine variance and range for complexity
        (variance.sqrt() + range) / 2.0
    }

    /// Get statistics on processed states
    pub async fn get_statistics(&self) -> Result<OrchestratorStatistics> {
        let counter = self.state_counter.read().await;
        let buffer = self.results_buffer.read().await;

        let avg_complexity = if !buffer.is_empty() {
            buffer
                .iter()
                .map(|r| r.topological_features.complexity_score)
                .sum::<f32>()
                / buffer.len() as f32
        } else {
            0.0
        };

        let total_patterns_detected = buffer
            .iter()
            .map(|r| r.detected_patterns.len())
            .sum::<usize>();

        Ok(OrchestratorStatistics {
            total_states_processed: *counter,
            states_in_buffer: buffer.len(),
            avg_complexity_score: avg_complexity,
            total_patterns_detected,
            total_learning_updates: buffer.iter().map(|r| r.learning_updates).sum(),
        })
    }

    /// Reset the orchestrator
    pub async fn reset(&self) -> Result<()> {
        let mut counter = self.state_counter.write().await;
        *counter = 0;
        let mut buffer = self.results_buffer.write().await;
        buffer.clear();
        Ok(())
    }
}

/// Statistics about orchestrator performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrchestratorStatistics {
    pub total_states_processed: usize,
    pub states_in_buffer: usize,
    pub avg_complexity_score: f32,
    pub total_patterns_detected: usize,
    pub total_learning_updates: usize,
}

/// Main entry point
#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    info!("🎼 Niodoo Unified Orchestrator Starting");

    // Create orchestrator
    let config = OrchestratorConfig::default();
    let orchestrator = UnifiedOrchestrator::new(config);

    info!("✓ Orchestrator initialized");

    // Process a few test states
    for i in 0..5 {
        // Generate test consciousness state
        let state = vec![
            0.5 + (i as f32 * 0.1).sin(),
            0.3 + (i as f32 * 0.2).cos(),
            0.7 + (i as f32 * 0.15).sin().abs(),
            0.2 + (i as f32 * 0.3).cos().abs(),
        ];

        match orchestrator.process_state(state).await {
            Ok(result) => {
                info!(
                    "✓ State {} processed: {} patterns detected",
                    result.state_id,
                    result.detected_patterns.len()
                );
            }
            Err(e) => {
                error!("✗ Failed to process state: {}", e);
            }
        }
    }

    // Print final statistics
    match orchestrator.get_statistics().await {
        Ok(stats) => {
            info!("📊 Final Statistics:");
            info!("  Total states processed: {}", stats.total_states_processed);
            info!("  States in buffer: {}", stats.states_in_buffer);
            info!(
                "  Avg complexity score: {:.3}",
                stats.avg_complexity_score
            );
            info!(
                "  Total patterns detected: {}",
                stats.total_patterns_detected
            );
            info!(
                "  Total learning updates: {}",
                stats.total_learning_updates
            );
        }
        Err(e) => {
            error!("Failed to get statistics: {}", e);
        }
    }

    info!("✓ Orchestrator completed successfully");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_orchestrator_creation() {
        let config = OrchestratorConfig::default();
        let orchestrator = UnifiedOrchestrator::new(config);
        let stats = orchestrator.get_statistics().await.unwrap();
        assert_eq!(stats.total_states_processed, 0);
    }

    #[tokio::test]
    async fn test_process_state() {
        let orchestrator = UnifiedOrchestrator::new(OrchestratorConfig::default());
        let state = vec![0.5, 0.3, 0.7, 0.2];
        let result = orchestrator.process_state(state).await.unwrap();
        assert!(!result.reasoning_applied.is_empty());
    }

    #[tokio::test]
    async fn test_statistics() {
        let orchestrator = UnifiedOrchestrator::new(OrchestratorConfig::default());
        let state = vec![0.5, 0.3, 0.7, 0.2];

        orchestrator.process_state(state.clone()).await.unwrap();
        let stats = orchestrator.get_statistics().await.unwrap();

        assert_eq!(stats.total_states_processed, 1);
        assert_eq!(stats.states_in_buffer, 1);
    }

    #[tokio::test]
    async fn test_reset() {
        let orchestrator = UnifiedOrchestrator::new(OrchestratorConfig::default());
        let state = vec![0.5, 0.3, 0.7, 0.2];

        orchestrator.process_state(state).await.unwrap();
        orchestrator.reset().await.unwrap();

        let stats = orchestrator.get_statistics().await.unwrap();
        assert_eq!(stats.total_states_processed, 0);
        assert_eq!(stats.states_in_buffer, 0);
    }
}
