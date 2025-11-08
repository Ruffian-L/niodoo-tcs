//! Fused Cognitive Architecture: Slow/Fast Agent Integration
//!
//! This module implements the dual-loop "Slow/Fast" agent paradigm:
//! - Slow Agent (TCS): Affective-cognitive system modeling user states via computational topology
//! - Fast Agent (GenerationEngine): Code-builder LLM modulated by Slow Agent's strategic imperative
//!
//! The fused loop enables topology-aware, affectively-modulated code generation.

use crate::compass::CompassOutcome;
use crate::config::CodeLanguage;
use crate::generation::{CodeGenerationResult, GenerationEngine};
use crate::tcs_analysis::TCSAnalyzer;
use crate::torus::PadGhostState;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{info, warn};

/// TCS strategic command for modulating code generation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TCSStrategy {
    /// Stabilize: Generate low-complexity code (CQS < 5)
    /// Used when user is stuck and needs simple, reliable solutions
    Stabilize,
    
    /// Explore: Allow higher complexity (CQS < 12)
    /// Used when user is exploring new approaches
    Explore,
    
    /// Optimize: Moderate complexity (CQS < 8)
    /// Used for performance-critical code
    Optimize,
    
    /// Refactor: Moderate-high complexity (CQS < 10)
    /// Used for restructuring existing code
    Refactor,
}

impl TCSStrategy {
    /// Get CQS threshold for this strategy
    pub fn cqs_threshold(&self) -> f64 {
        match self {
            TCSStrategy::Stabilize => 5.0,
            TCSStrategy::Explore => 12.0,
            TCSStrategy::Optimize => 8.0,
            TCSStrategy::Refactor => 10.0,
        }
    }
    
    /// Convert to string for Python API
    pub fn as_str(&self) -> &'static str {
        match self {
            TCSStrategy::Stabilize => "STABILIZE",
            TCSStrategy::Explore => "EXPLORE",
            TCSStrategy::Optimize => "OPTIMIZE",
            TCSStrategy::Refactor => "REFACTOR",
        }
    }
    
    /// Parse from string (for Python API)
    pub fn from_str(s: &str) -> Result<Self> {
        match s.to_uppercase().as_str() {
            "STABILIZE" => Ok(TCSStrategy::Stabilize),
            "EXPLORE" => Ok(TCSStrategy::Explore),
            "OPTIMIZE" => Ok(TCSStrategy::Optimize),
            "REFACTOR" => Ok(TCSStrategy::Refactor),
            _ => Err(anyhow::anyhow!("Unknown TCS strategy: {}", s)),
        }
    }
}

/// Map Compass strategic actions to TCS strategies
impl From<&CompassOutcome> for TCSStrategy {
    fn from(compass: &CompassOutcome) -> Self {
        // Map compass quadrant to TCS strategy
        // Panic/Persist = Stabilize (user is stuck, needs simple solutions)
        // Discover = Explore (user is learning, allow exploration)
        // Master = Optimize (user is confident, optimize performance)
        match compass.quadrant {
            crate::compass::CompassQuadrant::Panic => TCSStrategy::Stabilize,
            crate::compass::CompassQuadrant::Persist => TCSStrategy::Stabilize,
            crate::compass::CompassQuadrant::Discover => TCSStrategy::Explore,
            crate::compass::CompassQuadrant::Master => {
                // If in late cascade stage, optimize; otherwise refactor
                match compass.cascade_stage {
                    Some(stage) if matches!(stage, crate::compass::CascadeStage::Calm | crate::compass::CascadeStage::Motivation) => {
                        TCSStrategy::Optimize
                    }
                    _ => TCSStrategy::Refactor,
                }
            }
        }
    }
}

/// Fused agent that combines Slow Agent (TCS) with Fast Agent (GenerationEngine)
pub struct FusedAgent {
    /// Fast Agent: Code generation engine
    generation_engine: Arc<GenerationEngine>,
    
    /// Slow Agent: TCS analyzer for topological analysis
    tcs_analyzer: Option<Arc<Mutex<TCSAnalyzer>>>,
    
    /// Current TCS strategy (modulated by Slow Agent)
    current_strategy: Arc<tokio::sync::RwLock<TCSStrategy>>,
}

impl FusedAgent {
    /// Create a new fused agent
    pub fn new(
        generation_engine: Arc<GenerationEngine>,
        tcs_analyzer: Option<Arc<Mutex<TCSAnalyzer>>>,
    ) -> Self {
        Self {
            generation_engine,
            tcs_analyzer,
            current_strategy: Arc::new(tokio::sync::RwLock::new(TCSStrategy::Stabilize)),
        }
    }
    
    /// Update TCS strategy based on compass outcome (Slow Agent decision)
    pub async fn update_strategy_from_compass(&self, compass: &CompassOutcome) -> Result<()> {
        let strategy = TCSStrategy::from(compass);
        let mut current = self.current_strategy.write().await;
        *current = strategy;
        info!(
            strategy = %strategy.as_str(),
            quadrant = ?compass.quadrant,
            "Updated TCS strategy from compass"
        );
        Ok(())
    }
    
    /// Generate code with TCS strategy modulation (Fused Loop)
    ///
    /// This is the core fused loop:
    /// 1. Slow Agent (TCS) analyzes user state and determines strategy
    /// 2. Fast Agent (GenerationEngine) generates code modulated by strategy
    /// 3. Strategy is encoded in the prompt/system message to guide generation
    pub async fn generate_code_with_strategy(
        &self,
        goal: &str,
        language: CodeLanguage,
        pad_state: Option<&PadGhostState>,
    ) -> Result<FusedGenerationResult> {
        // Get current strategy from Slow Agent
        let strategy = {
            let current = self.current_strategy.read().await;
            *current
        };
        
        // Analyze topology if TCS analyzer is available (Slow Agent analysis)
        let topological_signature = if let Some(ref analyzer) = self.tcs_analyzer {
            if let Some(pad) = pad_state {
                let mut analyzer_guard = analyzer.lock().await;
                match analyzer_guard.analyze_state(pad) {
                    Ok(sig) => {
                        info!(
                            betti_0 = sig.betti_numbers[0],
                            betti_1 = sig.betti_numbers[1],
                            persistence_entropy = sig.persistence_entropy,
                            "Computed topological signature"
                        );
                        Some(sig)
                    }
                    Err(e) => {
                        warn!(%e, "Failed to compute topological signature");
                        None
                    }
                }
            } else {
                None
            }
        } else {
            None
        };
        
        // Build strategy-modulated prompt
        let strategy_prompt = self.build_strategy_prompt(goal, &strategy, &topological_signature);
        
        // Generate code using Fast Agent (GenerationEngine)
        let code_result = self.generation_engine
            .generate_code(&strategy_prompt, language)
            .await
            .context("Failed to generate code")?;
        
        info!(
            strategy = %strategy.as_str(),
            language = ?language,
            latency_ms = code_result.latency_ms,
            "Generated code with TCS strategy modulation"
        );
        
        Ok(FusedGenerationResult {
            code: code_result.code,
            language: code_result.language,
            strategy,
            topological_signature,
            latency_ms: code_result.latency_ms,
            failure_type: code_result.failure_type,
            failure_details: code_result.failure_details,
        })
    }
    
    /// Build prompt modulated by TCS strategy
    fn build_strategy_prompt(
        &self,
        goal: &str,
        strategy: &TCSStrategy,
        topology: &Option<crate::tcs_analysis::TopologicalSignature>,
    ) -> String {
        let strategy_instruction = match strategy {
            TCSStrategy::Stabilize => {
                "Generate simple, straightforward code with low complexity. \
                 Prioritize clarity and reliability over cleverness. \
                 Use well-established patterns and avoid unnecessary abstractions."
            }
            TCSStrategy::Explore => {
                "Generate exploratory code that can handle novel approaches. \
                 You may use more advanced patterns and abstractions. \
                 Focus on flexibility and extensibility."
            }
            TCSStrategy::Optimize => {
                "Generate optimized code for performance. \
                 Balance complexity with efficiency. \
                 Use efficient algorithms and data structures."
            }
            TCSStrategy::Refactor => {
                "Generate refactored code that improves structure while maintaining functionality. \
                 Focus on maintainability and code organization. \
                 Use appropriate design patterns."
            }
        };
        
        let mut prompt = format!(
            "Goal: {}\n\nStrategy: {}\n\n{}\n\n",
            goal,
            strategy.as_str(),
            strategy_instruction
        );
        
        // Add topological context if available
        if let Some(ref topo) = topology {
            prompt.push_str(&format!(
                "Topological Context:\n\
                 - Betti numbers: H0={}, H1={}, H2={}\n\
                 - Persistence entropy: {:.3}\n\
                 - Knot complexity: {:.3}\n\n",
                topo.betti_numbers[0],
                topo.betti_numbers[1],
                topo.betti_numbers[2],
                topo.persistence_entropy,
                topo.knot_complexity
            ));
        }
        
        prompt.push_str("Write code to accomplish the goal following the strategy guidelines.");
        
        prompt
    }
    
    /// Get current TCS strategy
    pub async fn get_current_strategy(&self) -> TCSStrategy {
        let current = self.current_strategy.read().await;
        *current
    }
    
    /// Set TCS strategy explicitly (for testing or manual override)
    pub async fn set_strategy(&self, strategy: TCSStrategy) {
        let mut current = self.current_strategy.write().await;
        *current = strategy;
        info!(strategy = %strategy.as_str(), "Manually set TCS strategy");
    }
}

/// Result of fused code generation
#[derive(Debug, Clone)]
pub struct FusedGenerationResult {
    pub code: String,
    pub language: CodeLanguage,
    pub strategy: TCSStrategy,
    pub topological_signature: Option<crate::tcs_analysis::TopologicalSignature>,
    pub latency_ms: f64,
    pub failure_type: Option<String>,
    pub failure_details: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_strategy_cqs_thresholds() {
        assert_eq!(TCSStrategy::Stabilize.cqs_threshold(), 5.0);
        assert_eq!(TCSStrategy::Explore.cqs_threshold(), 12.0);
        assert_eq!(TCSStrategy::Optimize.cqs_threshold(), 8.0);
        assert_eq!(TCSStrategy::Refactor.cqs_threshold(), 10.0);
    }
    
    #[test]
    fn test_strategy_from_str() {
        assert_eq!(TCSStrategy::from_str("STABILIZE").unwrap(), TCSStrategy::Stabilize);
        assert_eq!(TCSStrategy::from_str("explore").unwrap(), TCSStrategy::Explore);
        assert_eq!(TCSStrategy::from_str("OPTIMIZE").unwrap(), TCSStrategy::Optimize);
        assert_eq!(TCSStrategy::from_str("refactor").unwrap(), TCSStrategy::Refactor);
    }
}

