//! Code Quality Score (CQS) Calculator
//!
//! Computes Code Quality Score using rust-code-analysis for cyclomatic and cognitive complexity.
//! CQS = w₁·(Cyclomatic Complexity) + w₂·(Cognitive Complexity) + w₃·(Git Churn)

use crate::config::CodeLanguage;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// CQS weights configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CQSWeights {
    pub w_cc: f64,   // Weight for cyclomatic complexity
    pub w_cog: f64,  // Weight for cognitive complexity
    pub w_churn: f64, // Weight for git churn
}

impl Default for CQSWeights {
    fn default() -> Self {
        // Empirical starting point: 0.4/0.4/0.2 (cc/cog/churn)
        // Lower weight on churn as it's a lagging indicator
        Self {
            w_cc: 0.4,
            w_cog: 0.4,
            w_churn: 0.2,
        }
    }
}

impl CQSWeights {
    /// Normalize weights to sum to 1.0
    pub fn normalize(&self) -> Self {
        let total = self.w_cc + self.w_cog + self.w_churn;
        if total == 0.0 {
            return Self::default();
        }
        Self {
            w_cc: self.w_cc / total,
            w_cog: self.w_cog / total,
            w_churn: self.w_churn / total,
        }
    }
}

/// Code Quality Score result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CQS {
    pub score: f64,
    pub cyclomatic_complexity: u32,
    pub cognitive_complexity: u32,
    pub git_churn: u32,
}

/// CQS Calculator using rust-code-analysis
pub struct CQSCalculator {
    weights: CQSWeights,
}

impl CQSCalculator {
    /// Create a new CQS calculator with default weights
    pub fn new() -> Self {
        Self {
            weights: CQSWeights::default(),
        }
    }

    /// Create a new CQS calculator with custom weights
    pub fn with_weights(weights: CQSWeights) -> Self {
        Self {
            weights: weights.normalize(),
        }
    }

    /// Compute Code Quality Score for code
    /// 
    /// CQS = w₁·(Cyclomatic Complexity) + w₂·(Cognitive Complexity) + w₃·(Git Churn)
    /// 
    /// Lower CQS = better code quality
    pub fn compute_cqs(&self, code: &str, language: CodeLanguage, git_churn: u32) -> Result<CQS> {
        // Compute cyclomatic complexity
        let cyclomatic_complexity = self.compute_cyclomatic_complexity(code, language)?;

        // Compute cognitive complexity
        let cognitive_complexity = self.compute_cognitive_complexity(code, language)?;

        // Compute CQS score
        let score = self.weights.w_cc * cyclomatic_complexity as f64
            + self.weights.w_cog * cognitive_complexity as f64
            + self.weights.w_churn * git_churn as f64;

        Ok(CQS {
            score,
            cyclomatic_complexity,
            cognitive_complexity,
            git_churn,
        })
    }

    /// Compute cyclomatic complexity using rust-code-analysis
    /// 
    /// TODO: Integrate rust-code-analysis crate when available
    /// For now, use heuristic-based calculation
    fn compute_cyclomatic_complexity(&self, code: &str, language: CodeLanguage) -> Result<u32> {
        match language {
            CodeLanguage::Python => self.compute_cyclomatic_python(code),
            CodeLanguage::TypeScript => self.compute_cyclomatic_typescript(code),
        }
    }

    /// Compute cyclomatic complexity for Python code (heuristic)
    fn compute_cyclomatic_python(&self, code: &str) -> Result<u32> {
        let mut complexity = 1; // Base complexity

        // Count decision points
        for line in code.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("if ") || trimmed.starts_with("elif ") {
                complexity += 1;
            }
            if trimmed.starts_with("for ") || trimmed.starts_with("while ") {
                complexity += 1;
            }
            if trimmed.contains("and") || trimmed.contains("or") {
                complexity += 1;
            }
            if trimmed.contains("except") || trimmed.contains("case") {
                complexity += 1;
            }
        }

        Ok(complexity.max(1))
    }

    /// Compute cyclomatic complexity for TypeScript code (heuristic)
    fn compute_cyclomatic_typescript(&self, code: &str) -> Result<u32> {
        let mut complexity = 1; // Base complexity

        // Count decision points
        for line in code.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("if ") || trimmed.starts_with("else if ") {
                complexity += 1;
            }
            if trimmed.starts_with("for ") || trimmed.starts_with("while ") {
                complexity += 1;
            }
            if trimmed.contains("&&") || trimmed.contains("||") {
                complexity += 1;
            }
            if trimmed.contains("case ") || trimmed.contains("catch ") {
                complexity += 1;
            }
        }

        Ok(complexity.max(1))
    }

    /// Compute cognitive complexity
    /// 
    /// TODO: Integrate rust-code-analysis for accurate cognitive complexity
    /// For now, use heuristic: cognitive = cyclomatic + nesting penalty
    fn compute_cognitive_complexity(&self, code: &str, language: CodeLanguage) -> Result<u32> {
        let cyclomatic = self.compute_cyclomatic_complexity(code, language)?;
        
        // Compute nesting depth
        let nesting_depth = self.compute_nesting_depth(code, language)?;
        
        // Cognitive complexity = cyclomatic + nesting penalty
        let cognitive = cyclomatic + (nesting_depth as f64 * 0.5) as u32;
        
        Ok(cognitive.max(1))
    }

    /// Compute nesting depth (heuristic)
    fn compute_nesting_depth(&self, code: &str, _language: CodeLanguage) -> Result<u32> {
        let mut max_depth = 0;
        let mut current_depth: i32 = 0;

        for line in code.lines() {
            let trimmed = line.trim();
            
            // Increase depth for opening blocks
            if trimmed.ends_with("{") || trimmed.ends_with(":") {
                current_depth += 1;
                max_depth = max_depth.max(current_depth);
            }
            
            // Decrease depth for closing blocks
            if trimmed.starts_with("}") || (trimmed.is_empty() && current_depth > 0) {
                current_depth = current_depth.saturating_sub(1);
            }
        }

        Ok(max_depth.max(0) as u32)
    }
}

impl Default for CQSCalculator {
    fn default() -> Self {
        Self::new()
    }
}

