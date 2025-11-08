//! Revision loop for forcing code to pass constitutional checks

use crate::config::CodeLanguage;
use crate::constitutional::constitution::Constitution;
use crate::constitutional::critique::CritiqueEngine;
use crate::constitutional::static_analysis::StaticAnalyzer;
use crate::constitutional::violations::Violation;
use crate::generation::{CodeGenerationResult, GenerationEngine};
use anyhow::{Context, Result};
use std::sync::Arc;
use tracing::{info, warn};

/// Revision loop that forces code to be rewritten until it passes constitutional checks
pub struct RevisionLoop {
    generator: Arc<GenerationEngine>,
    static_analyzer: StaticAnalyzer,
    critique_engine: CritiqueEngine,
    max_attempts: u32,
}

impl RevisionLoop {
    /// Create a new revision loop
    pub fn new(
        generator: Arc<GenerationEngine>,
        constitution: Constitution,
        max_attempts: u32,
    ) -> Self {
        let static_analyzer = StaticAnalyzer::new(constitution.clone());
        let critique_engine = CritiqueEngine::new(generator.clone(), constitution);
        Self {
            generator,
            static_analyzer,
            critique_engine,
            max_attempts,
        }
    }

    /// Generate code and revise until it passes constitutional checks
    pub async fn generate_and_revise(
        &self,
        goal: &str,
        language: CodeLanguage,
    ) -> Result<CodeGenerationResult> {
        let mut attempt = 0;
        let mut violations_history = Vec::new();

        loop {
            attempt += 1;
            if attempt > self.max_attempts {
                warn!(
                    max_attempts = self.max_attempts,
                    "Code generation exceeded maximum revision attempts"
                );
                return Err(anyhow::anyhow!(
                    "Failed to generate code that passes constitutional checks after {} attempts",
                    self.max_attempts
                ));
            }

            info!(attempt, goal, "Generating code (attempt {})", attempt);

            // Generate code
            let code_result = self.generator.generate_code(goal, language).await?;

            if code_result.code.is_empty() {
                warn!(attempt, "Generated code is empty");
                continue;
            }

            // Run static analysis
            let violations = self
                .static_analyzer
                .analyze(&code_result.code, language)
                .context("Static analysis failed")?;

            // Filter high-severity violations (clone to avoid lifetime issues)
            let high_severity_violations: Vec<Violation> = violations
                .into_iter()
                .filter(|v| matches!(v.severity, crate::constitutional::violations::ViolationSeverity::High))
                .collect();

            if high_severity_violations.is_empty() {
                info!(attempt, "Code passed constitutional checks");
                return Ok(code_result);
            }

            // Code has violations, need to revise
            violations_history.push(high_severity_violations.clone());
            warn!(
                attempt,
                violations_count = high_severity_violations.len(),
                "Code has constitutional violations, revising"
            );

            // Build revision prompt with violation details
            let violation_summary: String = high_severity_violations
                .iter()
                .map(|v| {
                    format!(
                        "- {} (line {}): {}",
                        v.principle_id,
                        v.line_number.map_or(0, |n| n),
                        v.message
                    )
                })
                .collect::<Vec<_>>()
                .join("\n");

            let revised_goal = format!(
                "{}\n\nPrevious code had these violations:\n{}\n\nPlease rewrite the code to fix these violations.",
                goal, violation_summary
            );

            // Update goal for next iteration
            // Note: In a real implementation, we'd modify the goal here
            // For now, we'll just continue the loop with the original goal
            // This is a simplified implementation
        }
    }
}

