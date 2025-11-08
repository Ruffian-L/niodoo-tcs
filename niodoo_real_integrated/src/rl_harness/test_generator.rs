//! Test case generation for RL Execution Harness

use crate::config::CodeLanguage;
use crate::generation::GenerationEngine;
use anyhow::{Context, Result};
use std::sync::Arc;
use tracing::info;

/// Test generator that creates unit tests from problem descriptions
pub struct TestGenerator {
    generation_engine: Option<Arc<GenerationEngine>>,
}

impl TestGenerator {
    /// Create a new test generator
    pub fn new() -> Self {
        Self {
            generation_engine: None,
        }
    }

    /// Create a test generator with LLM-based test generation
    pub fn with_generation_engine(generation_engine: Arc<GenerationEngine>) -> Self {
        Self {
            generation_engine: Some(generation_engine),
        }
    }

    /// Generate unit tests from problem description
    pub async fn generate_tests(
        &self,
        problem_description: &str,
        language: CodeLanguage,
    ) -> Result<Vec<String>> {
        // If generation engine is available, use LLM to generate tests
        if let Some(ref engine) = self.generation_engine {
            return self.generate_tests_with_llm(problem_description, language, engine).await;
        }

        // Otherwise, use template-based generation
        self.generate_tests_template(problem_description, language)
    }

    /// Generate tests using LLM
    async fn generate_tests_with_llm(
        &self,
        problem_description: &str,
        language: CodeLanguage,
        engine: &GenerationEngine,
    ) -> Result<Vec<String>> {
        let lang_str = match language {
            CodeLanguage::Python => "Python",
            CodeLanguage::TypeScript => "TypeScript",
        };

        let prompt = format!(
            "Problem: {}\n\nGenerate 3-5 unit test cases in {} that verify the solution. \
            Return only the test code, one test per line, using assert statements.",
            problem_description, lang_str
        );

        // Use generation engine to generate test code
        // Note: This is a simplified approach. In production, you might want to
        // use a dedicated test generation prompt or fine-tuned model.
        let test_code = engine.generate_code(&prompt, language)
            .await
            .context("Failed to generate tests with LLM")?
            .code;

        // Parse test code into individual test cases
        let test_cases = self.parse_test_code(&test_code, language);

        info!(
            test_count = test_cases.len(),
            "Generated tests using LLM"
        );

        Ok(test_cases)
    }

    /// Generate tests using templates (fallback)
    fn generate_tests_template(
        &self,
        _problem_description: &str,
        language: CodeLanguage,
    ) -> Result<Vec<String>> {
        // Simple template-based test generation
        // This is a placeholder - in production, you'd have more sophisticated templates
        match language {
            CodeLanguage::Python => {
                Ok(vec![
                    "# Basic test case".to_string(),
                    "assert True  # Placeholder test".to_string(),
                ])
            }
            CodeLanguage::TypeScript => {
                Ok(vec![
                    "// Basic test case".to_string(),
                    "if (true !== true) throw new Error('Test failed');".to_string(),
                ])
            }
        }
    }

    /// Parse generated test code into individual test cases
    fn parse_test_code(&self, test_code: &str, language: CodeLanguage) -> Vec<String> {
        match language {
            CodeLanguage::Python => {
                // Split by lines and filter for assert statements
                test_code
                    .lines()
                    .filter(|line| line.trim().starts_with("assert"))
                    .map(|s| s.trim().to_string())
                    .collect()
            }
            CodeLanguage::TypeScript => {
                // Split by lines and filter for test-like statements
                test_code
                    .lines()
                    .filter(|line| {
                        let trimmed = line.trim();
                        trimmed.contains("assert") || trimmed.contains("expect") || trimmed.contains("if")
                    })
                    .map(|s| s.trim().to_string())
                    .collect()
            }
        }
    }
}

impl Default for TestGenerator {
    fn default() -> Self {
        Self::new()
    }
}

