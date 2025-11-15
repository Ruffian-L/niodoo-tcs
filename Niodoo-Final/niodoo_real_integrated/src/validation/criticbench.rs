//! CriticBench Integration
//!
//! Generation, Critique, Correction (GQC) protocol benchmark for validating:
//! - Self-correction capabilities
//! - Meta-cognitive evaluation
//! - Iterative refinement

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriticBenchTask {
    pub task_id: String,
    pub prompt: String,
    pub expected_corrections: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriticBenchResult {
    pub task_id: String,
    pub generation: String,
    pub critique: String,
    pub correction: String,
    pub generation_score: f64,
    pub critique_score: f64,
    pub correction_score: f64,
    pub improvement_detected: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriticBenchmarkResults {
    pub timestamp: String,
    pub results: Vec<CriticBenchResult>,
    pub avg_generation_score: f64,
    pub avg_critique_score: f64,
    pub avg_correction_score: f64,
    pub improvement_rate: f64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum GQCPhase {
    Generation,
    Critique,
    Correction,
}

pub struct CriticBenchRunner;

impl CriticBenchRunner {
    pub fn new() -> Self {
        Self
    }

    /// Load CriticBench tasks from JSON file
    pub fn load_tasks<P: AsRef<std::path::Path>>(
        path: P,
    ) -> Result<Vec<CriticBenchTask>, anyhow::Error> {
        let content = std::fs::read_to_string(path)?;
        let tasks: Vec<CriticBenchTask> = serde_json::from_str(&content)?;
        Ok(tasks)
    }

    /// Run a single CriticBench task (GQC protocol)
    pub async fn run_task<F>(
        &self,
        task: &CriticBenchTask,
        generate_fn: F,
        critique_fn: F,
        correct_fn: F,
    ) -> Result<CriticBenchResult, anyhow::Error>
    where
        F: Fn(
            &str,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = Result<String, anyhow::Error>> + Send>,
        >,
    {
        // Phase 1: Generation
        let generation_prompt = format!("Generate a response to: {}", task.prompt);
        let generation = generate_fn(&generation_prompt).await?;

        // Phase 2: Critique
        let critique_prompt = format!(
            "Critique the following response for accuracy, clarity, and completeness:\n\nResponse: {}\n\nCritique:",
            generation
        );
        let critique = critique_fn(&critique_prompt).await?;

        // Phase 3: Correction
        let correction_prompt = format!(
            "Original prompt: {}\n\nOriginal response: {}\n\nCritique: {}\n\nCorrected response:",
            task.prompt, generation, critique
        );
        let correction = correct_fn(&correction_prompt).await?;

        // Evaluate each phase
        let generation_score = self.evaluate_generation(&generation);
        let critique_score = self.evaluate_critique(&generation, &critique);
        let correction_score = self.evaluate_correction(&generation, &correction, task);
        let improvement_detected = correction_score > generation_score;

        Ok(CriticBenchResult {
            task_id: task.task_id.clone(),
            generation,
            critique,
            correction,
            generation_score,
            critique_score,
            correction_score,
            improvement_detected,
        })
    }

    fn evaluate_generation(&self, generation: &str) -> f64 {
        // Simple heuristics: length, completeness, structure
        let length_score = (generation.len().min(500) as f64 / 500.0).min(1.0);
        let has_structure = generation.contains(".") || generation.contains("\n");
        let structure_score = if has_structure { 0.5 } else { 0.0 };

        length_score * 0.7 + structure_score * 0.3
    }

    fn evaluate_critique(&self, generation: &str, critique: &str) -> f64 {
        // Check if critique identifies issues
        let critique_lower = critique.to_lowercase();
        let has_issues = critique_lower.contains("issue")
            || critique_lower.contains("problem")
            || critique_lower.contains("error")
            || critique_lower.contains("improve");

        if !has_issues {
            return 0.3; // Low score if no issues identified
        }

        // Check if critique references generation
        let gen_keywords: Vec<&str> = generation.split_whitespace().take(5).collect();
        let references_gen = gen_keywords.iter().any(|kw| critique.contains(kw));

        if references_gen {
            0.8 + if critique.len() > 50 { 0.2 } else { 0.0 }
        } else {
            0.5
        }
    }

    fn evaluate_correction(&self, original: &str, correction: &str, task: &CriticBenchTask) -> f64 {
        // Check if correction improves on original
        let original_score = self.evaluate_generation(original);
        let correction_score = self.evaluate_generation(correction);

        // Check if expected corrections are present
        let correction_lower = correction.to_lowercase();
        let expected_matches = task
            .expected_corrections
            .iter()
            .filter(|exp| correction_lower.contains(&exp.to_lowercase()))
            .count();

        let expected_score = if !task.expected_corrections.is_empty() {
            expected_matches as f64 / task.expected_corrections.len() as f64
        } else {
            0.5
        };

        // Combined score: improvement + expected corrections
        let improvement_factor = if correction_score > original_score {
            (correction_score - original_score) * 0.5
        } else {
            0.0
        };

        (correction_score * 0.5 + expected_score * 0.3 + improvement_factor * 0.2).min(1.0)
    }

    /// Run full benchmark suite
    pub async fn run_benchmark<F>(
        &self,
        tasks: &[CriticBenchTask],
        generate_fn: F,
        critique_fn: F,
        correct_fn: F,
    ) -> Result<CriticBenchmarkResults, anyhow::Error>
    where
        F: Fn(
            &str,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = Result<String, anyhow::Error>> + Send>,
        >,
    {
        let mut results = Vec::new();

        for task in tasks {
            let result = self
                .run_task(task, &generate_fn, &critique_fn, &correct_fn)
                .await?;
            results.push(result);
        }

        let avg_generation_score =
            results.iter().map(|r| r.generation_score).sum::<f64>() / results.len() as f64;
        let avg_critique_score =
            results.iter().map(|r| r.critique_score).sum::<f64>() / results.len() as f64;
        let avg_correction_score =
            results.iter().map(|r| r.correction_score).sum::<f64>() / results.len() as f64;
        let improvement_rate =
            results.iter().filter(|r| r.improvement_detected).count() as f64 / results.len() as f64;

        Ok(CriticBenchmarkResults {
            timestamp: chrono::Utc::now().to_rfc3339(),
            results,
            avg_generation_score,
            avg_critique_score,
            avg_correction_score,
            improvement_rate,
        })
    }
}
