//! CounterBench Integration
//!
//! Counterfactual reasoning benchmark for validating:
//! - What-if scenario reasoning
//! - Alternative outcome prediction
//! - Causal reasoning

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CounterBenchTask {
    pub task_id: String,
    pub scenario: String,
    pub factual_outcome: String,
    pub counterfactual_condition: String,
    pub question: String,
    pub expected_answer: Option<String>,
    pub expected_keywords: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CounterBenchResult {
    pub task_id: String,
    pub answer: String,
    pub accuracy: f64,
    pub keyword_match_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CounterBenchmarkResults {
    pub timestamp: String,
    pub results: Vec<CounterBenchResult>,
    pub avg_accuracy: f64,
    pub avg_keyword_match_rate: f64,
}

pub struct CounterBenchRunner;

impl CounterBenchRunner {
    pub fn new() -> Self {
        Self
    }

    /// Load CounterBench tasks from JSON file
    pub fn load_tasks<P: AsRef<std::path::Path>>(
        path: P,
    ) -> Result<Vec<CounterBenchTask>, anyhow::Error> {
        let content = std::fs::read_to_string(path)?;
        let tasks: Vec<CounterBenchTask> = serde_json::from_str(&content)?;
        Ok(tasks)
    }

    /// Run a single CounterBench task
    pub async fn run_task<F>(
        &self,
        task: &CounterBenchTask,
        solve_fn: F,
    ) -> Result<CounterBenchResult, anyhow::Error>
    where
        F: Fn(
            &str,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = Result<String, anyhow::Error>> + Send>,
        >,
    {
        let prompt = self.build_task_prompt(task);
        let answer = solve_fn(&prompt).await?;

        let accuracy = self.evaluate_answer(task, &answer);
        let keyword_match_rate = self.evaluate_keywords(task, &answer);

        Ok(CounterBenchResult {
            task_id: task.task_id.clone(),
            answer,
            accuracy,
            keyword_match_rate,
        })
    }

    fn build_task_prompt(&self, task: &CounterBenchTask) -> String {
        format!(
            "Scenario: {}\n\nFactual Outcome: {}\n\nCounterfactual Condition: {}\n\nQuestion: {}\n\nAnswer:",
            task.scenario,
            task.factual_outcome,
            task.counterfactual_condition,
            task.question
        )
    }

    fn evaluate_answer(&self, task: &CounterBenchTask, answer: &str) -> f64 {
        let answer_lower = answer.to_lowercase();

        // Check exact match
        if let Some(ref expected) = task.expected_answer {
            if answer_lower.contains(&expected.to_lowercase()) {
                return 1.0;
            }
        }

        // Check keyword matching
        if !task.expected_keywords.is_empty() {
            let matches = task
                .expected_keywords
                .iter()
                .filter(|kw| answer_lower.contains(&kw.to_lowercase()))
                .count();
            matches as f64 / task.expected_keywords.len() as f64
        } else {
            0.5 // Partial credit
        }
    }

    fn evaluate_keywords(&self, task: &CounterBenchTask, answer: &str) -> f64 {
        if task.expected_keywords.is_empty() {
            return 1.0;
        }

        let answer_lower = answer.to_lowercase();
        let matches = task
            .expected_keywords
            .iter()
            .filter(|kw| answer_lower.contains(&kw.to_lowercase()))
            .count();

        matches as f64 / task.expected_keywords.len() as f64
    }

    /// Run full benchmark suite
    pub async fn run_benchmark<F>(
        &self,
        tasks: &[CounterBenchTask],
        solve_fn: F,
    ) -> Result<CounterBenchmarkResults, anyhow::Error>
    where
        F: Fn(
            &str,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = Result<String, anyhow::Error>> + Send>,
        >,
    {
        let mut results = Vec::new();

        for task in tasks {
            let result = self.run_task(task, &solve_fn).await?;
            results.push(result);
        }

        let avg_accuracy = results.iter().map(|r| r.accuracy).sum::<f64>() / results.len() as f64;
        let avg_keyword_match_rate =
            results.iter().map(|r| r.keyword_match_rate).sum::<f64>() / results.len() as f64;

        Ok(CounterBenchmarkResults {
            timestamp: chrono::Utc::now().to_rfc3339(),
            results,
            avg_accuracy,
            avg_keyword_match_rate,
        })
    }
}
