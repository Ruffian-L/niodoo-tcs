//! DocPuzzle Benchmark Integration
//!
//! Multi-step reasoning benchmark with checklist-guided process analysis.
//! Validates process-aware reasoning and structured problem-solving.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocPuzzleTask {
    pub task_id: String,
    pub document: String,
    pub question: String,
    pub required_steps: Vec<String>,
    pub expected_answer: Option<String>,
    pub expected_keywords: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocPuzzleResult {
    pub task_id: String,
    pub answer: String,
    pub steps_taken: Vec<String>,
    pub answer_correctness: f64,
    pub process_score: f64,
    pub checklist_compliance: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocPuzzleBenchmarkResults {
    pub timestamp: String,
    pub results: Vec<DocPuzzleResult>,
    pub avg_process_score: f64,
    pub avg_answer_correctness: f64,
    pub avg_checklist_compliance: f64,
}

pub struct DocPuzzleRunner;

impl DocPuzzleRunner {
    pub fn new() -> Self {
        Self
    }

    /// Load DocPuzzle tasks from JSON file
    pub fn load_tasks<P: AsRef<std::path::Path>>(path: P) -> Result<Vec<DocPuzzleTask>, anyhow::Error> {
        let content = std::fs::read_to_string(path)?;
        let tasks: Vec<DocPuzzleTask> = serde_json::from_str(&content)?;
        Ok(tasks)
    }

    /// Run a single DocPuzzle task
    pub async fn run_task<F>(
        &self,
        task: &DocPuzzleTask,
        solve_fn: F,
    ) -> Result<DocPuzzleResult, anyhow::Error>
    where
        F: Fn(&str) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<String, anyhow::Error>> + Send>>,
    {
        let prompt = self.build_task_prompt(task);
        let response = solve_fn(&prompt).await?;

        // Parse response to extract answer and steps
        let (answer, steps) = self.parse_response(&response);

        // Evaluate
        let answer_correctness = self.evaluate_answer(task, &answer);
        let checklist_compliance = self.evaluate_checklist(task, &steps);
        let process_score = (answer_correctness + checklist_compliance) / 2.0;

        Ok(DocPuzzleResult {
            task_id: task.task_id.clone(),
            answer,
            steps_taken: steps,
            answer_correctness,
            process_score,
            checklist_compliance,
        })
    }

    fn build_task_prompt(&self, task: &DocPuzzleTask) -> String {
        format!(
            "Document:\n{}\n\nQuestion: {}\n\nRequired Steps:\n{}\n\nAnswer this question step-by-step, following the checklist above.",
            task.document,
            task.question,
            task.required_steps.iter().enumerate().map(|(i, step)| format!("{}. {}", i + 1, step)).collect::<Vec<_>>().join("\n")
        )
    }

    fn parse_response(&self, response: &str) -> (String, Vec<String>) {
        // Simple parsing: look for numbered steps and final answer
        let lines: Vec<&str> = response.lines().collect();
        let mut steps = Vec::new();
        let mut answer = String::new();

        let mut in_steps = false;
        for line in &lines {
            let line_lower = line.to_lowercase();
            if line_lower.contains("step") || line_lower.contains("1.") || line_lower.contains("first") {
                in_steps = true;
            }
            if in_steps && (line.trim().starts_with(|c: char| c.is_numeric()) || line.trim().starts_with("-")) {
                steps.push(line.trim().to_string());
            }
            if line_lower.contains("answer") || line_lower.contains("conclusion") {
                answer = line.to_string();
            }
        }

        // If no structured answer found, use last sentence
        if answer.is_empty() {
            answer = lines.last().unwrap_or(&"").to_string();
        }

        (answer, steps)
    }

    fn evaluate_answer(&self, task: &DocPuzzleTask, answer: &str) -> f64 {
        let answer_lower = answer.to_lowercase();
        
        // Check exact match
        if let Some(ref expected) = task.expected_answer {
            if answer_lower.contains(&expected.to_lowercase()) {
                return 1.0;
            }
        }

        // Check keyword matching
        if !task.expected_keywords.is_empty() {
            let matches = task.expected_keywords.iter()
                .filter(|kw| answer_lower.contains(&kw.to_lowercase()))
                .count();
            matches as f64 / task.expected_keywords.len() as f64
        } else {
            0.5 // Partial credit if no expected keywords
        }
    }

    fn evaluate_checklist(&self, task: &DocPuzzleTask, steps: &[String]) -> f64 {
        if task.required_steps.is_empty() {
            return 1.0;
        }

        let steps_text = steps.join(" ").to_lowercase();
        let matches = task.required_steps.iter()
            .filter(|required| {
                // Check if required step concept is present
                let required_lower = required.to_lowercase();
                required_lower.split_whitespace().any(|word| {
                    word.len() > 3 && steps_text.contains(word)
                })
            })
            .count();

        matches as f64 / task.required_steps.len() as f64
    }

    /// Run full benchmark suite
    pub async fn run_benchmark<F>(
        &self,
        tasks: &[DocPuzzleTask],
        solve_fn: F,
    ) -> Result<DocPuzzleBenchmarkResults, anyhow::Error>
    where
        F: Fn(&str) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<String, anyhow::Error>> + Send>>,
    {
        let mut results = Vec::new();

        for task in tasks {
            let result = self.run_task(task, &solve_fn).await?;
            results.push(result);
        }

        let avg_process_score = results.iter().map(|r| r.process_score).sum::<f64>() / results.len() as f64;
        let avg_answer_correctness = results.iter().map(|r| r.answer_correctness).sum::<f64>() / results.len() as f64;
        let avg_checklist_compliance = results.iter().map(|r| r.checklist_compliance).sum::<f64>() / results.len() as f64;

        Ok(DocPuzzleBenchmarkResults {
            timestamp: chrono::Utc::now().to_rfc3339(),
            results,
            avg_process_score,
            avg_answer_correctness,
            avg_checklist_compliance,
        })
    }
}

