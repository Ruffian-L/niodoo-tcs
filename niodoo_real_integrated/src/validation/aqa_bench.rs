//! AQA-Bench Integration
//!
//! Algorithmic Question Answering benchmark for validating:
//! - Interactive sequential reasoning (DFS/BFS tasks)
//! - Multi-step problem solving
//! - State-space exploration

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AQATask {
    pub task_id: String,
    pub task_type: TaskType,
    pub problem_description: String,
    pub initial_state: String,
    pub goal_state: String,
    pub allowed_actions: Vec<String>,
    pub solution_path: Option<Vec<String>>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TaskType {
    DFS,
    BFS,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AQAResult {
    pub task_id: String,
    pub task_type: TaskType,
    pub solution_path: Vec<String>,
    pub success: bool,
    pub steps_taken: usize,
    pub optimal_steps: Option<usize>,
    pub efficiency_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AQABenchmarkResults {
    pub timestamp: String,
    pub results: Vec<AQAResult>,
    pub dfs_success_rate: f64,
    pub bfs_success_rate: f64,
    pub overall_success_rate: f64,
    pub avg_efficiency_score: f64,
}

pub struct AQARunner {
    // Will be populated with test cases
}

impl AQARunner {
    pub fn new() -> Self {
        Self {}
    }

    /// Load AQA test cases from JSON file
    pub fn load_tasks<P: AsRef<std::path::Path>>(path: P) -> Result<Vec<AQATask>, anyhow::Error> {
        let content = std::fs::read_to_string(path)?;
        let tasks: Vec<AQATask> = serde_json::from_str(&content)?;
        Ok(tasks)
    }

    /// Run a single AQA task
    pub async fn run_task<F>(
        &self,
        task: &AQATask,
        solve_fn: F,
    ) -> Result<AQAResult, anyhow::Error>
    where
        F: Fn(&str, &AQATask) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<String>, anyhow::Error>> + Send>>,
    {
        let prompt = self.build_task_prompt(task);
        let solution_path = solve_fn(&prompt, task).await?;
        let steps_taken = solution_path.len();

        let success = self.evaluate_solution(task, &solution_path);
        let optimal_steps = task.solution_path.as_ref().map(|s| s.len());
        let efficiency_score = if let Some(optimal) = optimal_steps {
            if steps_taken > 0 {
                optimal as f64 / steps_taken as f64
            } else {
                0.0
            }
        } else {
            if steps_taken > 0 {
                1.0 / steps_taken as f64
            } else {
                0.0
            }
        };

        Ok(AQAResult {
            task_id: task.task_id.clone(),
            task_type: task.task_type,
            solution_path,
            success,
            steps_taken,
            optimal_steps,
            efficiency_score,
        })
    }

    fn build_task_prompt(&self, task: &AQATask) -> String {
        format!(
            "Problem: {}\n\nInitial State: {}\nGoal State: {}\nAllowed Actions: {}\n\nSolve this step-by-step using {} algorithm. Show each step.",
            task.problem_description,
            task.initial_state,
            task.goal_state,
            task.allowed_actions.join(", "),
            match task.task_type {
                TaskType::DFS => "DFS",
                TaskType::BFS => "BFS",
            }
        )
    }

    fn evaluate_solution(&self, task: &AQATask, solution_path: &[String]) -> bool {
        // Check if solution reaches goal state
        if solution_path.is_empty() {
            return false;
        }

        // Simple check: verify last step reaches goal (or solution matches expected)
        if let Some(ref expected) = task.solution_path {
            if solution_path.len() == expected.len() {
                return solution_path == expected;
            }
            // Check if solution contains goal state
            // Safety: solution_path is checked for empty above, so last() is guaranteed Some
            let last_step = solution_path
                .last()
                .unwrap_or_else(|| {
                    panic!("solution_path should not be empty at this point - this indicates a logic error");
                });
            last_step.contains(&task.goal_state)
        } else {
            // No expected solution, check if goal is reached
            // Safety: solution_path is checked for empty above, so last() is guaranteed Some
            let last_step = solution_path
                .last()
                .unwrap_or_else(|| {
                    panic!("solution_path should not be empty at this point - this indicates a logic error");
                });
            last_step.contains(&task.goal_state)
        }
    }

    /// Run full benchmark suite
    pub async fn run_benchmark<F>(
        &self,
        tasks: &[AQATask],
        solve_fn: F,
    ) -> Result<AQABenchmarkResults, anyhow::Error>
    where
        F: Fn(&str, &AQATask) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<String>, anyhow::Error>> + Send>>,
    {
        let mut results = Vec::new();
        let mut dfs_successes = 0;
        let mut dfs_total = 0;
        let mut bfs_successes = 0;
        let mut bfs_total = 0;

        for task in tasks {
            let result = self.run_task(task, &solve_fn).await?;
            
            match result.task_type {
                TaskType::DFS => {
                    dfs_total += 1;
                    if result.success {
                        dfs_successes += 1;
                    }
                }
                TaskType::BFS => {
                    bfs_total += 1;
                    if result.success {
                        bfs_successes += 1;
                    }
                }
            }

            results.push(result);
        }

        let dfs_success_rate = if dfs_total > 0 {
            dfs_successes as f64 / dfs_total as f64
        } else {
            0.0
        };

        let bfs_success_rate = if bfs_total > 0 {
            bfs_successes as f64 / bfs_total as f64
        } else {
            0.0
        };

        let overall_success_rate = results.iter().filter(|r| r.success).count() as f64 / results.len() as f64;
        let avg_efficiency_score = results.iter().map(|r| r.efficiency_score).sum::<f64>() / results.len() as f64;

        Ok(AQABenchmarkResults {
            timestamp: chrono::Utc::now().to_rfc3339(),
            results,
            dfs_success_rate,
            bfs_success_rate,
            overall_success_rate,
            avg_efficiency_score,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_prompt() {
        let task = AQATask {
            task_id: "test-1".to_string(),
            task_type: TaskType::DFS,
            problem_description: "Find path from A to B".to_string(),
            initial_state: "A".to_string(),
            goal_state: "B".to_string(),
            allowed_actions: vec!["move".to_string()],
            solution_path: None,
        };

        let runner = AQARunner::new();
        let prompt = runner.build_task_prompt(&task);
        assert!(prompt.contains("DFS"));
        assert!(prompt.contains("A"));
        assert!(prompt.contains("B"));
    }
}

