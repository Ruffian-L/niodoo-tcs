use anyhow::Result;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::time::Duration;
use tokio::time::{timeout, sleep};
use crate::curator::Curator;
use crate::memory_core::{Experience, MemoryCore};

/// Configuration for the Executor
#[derive(Debug, Clone)]
pub struct ExecutorConfig {
    pub vllm_endpoint: String,
    pub model_name: String,
    pub max_context_length: usize,
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub repetition_penalty: f32,
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        Self {
            vllm_endpoint: std::env::var("VLLM_ENDPOINT").unwrap_or_else(|_| "http://localhost:5001".to_string()),
            model_name: "Qwen2.5-Coder-7B-Instruct".to_string(),
            max_context_length: 8192,
            max_new_tokens: 2048,
            temperature: 0.7,
            top_p: 0.9,
            repetition_penalty: 1.1,
        }
    }
}

/// Task execution result
#[derive(Debug, Clone)]
pub struct TaskResult {
    pub output: String,
    pub success_score: f32,
    pub execution_time_ms: u64,
    pub error_message: Option<String>,
}

/// The Executor: Task execution engine
pub struct Executor {
    client: Client,
    config: ExecutorConfig,
}

impl Executor {
    /// Initialize the Executor with vLLM connection
    pub fn new(config: ExecutorConfig) -> Result<Self> {
        println!("Initializing Executor with vLLM endpoint: {}", config.vllm_endpoint);

        let client = Client::builder().timeout(Duration::from_secs(10)).build()?;

        println!("Executor initialized successfully");

        Ok(Self {
            client,
            config,
        })
    }



    /// Execute a task with optional memory retrieval
    pub async fn execute_task(
        &self,
        task_input: &str,
        task_type: &str,
        curator: Option<&Curator>,
        memory: Option<&MemoryCore>,
    ) -> Result<TaskResult> {
        let start_time = std::time::Instant::now();

        // Retrieve relevant context from memory if available
        let context = if let (Some(curator), Some(memory)) = (curator, memory) {
            self.retrieve_context(task_input, curator, memory).await?
        } else {
            String::new()
        };

        // Build the full prompt
        let prompt = self.build_prompt(task_input, task_type, &context);

        // Generate response
        let output = self.generate(&prompt).await?;

        // Evaluate success (simple heuristic for now)
        let success_score = self.evaluate_success(&output, task_type);

        let execution_time = start_time.elapsed().as_millis() as u64;

        Ok(TaskResult {
            output,
            success_score,
            execution_time_ms: execution_time,
            error_message: None,
        })
    }

    /// Retrieve relevant context from memory
    async fn retrieve_context(
        &self,
        task_input: &str,
        curator: &Curator,
        memory: &MemoryCore,
    ) -> Result<String> {
        // Embed the task input
        let query_embedding = curator.embed_text(task_input).await?;

        // Search for similar experiences
        let similar_experiences = memory.search_similar(&query_embedding, 5).await?;

        // Build context string from similar experiences
        let mut context_parts = Vec::new();
        for exp in similar_experiences {
            context_parts.push(format!(
                "Previous experience:\nInput: {}\nOutput: {}\nContext: {}",
                exp.input, exp.output, exp.context
            ));
        }

        Ok(context_parts.join("\n\n"))
    }

    /// Build the prompt for the model
    fn build_prompt(&self, task_input: &str, task_type: &str, context: &str) -> String {
        let system_prompt = match task_type {
            "code_generation" => "You are an expert programmer. Generate high-quality, correct code based on the user's request.",
            "code_analysis" => "You are a code reviewer. Analyze the provided code and give constructive feedback.",
            "debugging" => "You are a debugging expert. Help identify and fix bugs in the code.",
            "documentation" => "You are a technical writer. Create clear, comprehensive documentation.",
            _ => "You are a helpful AI assistant. Provide accurate and useful responses.",
        };

        if context.is_empty() {
            format!("System: {}\n\nUser: {}\n\nAssistant:", system_prompt, task_input)
        } else {
            format!("System: {}\n\nContext:\n{}\n\nUser: {}\n\nAssistant:",
                   system_prompt, context, task_input)
        }
    }

    /// Generate text using vLLM API with timeout and retry
    async fn generate(&self, prompt: &str) -> Result<String> {
        let request = json!({
            "model": self.config.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_new_tokens,
            "top_p": self.config.top_p
        });

        let max_retries = 3;
        let base_delay = Duration::from_millis(500);

        for attempt in 0..max_retries {
            let result = timeout(Duration::from_secs(30), async {
                self.client
                    .post(&format!("{}/v1/chat/completions", self.config.vllm_endpoint))
                    .json(&request)
                    .send()
                    .await?
                    .json::<Value>()
                    .await
            }).await;

            match result {
                Ok(Ok(response)) => {
                    let content = response["choices"][0]["message"]["content"]
                        .as_str()
                        .ok_or_else(|| anyhow::anyhow!("Invalid response format"))?;
                    return Ok(content.to_string());
                }
                Ok(Err(e)) => {
                    // Request succeeded but parsing failed, don't retry
                    return Err(e.into());
                }
                Err(_) => {
                    // Timeout occurred
                    if attempt < max_retries - 1 {
                        let delay = base_delay * 2_u32.pow(attempt);
                        sleep(delay).await;
                        continue;
                    } else {
                        return Err(anyhow::anyhow!("Request timed out after {} retries", max_retries));
                    }
                }
            }
        }

        unreachable!()
    }

    /// Evaluate the success of a task (simple heuristics)
    fn evaluate_success(&self, output: &str, task_type: &str) -> f32 {
        match task_type {
            "code_generation" => {
                // Check for basic code structure
                let has_functions = output.contains("def ") || output.contains("fn ") || output.contains("function");
                let has_structure = output.contains("{") || output.contains("class") || output.contains("import");
                let not_empty = !output.trim().is_empty();

                let mut score = 0.0;
                if has_functions { score += 0.4; }
                if has_structure { score += 0.4; }
                if not_empty { score += 0.2; }

                score
            },
            "code_analysis" => {
                // Check for analysis keywords
                let has_analysis = output.to_lowercase().contains("review") ||
                                 output.to_lowercase().contains("issue") ||
                                 output.to_lowercase().contains("improvement");
                let not_empty = !output.trim().is_empty();

                if has_analysis && not_empty { 0.8 } else if not_empty { 0.5 } else { 0.0 }
            },
            "debugging" => {
                // Check for debugging keywords
                let has_debug = output.to_lowercase().contains("error") ||
                              output.to_lowercase().contains("bug") ||
                              output.to_lowercase().contains("fix");
                let not_empty = !output.trim().is_empty();

                if has_debug && not_empty { 0.9 } else if not_empty { 0.6 } else { 0.0 }
            },
            _ => {
                // General evaluation
                let length_score = (output.len() as f32 / 1000.0).min(1.0) * 0.7;
                let coherence_score = if output.contains(". ") { 0.3 } else { 0.0 };

                length_score + coherence_score
            }
        }
    }

    /// Log an experience to memory via the Curator
    pub async fn log_experience(
        &self,
        task_input: &str,
        task_result: &TaskResult,
        task_type: &str,
        context: &str,
        curator: &mut Curator,
        memory: &MemoryCore,
    ) -> Result<()> {
        let experience = Experience::new(
            task_input.to_string(),
            task_result.output.clone(),
            context.to_string(),
            task_type.to_string(),
            task_result.success_score,
        );

        curator.process_experience(experience, memory).await?;

        Ok(())
    }

    /// Get executor status
    pub fn get_status(&self) -> ExecutorStatus {
        ExecutorStatus {
            model_loaded: true,
            device: format!("vLLM:{}", self.config.vllm_endpoint),
            max_context_length: self.config.max_context_length,
        }
    }
}

/// Executor status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutorStatus {
    pub model_loaded: bool,
    pub device: String,
    pub max_context_length: usize,
}