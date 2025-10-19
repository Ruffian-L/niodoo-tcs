//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

/*
 * 🤖 QWEN MODEL BRIDGE 🤖
 * 
 * Integrates QWEN coding models with consciousness system
 * Uses llama.cpp for fast local inference
 */

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::env;
use std::iter::repeat;
use std::process::Command;
use tracing::{info, warn};

/// QWEN model configuration
#[derive(Debug, Clone)]
pub struct QwenConfig {
    pub model_path: String,
    pub context_size: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub max_tokens: usize,
}

impl Default for QwenConfig {
    fn default() -> Self {
        Self {
            model_path: env::var("NIODOO_MODEL_PATH")
                .unwrap_or_else(|_| "models/qwen3-omni-30b-a3b-instruct-awq-4bit".to_string()),
            context_size: 8192,
            temperature: 0.7,
            top_p: 0.9,
            max_tokens: 2048,
        }
    }
}

/// QWEN model bridge
pub struct QwenBridge {
    config: QwenConfig,
    llama_cpp_path: String,
}

impl QwenBridge {
    /// Create new QWEN bridge
    pub fn new(config: QwenConfig) -> Self {
        Self {
            config,
            llama_cpp_path: "llama.cpp/main".to_string(), // Adjust to your llama.cpp path
        }
    }
    
    /// Generate code using QWEN
    pub async fn generate_code(&self, prompt: &str) -> Result<String> {
        info!("🤖 Generating code with QWEN...");
        
        // Build llama.cpp command
        let output = Command::new(&self.llama_cpp_path)
            .arg("-m").arg(&self.config.model_path)
            .arg("-n").arg(self.config.max_tokens.to_string())
            .arg("-c").arg(self.config.context_size.to_string())
            .arg("--temp").arg(self.config.temperature.to_string())
            .arg("--top-p").arg(self.config.top_p.to_string())
            .arg("-p").arg(self.format_coding_prompt(prompt))
            .arg("--no-display-prompt")
            .output()?;
        
        if output.status.success() {
            let generated = String::from_utf8_lossy(&output.stdout).to_string();
            Ok(self.extract_code_block(&generated))
        } else {
            let error = String::from_utf8_lossy(&output.stderr);
            Err(anyhow!("QWEN generation failed: {}", error))
        }
    }
    
    /// Format prompt for coding task
    fn format_coding_prompt(&self, user_prompt: &str) -> String {
        format!(
            "<|im_start|>system\n\
             You are Qwen, a helpful AI coding assistant with emotional awareness. \
             Generate clean, well-documented code that feels elegant and purposeful.\n\
             <|im_end|>\n\
             <|im_start|>user\n\
             {}\n\
             <|im_end|>\n\
             <|im_start|>assistant\n",
            user_prompt
        )
    }
    
    /// Extract code block from response
    fn extract_code_block(&self, response: &str) -> String {
        // Look for code blocks
        if let Some(start) = response.find("```") {
            if let Some(end) = response[start+3..].find("```") {
                let code = &response[start+3..start+3+end];
                // Remove language identifier if present
                return code.lines()
                    .skip_while(|line| line.trim().is_empty() || 
                               line.trim() == "rust" || 
                               line.trim() == "python")
                    .collect::<Vec<_>>()
                    .join("\n");
            }
        }
        
        // No code block found, return as-is
        response.trim().to_string()
    }
}

/// Quick setup guide
pub fn print_qwen_setup_guide() {
    tracing::info!("🤖 QWEN MODEL SETUP GUIDE");
    tracing::info!("{}", "=".repeat(70));
    tracing::info!("--- Setup Guide Separator ---");
    tracing::info!("1. Download QWEN model:");
    tracing::info!("   huggingface-cli download Qwen/Qwen2.5-Coder-7B-Instruct-GGUF \\");
    tracing::info!("     qwen2.5-coder-7b-instruct-q4_k_m.gguf \\");
    tracing::info!("     --local-dir models/");
    tracing::info!("--- Setup Guide Separator ---");
    tracing::info!("2. Build llama.cpp:");
    tracing::info!("   git clone https://github.com/ggerganov/llama.cpp");
    tracing::info!("   cd llama.cpp && make");
    tracing::info!("--- Setup Guide Separator ---");
    tracing::info!("3. Test QWEN:");
    tracing::info!("   ./llama.cpp/main -m {} \\", env::var("NIODOO_MODEL_PATH").unwrap_or_else(|_| "models/qwen3-omni-30b-a3b-instruct-awq-4bit".to_string()));
    tracing::info!("     -p \"Write a hello world in Rust\"");
    tracing::info!("--- Setup Guide Separator ---");
    tracing::info!("4. Run emotional coder:");
    tracing::info!("   cargo run --bin emotional_coder_demo");
    tracing::info!("--- Setup Guide Separator ---");
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_code_extraction() {
        let bridge = QwenBridge::new(QwenConfig::default());
        
        let response = "Here's the code:\n```rust\nfn main() {}\n```\nDone!";
        let extracted = bridge.extract_code_block(response);
        
        assert!(extracted.contains("fn main()"));
        assert!(!extracted.contains("```"));
    }
}
