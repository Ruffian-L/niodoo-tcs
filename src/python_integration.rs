//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

//! Python Integration for QLoRA Fine-Tuning and Model Comparison
//!
//! This module provides Rust interfaces to call Python QLoRA training
//! and model comparison scripts.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::process::Command;

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelComparisonResult {
    pub total_tests: usize,
    pub avg_latency_improvement: f64,
    pub avg_rouge_improvement: f64,
    pub avg_coherence_improvement: f64,
    pub results: Vec<TestResult>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TestResult {
    pub prompt: String,
    pub base_response: String,
    pub ft_response: String,
    pub base_latency: f64,
    pub ft_latency: f64,
    pub base_rouge1: f64,
    pub ft_rouge1: f64,
    pub base_coherence: f64,
    pub ft_coherence: f64,
    pub latency_improvement: f64,
    pub rouge_improvement: f64,
    pub coherence_improvement: f64,
}

pub struct PythonQLoRAIntegration {
    python_path: String,
    project_root: String,
}

impl PythonQLoRAIntegration {
    pub fn new() -> Self {
        Self {
            python_path: "python3".to_string(),
            project_root: std::env::current_dir()
                .unwrap()
                .to_string_lossy()
                .to_string(),
        }
    }

    /// Run QLoRA fine-tuning on accumulated learning events
    pub fn run_fine_tuning(&self) -> Result<bool> {
        let script_path = Path::new(&self.project_root)
            .join("python")
            .join("qlora")
            .join("finetune.py");

        if !script_path.exists() {
            return Err(anyhow!("QLoRA script not found at {:?}", script_path));
        }

        println!("🚀 Starting QLoRA fine-tuning...");

        let output = Command::new(&self.python_path)
            .arg(script_path)
            .current_dir(&self.project_root)
            .output()?;

        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            println!("✅ QLoRA fine-tuning completed:\n{}", stdout);
            Ok(true)
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            eprintln!("❌ QLoRA fine-tuning failed:\n{}", stderr);
            Ok(false)
        }
    }

    /// Run model comparison between base and fine-tuned Qwen
    pub fn run_model_comparison(&self) -> Result<ModelComparisonResult> {
        let script_path = Path::new(&self.project_root)
            .join("python")
            .join("inference")
            .join("compare.py");

        if !script_path.exists() {
            return Err(anyhow!("Comparison script not found at {:?}", script_path));
        }

        println!("🔍 Running model comparison...");

        let output = Command::new(&self.python_path)
            .arg(script_path)
            .current_dir(&self.project_root)
            .output()?;

        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            println!("✅ Model comparison completed:\n{}", stdout);

            // Load results from file
            let results_path = Path::new(&self.project_root)
                .join("results")
                .join("model_comparison.json");

            if results_path.exists() {
                let content = std::fs::read_to_string(results_path)?;
                let results: ModelComparisonResult = serde_json::from_str(&content)?;
                Ok(results)
            } else {
                Err(anyhow!("Model comparison results file not found"))
            }
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            eprintln!("❌ Model comparison failed:\n{}", stderr);
            Err(anyhow!("Model comparison failed"))
        }
    }

    /// Check if Python environment is properly set up
    pub fn check_environment(&self) -> Result<bool> {
        println!("🔧 Checking Python environment...");

        // Check if Python is available
        let python_check = Command::new(&self.python_path).arg("--version").output()?;

        if !python_check.status.success() {
            eprintln!("❌ Python not found");
            return Ok(false);
        }

        let python_version = String::from_utf8_lossy(&python_check.stdout);
        println!("✅ Python available: {}", python_version.trim());

        // Check if required packages are installed
        let requirements_path = Path::new(&self.project_root)
            .join("python")
            .join("requirements.txt");

        if requirements_path.exists() {
            println!("📦 Checking Python dependencies...");
            let pip_check = Command::new(&self.python_path)
                .arg("-c")
                .arg("import torch, transformers, peft, bitsandbytes, trl, datasets, rouge_score; print('All dependencies available')")
                .output()?;

            if pip_check.status.success() {
                println!("✅ All Python dependencies available");
                Ok(true)
            } else {
                let stderr = String::from_utf8_lossy(&pip_check.stderr);
                eprintln!("❌ Missing Python dependencies:\n{}", stderr);
                eprintln!("💡 Run: pip install -r python/requirements.txt");
                Ok(false)
            }
        } else {
            eprintln!("⚠️ Requirements file not found");
            Ok(false)
        }
    }

    /// Install Python dependencies
    pub fn install_dependencies(&self) -> Result<()> {
        let requirements_path = Path::new(&self.project_root)
            .join("python")
            .join("requirements.txt");

        if !requirements_path.exists() {
            return Err(anyhow!("Requirements file not found"));
        }

        println!("📦 Installing Python dependencies...");

        let output = Command::new(&self.python_path)
            .arg("-m")
            .arg("pip")
            .arg("install")
            .arg("-r")
            .arg(requirements_path)
            .output()?;

        if output.status.success() {
            println!("✅ Python dependencies installed");
            Ok(())
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Err(anyhow!("Failed to install dependencies: {}", stderr))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_python_integration_creation() {
        let integration = PythonQLoRAIntegration::new();
        assert!(!integration.python_path.is_empty());
        assert!(!integration.project_root.is_empty());
    }
}
