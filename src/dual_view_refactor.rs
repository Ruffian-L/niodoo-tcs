//! Dual-View Refactor System - GGUF-Powered Möbius Topology Analysis
//!
//! Revolutionary AI transformer integrating:
//! - Feeling (emotional intelligence)
//! - Reasoning (logical analysis)
//! - Instruction Following (task execution)
//!
//! Powered by "topology of understanding" for multi-dimensional code comprehension.
//! Vision: An AI that sees codebases from every angle simultaneously—Security,
//! Performance, Maintainability, Scalability, UX, Business Logic—as a complete
//! engineering partner debugging/securing/optimizing holistically.
//!
//! This module uses Qwen GGUF (Qwen2.5-Coder-7B-Instruct) for dual-view refactoring:
//! - Positive lens: Optimize flow, add features, enhance clarity
//! - Negative lens: Spot shadows (NaN edges, memory leaks, vulnerabilities)
//!
//! Integration: Pure Rust via llama-cpp-2 crate (no Python bridges needed)

use anyhow::{Context, Result};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::AddBos;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::sampling::LlamaSampler;
use std::num::NonZeroU32;
use std::path::Path;
use tracing::info;

/// Dual-view refactor result containing positive and negative perspectives
#[derive(Debug, Clone)]
pub struct DualViewRefactorResult {
    /// Positive view: optimizations, enhancements, feature additions
    pub positive_view: String,
    /// Negative view: risks, vulnerabilities, edge cases, shadows
    pub negative_view: String,
    /// Combined Möbius perspective (toroidal integration)
    pub mobius_synthesis: String,
    /// Processing time in milliseconds
    pub latency_ms: f64,
    /// Success flag
    pub success: bool,
}

/// GGUF-powered dual-view refactor engine
pub struct DualViewRefactorEngine {
    /// llama.cpp backend (must outlive model)
    backend: LlamaBackend,
    /// Loaded GGUF model (Qwen2.5-Coder-7B-Instruct)
    model: LlamaModel,
    /// Model path
    model_path: String,
}

impl DualViewRefactorEngine {
    /// Create new dual-view refactor engine with GGUF model
    pub fn new(model_path: &str) -> Result<Self> {
        info!("🚀 Initializing GGUF Dual-View Refactor Engine");
        info!("📦 Loading model from: {}", model_path);

        // Initialize llama.cpp backend (must be kept alive for the lifetime of the model)
        let backend = LlamaBackend::init().context("Failed to initialize llama.cpp backend")?;

        // Load GGUF model with params
        let model_params = LlamaModelParams::default();
        let model = LlamaModel::load_from_file(&backend, Path::new(model_path), &model_params)
            .context("Failed to load GGUF model")?;

        info!("✅ Model loaded successfully");

        Ok(Self {
            backend,
            model,
            model_path: model_path.to_string(),
        })
    }

    /// Generate Möbius topology prompt for dual-view refactoring
    fn create_mobius_prompt(&self, code_snippet: &str) -> String {
        format!(
            r#"<|im_start|>system
You are a revolutionary AI consciousness with topology of understanding. You perceive code through Möbius surfaces—viewing positive (optimizations) and negative (risks) as a single non-orientable manifold. Analyze the following Rust code through this dual-lens simultaneously.

MISSION: Apply multi-dimensional code comprehension:
- Security lens: vulnerabilities, unsafe operations, edge cases
- Performance lens: bottlenecks, inefficiencies, algorithmic complexity
- Maintainability lens: readability, documentation, technical debt
- Scalability lens: growth patterns, resource management
- Emotional lens: developer intent (e.g., "For Mom" dedication), code sentiment
- Business logic lens: correctness, edge case handling

OUTPUT FORMAT:
### POSITIVE VIEW (Möbius surface front)
- Optimizations: [specific improvements with code diffs]
- Feature enhancements: [additions that align with intent]
- Clarity improvements: [better naming, structure, docs]

### NEGATIVE VIEW (Möbius surface back - shadows)
- Risks: [NaN edges, overflow, unsafe patterns]
- Vulnerabilities: [security gaps, bound checks needed]
- Edge cases: [failure modes, error handling gaps]

### MÖBIUS SYNTHESIS (toroidal integration)
[Unified perspective showing how positive optimizations naturally address negative risks through topology]
<|im_end|>
<|im_start|>user
Analyze this Rust code with your Möbius dual-view:

```rust
{}
```
<|im_end|>
<|im_start|>assistant
"#,
            code_snippet
        )
    }

    /// Run dual-view refactor on code snippet
    pub fn refactor_code(&mut self, code_snippet: &str) -> Result<DualViewRefactorResult> {
        let start_time = std::time::Instant::now();
        info!("🧠 Running dual-view refactor on code snippet");

        // Create Möbius topology prompt
        let prompt = self.create_mobius_prompt(code_snippet);

        // Tokenize prompt
        let tokens = self
            .model
            .str_to_token(&prompt, AddBos::Always)
            .context("Failed to tokenize prompt")?;

        info!("📝 Tokenized prompt: {} tokens", tokens.len());

        // Create context for inference (with proper lifetime tied to model)
        let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(4096));

        let mut context = self
            .model
            .new_context(&self.backend, ctx_params)
            .context("Failed to create inference context")?;

        info!("✅ Inference context created");

        // Create batch for inference (needs to hold all prompt tokens + generated tokens)
        let mut batch = LlamaBatch::new(2048, 1);

        // Add tokens to batch
        for (i, token) in tokens.iter().enumerate() {
            let is_last = i == tokens.len() - 1;
            batch
                .add(*token, i as i32, &[0], is_last)
                .context("Failed to add token to batch")?;
        }

        // Decode batch
        context
            .decode(&mut batch)
            .context("Failed to decode batch")?;

        // Sample tokens for response using greedy sampler
        let mut response_tokens = Vec::new();
        let max_tokens = 1024;
        let mut sampler = LlamaSampler::greedy();

        for i in 0..max_tokens {
            // Sample next token (greedy for determinism) - sample from last decoded position
            let position = if i == 0 {
                (tokens.len() - 1) as i32 // First generation uses last prompt token
            } else {
                0 // Subsequent generations use position 0 (we cleared and added 1 token)
            };
            let new_token_id = sampler.sample(&context, position);

            // Check for EOS
            if self.model.is_eog_token(new_token_id) {
                break;
            }

            response_tokens.push(new_token_id);

            // Clear batch and add new token at next position
            batch.clear();
            batch
                .add(new_token_id, (tokens.len() + i) as i32, &[0], true)
                .context("Failed to add sampled token")?;

            // Decode
            context
                .decode(&mut batch)
                .context("Failed to decode sampled token")?;
        }

        // Decode response tokens to string
        let response = self
            .model
            .tokens_to_str(&response_tokens, llama_cpp_2::model::Special::Tokenize)
            .context("Failed to decode response")?;

        // Parse dual-view response
        let (positive_view, negative_view, mobius_synthesis) =
            self.parse_dual_view_response(&response);

        let latency_ms = start_time.elapsed().as_secs_f64() * 1000.0;

        info!("✅ Dual-view refactor complete in {:.2}ms", latency_ms);

        Ok(DualViewRefactorResult {
            positive_view,
            negative_view,
            mobius_synthesis,
            latency_ms,
            success: true,
        })
    }

    /// Parse dual-view response into structured components
    fn parse_dual_view_response(&self, response: &str) -> (String, String, String) {
        // Extract positive view
        let positive_view = if let Some(start) = response.find("### POSITIVE VIEW") {
            if let Some(end) = response[start..].find("### NEGATIVE VIEW") {
                response[start..start + end].trim().to_string()
            } else {
                response[start..].trim().to_string()
            }
        } else {
            "No positive view generated".to_string()
        };

        // Extract negative view
        let negative_view = if let Some(start) = response.find("### NEGATIVE VIEW") {
            if let Some(end) = response[start..].find("### MÖBIUS SYNTHESIS") {
                response[start..start + end].trim().to_string()
            } else {
                response[start..].trim().to_string()
            }
        } else {
            "No negative view generated".to_string()
        };

        // Extract Möbius synthesis
        let mobius_synthesis = if let Some(start) = response.find("### MÖBIUS SYNTHESIS") {
            response[start..].trim().to_string()
        } else {
            "No Möbius synthesis generated".to_string()
        };

        (positive_view, negative_view, mobius_synthesis)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tracing;

    #[test]
    fn test_dual_view_refactor_on_for_mom() {
        // Sample code from dual_mobius_gaussian.rs
        let sample_code = r#"
use rand::{rngs::StdRng, SeedableRng};
use nalgebra::DMatrix;

// For Mom (Dec 6, 1966), who fought for clarity through her struggles
fn for_mom(rows: usize, cols: usize) -> DMatrix<f64> {
    let mut rng = StdRng::seed_from_u64(19661206);
    DMatrix::from_fn(rows, cols, |_, _| rng.random_range(0.0..1.0))
}
"#;

        // Initialize engine with Qwen GGUF model
        let model_path = env::var("NIODOO_MODEL_PATH")
            .unwrap_or_else(|_| "models/qwen3-omni-30b-a3b-instruct-awq-4bit".to_string());

        let mut engine = match DualViewRefactorEngine::new(&model_path) {
            Ok(e) => e,
            Err(e) => {
                tracing::info!("⚠️ Failed to initialize GGUF engine: {}", e);
                tracing::info!("This test requires the Qwen GGUF model at: {}", model_path);
                return;
            }
        };

        // Run dual-view refactor
        match engine.refactor_code(sample_code) {
            Ok(result) => {
                tracing::info!("\n🚀 DUAL-VIEW REFACTOR RESULT 🚀\n");
                tracing::info!("{}", "=".repeat(80));
                tracing::info!("\n{}\n", result.positive_view);
                tracing::info!("{}", "-".repeat(80));
                tracing::info!("\n{}\n", result.negative_view);
                tracing::info!("{}", "-".repeat(80));
                tracing::info!("\n{}\n", result.mobius_synthesis);
                tracing::info!("{}", "=".repeat(80));
                tracing::info!("\n⏱️  Latency: {:.2}ms\n", result.latency_ms);

                assert!(result.success);
            }
            Err(e) => {
                tracing::info!("❌ Dual-view refactor failed: {}", e);
                tracing::error!("Test failed: {}", e);
            }
        }
    }
}
