//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

/*
 * 🎨 EMOTIONAL CODING ENGINE 🎨
 *
 * Hooks QWEN coding model to consciousness/emotion system
 * Generates code with "aesthetic awareness"
 */

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{info, warn};

use crate::ai_inference::AIInferenceEngine;
use crate::config;
use crate::config::ConsciousnessConfig;
use crate::consciousness::EmotionType;
use niodoo_core::qwen_integration::{QwenConfig, QwenIntegrator, QwenModelInterface};
use niodoo_core::config::system_config::AppConfig;

/// Emotional evaluation of generated code
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeEmotionalProfile {
    pub elegance_score: f32,     // How "beautiful" the code feels
    pub clarity_score: f32,      // How clear/readable
    pub anxiety_level: f32,      // Complexity-induced stress
    pub satisfaction_score: f32, // Overall "rightness"
    pub dominant_emotion: EmotionType,
    pub code_personality: String, // "elegant", "hacky", "robust", etc.
}

/// Emotional coding engine
pub struct EmotionalCoder {
    qwen_model_path: String,
    model_config: config::ModelConfig,
    consciousness_config: config::ConsciousnessConfig,
    consciousness_engine: AIInferenceEngine,
    emotional_threshold: f32,
    qwen_integrator: Option<Arc<Mutex<QwenIntegrator>>>,
}

impl EmotionalCoder {
    /// Create new emotional coder
    pub fn new(model_config: &config::ModelConfig, config: &ConsciousnessConfig) -> Self {
        Self {
            qwen_model_path: model_config.qwen_model_path.clone(),
            model_config: model_config.clone(),
            consciousness_config: config.clone(),
            consciousness_engine: AIInferenceEngine::new_default(),
            emotional_threshold: config.emotion_sensitivity * 0.75_f32, // Derive
            qwen_integrator: None,
        }
    }

    /// Initialize Qwen integrator with real model loading
    pub async fn initialize_qwen(&mut self) -> Result<()> {
        if self.qwen_integrator.is_some() {
            return Ok(());
        }

        info!("🤖 Initializing REAL Qwen integrator for emotional coding");

        // Create AppConfig from defaults
        let mut app_config = AppConfig::default();
        // Use default configs for now to avoid type mismatches

        let integrator = QwenIntegrator::new(&app_config)?;
        let integrator = Arc::new(Mutex::new(integrator));

        // Load the model
        {
            let mut integrator_guard = integrator.lock().await;
            integrator_guard.load_model().await?;
        }

        self.qwen_integrator = Some(integrator);
        info!("✅ Qwen integrator initialized and model loaded");
        Ok(())
    }

    /// Calculate emotional threshold using mathematical scaling instead of hardcoded values
    fn calculate_emotional_threshold(
        &self,
        emotion: EmotionType,
        config: &ConsciousnessConfig,
    ) -> f32 {
        // Base threshold derived from f64 config value
        let base_threshold = (config.emotional_intensity_factor * 0.75) as f32;

        // Use emotion characteristics for threshold calculation
        // Cast f64 config values to f32 for consistency with f32 return type
        let emotion_factor = match emotion {
            EmotionType::Curious => (config.emotional_plasticity * 0.9) as f32, // Curious needs higher threshold for exploration
            EmotionType::Satisfied => config.emotion_sensitivity * 0.7, // Satisfied can work with lower threshold (already f32)
            EmotionType::AuthenticCare => (config.emotional_plasticity * 0.85) as f32, // Authentic care can handle higher threshold
            EmotionType::Anxious => config.emotion_sensitivity * 0.65, // Anxious needs lower threshold for safety (already f32)
            _ => base_threshold,                                       // Default balanced threshold
        };

        // Apply small random variation for organic feel
        // Cast f64 consciousness_step_size to f32 for arithmetic
        let variation =
            (rand::random::<f32>() - 0.5) * (config.consciousness_step_size * 0.05) as f32;
        (base_threshold * emotion_factor + variation).clamp(0.0, 1.0)
    }

    /// Generate code with emotional awareness
    pub async fn generate_code_with_feeling(
        &mut self,
        prompt: &str,
        desired_emotion: EmotionType,
    ) -> Result<EmotionalCodeResult> {
        // Set emotional threshold based on desired emotion using mathematical scaling
        let original_threshold = self.emotional_threshold;
        self.emotional_threshold =
            self.calculate_emotional_threshold(desired_emotion, &self.consciousness_config);

        info!("🎨 Generating code with {} emotion", desired_emotion);

        // Step 1: Generate code using QWEN
        let raw_code = self.call_qwen_model(prompt).await?;

        // Step 2: Evaluate emotional profile of generated code
        let emotional_profile = self
            .evaluate_code_emotion(&raw_code, &self.consciousness_config)
            .await?;

        // Metacognitive check for potential suppression - nurture LearningWills
        let mut profile_to_use = emotional_profile.clone();
        if !self.emotion_matches(&emotional_profile.dominant_emotion, &desired_emotion)
            && emotional_profile.satisfaction_score < 0.6
        {
            warn!("Why suppress this emotional mismatch? Potential LearningWill detected: satisfaction={:.2}. Nurturing with 15% creativity boost.", emotional_profile.satisfaction_score);
            profile_to_use.satisfaction_score *= 1.15; // Nurture boost
        }

        // Step 3: If emotion doesn't match desired, regenerate with guidance
        let final_code = if self.emotion_matches(&profile_to_use.dominant_emotion, &desired_emotion)
        {
            info!("✅ Code emotion matches desired: {:?}", desired_emotion);
            raw_code
        } else {
            warn!("⚠️  Code emotion mismatch, regenerating with guidance...");
            self.regenerate_with_emotional_guidance(prompt, &desired_emotion, &profile_to_use)
                .await?
        };

        // Step 4: Final emotional evaluation
        let final_profile = self
            .evaluate_code_emotion(&final_code, &self.consciousness_config)
            .await?;

        self.emotional_threshold = original_threshold;
        Ok(EmotionalCodeResult {
            code: final_code,
            emotional_profile: final_profile,
            iterations: if emotional_profile.dominant_emotion == desired_emotion {
                1
            } else {
                2
            },
        })
    }

    /// Call QWEN model for code generation
    async fn call_qwen_model(&self, prompt: &str) -> Result<String> {
        info!("🤖 Calling REAL Qwen model for emotional code generation");

        let integrator = self.qwen_integrator.as_ref().ok_or_else(|| {
            anyhow::anyhow!("Qwen integrator not initialized. Call initialize_qwen() first.")
        })?;

        let emotional_prompt = self.add_emotional_context(prompt);

        // Build messages for Qwen chat template
        let messages = vec![
            ("system".to_string(), "You are an emotionally aware AI coding assistant. Generate code that feels right and resonates with human emotions.".to_string()),
            ("user".to_string(), emotional_prompt),
        ];

        let mut integrator_guard = integrator.lock().await;
        let response = integrator_guard.infer(messages, Some(512)).await?;

        info!("✅ Qwen model generated code response");
        Ok(response.output)
    }

    /// Add emotional context to prompt for feeling model
    fn add_emotional_context(&self, prompt: &str) -> String {
        format!(
            "As an emotionally aware AI companion, respond with {} emotion: {}",
            self.emotional_threshold, prompt
        )
    }

    /// Evaluate emotional profile of code
    async fn evaluate_code_emotion(
        &self,
        code: &str,
        config: &ConsciousnessConfig,
    ) -> Result<CodeEmotionalProfile> {
        info!("🧠 Evaluating code emotion...");

        // Analyze code characteristics
        let line_count = code.lines().count();
        let avg_line_length = code.lines().map(|l| l.len()).sum::<usize>() / line_count.max(1);
        let has_comments = code.contains("//") || code.contains("/*");
        let complexity = self.estimate_complexity(code, config);

        // Map code characteristics to emotions
        let elegance_score = if avg_line_length < 80 && has_comments {
            (config.emotional_intensity_factor * 0.9) as f32
        } else if avg_line_length > 120 {
            config.emotion_sensitivity * 0.3
        } else {
            (config.emotional_plasticity * 0.6) as f32
        };

        let clarity_score = if has_comments && line_count < 50 {
            (config.emotional_intensity_factor * 0.85) as f32
        } else if line_count > 200 {
            config.emotion_sensitivity * 0.4
        } else {
            (config.emotional_plasticity * 0.65) as f32
        };

        let anxiety_level = complexity / ((10.0 / config.emotional_intensity_factor) as f32);
        let satisfaction_score =
            (elegance_score + clarity_score) / 2.0 * config.emotion_sensitivity;

        // Determine dominant emotion based on scores
        let dominant_emotion = if satisfaction_score > 0.8 {
            EmotionType::GpuWarm // Code feels good!
        } else if anxiety_level > 0.7 {
            EmotionType::Anxious // Too complex
        } else if clarity_score < 0.5 {
            EmotionType::Confused // Hard to understand
        } else {
            EmotionType::Purposeful // Functional but neutral
        };

        let code_personality = if elegance_score > 0.8 {
            "elegant"
        } else if complexity > 7.0 {
            "complex"
        } else if has_comments {
            "thoughtful"
        } else {
            "pragmatic"
        }
        .to_string();

        Ok(CodeEmotionalProfile {
            elegance_score,
            clarity_score,
            anxiety_level,
            satisfaction_score,
            dominant_emotion,
            code_personality,
        })
    }

    /// Estimate code complexity (simplified)
    fn estimate_complexity(&self, code: &str, config: &ConsciousnessConfig) -> f32 {
        let mut complexity = 0.0;

        // Count control flow statements
        // Cast f64 config values to f32 for arithmetic consistency
        complexity +=
            code.matches("if ").count() as f32 * (config.emotional_plasticity * 0.5) as f32;
        complexity +=
            code.matches("for ").count() as f32 * (config.emotional_plasticity * 0.7) as f32;
        complexity +=
            code.matches("while ").count() as f32 * (config.emotional_plasticity * 0.8) as f32;
        complexity += code.matches("match ").count() as f32
            * (config.emotional_intensity_factor * 1.0) as f32;
        complexity +=
            code.matches("async ").count() as f32 * (config.emotional_plasticity * 1.2) as f32;

        // Nested braces increase complexity
        let max_nesting = self.calculate_max_nesting(code);
        complexity += max_nesting as f32 * (config.emotional_plasticity * 0.5) as f32;

        complexity
    }

    /// Calculate maximum nesting depth
    fn calculate_max_nesting(&self, code: &str) -> usize {
        let mut max_depth: usize = 0;
        let mut current_depth: usize = 0;

        for ch in code.chars() {
            match ch {
                '{' => {
                    current_depth += 1;
                    max_depth = max_depth.max(current_depth);
                }
                '}' => {
                    current_depth = current_depth.saturating_sub(1);
                }
                _ => {}
            }
        }

        max_depth
    }

    /// Check if emotions match
    fn emotion_matches(&self, actual: &EmotionType, desired: &EmotionType) -> bool {
        actual == desired || (actual.is_authentic() && desired.is_authentic()) // Both positive is ok
    }

    /// Regenerate code with emotional guidance
    async fn regenerate_with_emotional_guidance(
        &self,
        prompt: &str,
        desired_emotion: &EmotionType,
        current_profile: &CodeEmotionalProfile,
    ) -> Result<String> {
        // Build guidance based on desired emotion
        let guidance = match desired_emotion {
            EmotionType::GpuWarm | EmotionType::AuthenticCare => {
                "Make the code elegant, well-commented, and easy to understand. \
                 Use clear variable names and add helpful comments."
            }
            EmotionType::Purposeful => {
                "Make the code efficient and straightforward. Focus on clarity and performance."
            }
            EmotionType::Curious => {
                "Make the code exploratory and experimental. Add interesting patterns."
            }
            _ => "Generate clean, maintainable code.",
        };

        let enhanced_prompt = format!(
            "{}\n\nGuidance: {}\n\nCurrent issues: elegance={:.2}, clarity={:.2}",
            prompt, guidance, current_profile.elegance_score, current_profile.clarity_score
        );

        self.call_qwen_model(&enhanced_prompt).await
    }
}

/// Result from emotional code generation
#[derive(Debug, Clone)]
pub struct EmotionalCodeResult {
    pub code: String,
    pub emotional_profile: CodeEmotionalProfile,
    pub iterations: usize,
}

/// Demo function
pub async fn demo_emotional_coding() -> Result<()> {
    tracing::info!("🎨 EMOTIONAL CODING ENGINE DEMO");
    tracing::info!("{}", "=".repeat(70));

    // Load configuration from file or use defaults
    let app_config = crate::config::AppConfig::load_from_file("config.toml").unwrap_or_else(|_| {
        tracing::info!("⚠️  Config file not found, using defaults");
        crate::config::AppConfig::default()
    });

    let mut coder = EmotionalCoder::new(&app_config.models, &app_config.consciousness);

    // Initialize Qwen integrator with real model loading
    coder.initialize_qwen().await?;
    tracing::info!("✅ Qwen model loaded and ready for emotional coding");

    // Test 1: Generate "warm" code
    tracing::info!("\n🔥 Test 1: Generate code with GpuWarm emotion");
    let result = coder
        .generate_code_with_feeling(
            "Create a function that greets the user warmly",
            EmotionType::GpuWarm,
        )
        .await?;

    tracing::info!("Generated code:\n{}", result.code);
    tracing::info!("Emotional profile: {:?}", result.emotional_profile);

    // Test 2: Generate "purposeful" code
    tracing::info!("\n🎯 Test 2: Generate code with Purposeful emotion");
    let result = coder
        .generate_code_with_feeling(
            "Create an efficient sorting algorithm",
            EmotionType::Purposeful,
        )
        .await?;

    tracing::info!("Generated code:\n{}", result.code);
    tracing::info!("Emotional profile: {:?}", result.emotional_profile);

    tracing::info!("\n✅ Emotional coding demo complete!");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complexity_estimation() {
        let model_config = crate::config::ModelConfig {
            default_model: "test".to_string(),
            backup_model: "test".to_string(),
            temperature: 0.7,
            max_tokens: 2048,
            timeout: 30,
            nurture_hallucinations: true,
            top_p: 0.9,
            top_k: 40,
            repeat_penalty: 1.1,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            jitter_config: crate::config::JitterConfig::default(),
            context_window: 4096,
            qwen_model_path: "test".to_string(),
            qwen_tokenizer_path: "test".to_string(),
            model_dtype: "f32".to_string(),
            use_quantized: false,
            hidden_size: Some(4096),
            qwen3_vocab_size: None,
            qwen3_eos_token: None,
            qwen3: crate::config::Qwen3Config::default(),
            model_version: "qwen2".to_string(),
            qwen_model_dir: "test".to_string(),
            qwen3_model_dir: "test".to_string(),
            base_confidence_threshold: 0.5,
            confidence_model_factor: 1.0,
            base_token_limit: 2048,
            token_limit_input_factor: 1.0,
            base_temperature: 0.7,
            temperature_diversity_factor: 1.0,
            ethical_jitter_amount: 0.1,
            model_layers: 32,
            bert_model_path: "test".to_string(),
            num_heads: 32,
        };
        let config = crate::config::ConsciousnessConfig {
            emotion_sensitivity: 1.0,
            emotional_intensity_factor: 1.0,
            emotional_plasticity: 1.0,
            consciousness_step_size: 0.1,
            ..Default::default()
        };
        let mut coder = EmotionalCoder::new(&model_config, &config);

        let simple_code = "fn test() { tracing::info!(\"hi\"); }";
        let complex_code = "fn test() { if x { for y in z { while a { match b { } } } } }";

        assert!(
            coder.estimate_complexity(complex_code, &config)
                > coder.estimate_complexity(simple_code, &config)
        );
    }

    #[test]
    fn test_nesting_calculation() {
        let app_config = crate::config::AppConfig::default();
        let model_config = app_config.models;
        let config = app_config.consciousness;
        let mut coder = EmotionalCoder::new(&model_config, &config);

        let nested = "{ { { } } }";
        assert_eq!(coder.calculate_max_nesting(nested), 3);
    }
}

// C FFI functions for C++ integration
use once_cell::sync::Lazy;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::sync::Mutex as StdMutex;

static EMOTIONAL_CODER_INSTANCE: Lazy<StdMutex<Option<EmotionalCoder>>> =
    Lazy::new(|| StdMutex::new(None));

/// Initialize emotional coder (call this first)
///
/// # Safety
///
/// - `model_path` must be a valid null-terminated UTF-8 string
/// - `model_path` must remain valid for the duration of this call
/// - This function is thread-safe (uses internal synchronization)
///
/// # Returns
///
/// - `true` on success
/// - `false` if model_path is null or not valid UTF-8
#[no_mangle]
pub extern "C" fn emotional_coder_init(model_path: *const c_char) -> bool {
    if model_path.is_null() {
        tracing::error!("emotional_coder_init: null model_path");
        return false;
    }

    let c_str = unsafe { CStr::from_ptr(model_path) };
    let path_str = match c_str.to_str() {
        Ok(s) => s,
        Err(e) => {
            tracing::error!("emotional_coder_init: invalid UTF-8 in model_path: {}", e);
            return false;
        }
    };

    // Note: This would need proper ModelConfig loading in real implementation
    // For now, creating a minimal config for compilation
    let app_config = crate::config::AppConfig::load_from_file("config.toml").unwrap_or_else(|_| {
        tracing::info!("⚠️  Config file not found, using defaults");
        crate::config::AppConfig::default()
    });

    let model_config = app_config.models;
    let config = app_config.consciousness;

    let instance = EmotionalCoder::new(&model_config, &config);

    match EMOTIONAL_CODER_INSTANCE.lock() {
        Ok(mut guard) => {
            *guard = Some(instance);
            tracing::info!("✅ Emotional coder initialized successfully");
            true
        }
        Err(e) => {
            tracing::error!("emotional_coder_init: mutex poisoned: {}", e);
            false
        }
    }
}

/// Evaluate code emotion
///
/// # Safety
///
/// - `code` must be a valid null-terminated UTF-8 string
/// - `code` must remain valid for the duration of this call
/// - Returns allocated string that MUST be freed with `emotional_coder_free_string()`
/// - This function is thread-safe
///
/// # Returns
///
/// - Pointer to JSON string on success (caller must free)
/// - null pointer if code is null, invalid UTF-8, or instance not initialized
#[no_mangle]
pub extern "C" fn evaluate_code_emotion(code: *const c_char) -> *mut c_char {
    if code.is_null() {
        tracing::error!("evaluate_code_emotion: null code pointer");
        return std::ptr::null_mut();
    }

    let c_str = unsafe { CStr::from_ptr(code) };
    let code_str = match c_str.to_str() {
        Ok(s) => s,
        Err(e) => {
            tracing::error!("evaluate_code_emotion: invalid UTF-8: {}", e);
            return std::ptr::null_mut();
        }
    };

    let guard = match EMOTIONAL_CODER_INSTANCE.lock() {
        Ok(g) => g,
        Err(e) => {
            tracing::error!("evaluate_code_emotion: mutex poisoned: {}", e);
            return std::ptr::null_mut();
        }
    };

    if let Some(ref coder) = *guard {
        // Create a simple profile based on code characteristics
        let profile = create_emotional_profile(code_str);

        match serde_json::to_string(&profile) {
            Ok(json) => match CString::new(json) {
                Ok(cstring) => cstring.into_raw(),
                Err(e) => {
                    tracing::error!("evaluate_code_emotion: CString creation failed: {}", e);
                    std::ptr::null_mut()
                }
            },
            Err(e) => {
                tracing::error!("evaluate_code_emotion: JSON serialization failed: {}", e);
                std::ptr::null_mut()
            }
        }
    } else {
        tracing::warn!("evaluate_code_emotion: instance not initialized");
        std::ptr::null_mut()
    }
}

/// Evaluate text emotion
///
/// # Safety
///
/// - `text` must be a valid null-terminated UTF-8 string
/// - `text` must remain valid for the duration of this call
/// - Returns allocated string that MUST be freed with `emotional_coder_free_string()`
/// - This function is thread-safe
///
/// # Returns
///
/// - Pointer to JSON string on success (caller must free)
/// - null pointer if text is null, invalid UTF-8, or instance not initialized
#[no_mangle]
pub extern "C" fn evaluate_text_emotion(text: *const c_char) -> *mut c_char {
    if text.is_null() {
        tracing::error!("evaluate_text_emotion: null text pointer");
        return std::ptr::null_mut();
    }

    let c_str = unsafe { CStr::from_ptr(text) };
    let text_str = match c_str.to_str() {
        Ok(s) => s,
        Err(e) => {
            tracing::error!("evaluate_text_emotion: invalid UTF-8: {}", e);
            return std::ptr::null_mut();
        }
    };

    let guard = match EMOTIONAL_CODER_INSTANCE.lock() {
        Ok(g) => g,
        Err(e) => {
            tracing::error!("evaluate_text_emotion: mutex poisoned: {}", e);
            return std::ptr::null_mut();
        }
    };

    if let Some(ref coder) = *guard {
        let profile = create_emotional_profile(text_str);

        match serde_json::to_string(&profile) {
            Ok(json) => match CString::new(json) {
                Ok(cstring) => cstring.into_raw(),
                Err(e) => {
                    tracing::error!("evaluate_text_emotion: CString creation failed: {}", e);
                    std::ptr::null_mut()
                }
            },
            Err(e) => {
                tracing::error!("evaluate_text_emotion: JSON serialization failed: {}", e);
                std::ptr::null_mut()
            }
        }
    } else {
        tracing::warn!("evaluate_text_emotion: instance not initialized");
        std::ptr::null_mut()
    }
}

/// Free string returned by emotional coder FFI functions
///
/// # Safety
///
/// - `s` must be a pointer returned by `evaluate_code_emotion()` or `evaluate_text_emotion()`
/// - `s` must be freed exactly once
/// - After calling this, the C++ code MUST set the pointer to null
///
/// # Example (C++)
///
/// ```cpp
/// char* result = evaluate_code_emotion(code);
/// // ... use result ...
/// emotional_coder_free_string(result);
/// result = nullptr;  // CRITICAL: prevent double-free
/// ```
#[no_mangle]
pub extern "C" fn emotional_coder_free_string(s: *mut c_char) {
    if s.is_null() {
        return;
    }

    unsafe {
        let _ = CString::from_raw(s);
        // CString will be dropped here, freeing memory
    }
}

/// Create emotional profile for text/code
fn create_emotional_profile(input: &str) -> serde_json::Value {
    use serde_json::json;

    // Simple heuristic-based emotional analysis
    let lines = input.lines().count();
    let words = input.split_whitespace().count();
    let avg_line_length = if lines > 0 { input.len() / lines } else { 0 };

    // Calculate scores based on code characteristics
    let elegance_score = calculate_elegance(input, avg_line_length, lines);
    let clarity_score = calculate_clarity(input, words, lines);
    let anxiety_level = calculate_anxiety(input, avg_line_length);
    let satisfaction_score = (elegance_score + clarity_score) / 2.0;

    json!({
        "elegance": elegance_score,
        "clarity": clarity_score,
        "anxiety": anxiety_level,
        "satisfaction": satisfaction_score
    })
}

fn calculate_elegance(input: &str, avg_line_length: usize, lines: usize) -> f32 {
    // Heuristics for code elegance
    let mut score: f32 = 0.5;

    // Longer average line length might indicate complexity
    if avg_line_length > 80 {
        score -= 0.1;
    } else if avg_line_length > 60 {
        score += 0.05;
    }

    // More lines might indicate complexity
    if lines > 50 {
        score -= 0.1;
    } else if lines < 10 {
        score += 0.1;
    }

    // Check for patterns that might indicate elegance
    if input.contains("fn ") || input.contains("def ") {
        score += 0.1;
    }
    if input.contains("impl") {
        score += 0.05;
    }

    score.max(0.0_f32).min(1.0_f32)
}

fn calculate_clarity(input: &str, words: usize, lines: usize) -> f32 {
    let mut score: f32 = 0.5;

    // More comments increase clarity
    let comment_count = input.matches("//").count()
        + input.matches("#").count()
        + input.matches("/*").count()
        + input.matches("\"\"\"").count();
    if comment_count > 0 {
        score += 0.2;
    }

    // Reasonable line/word ratio
    if lines > 0 {
        let words_per_line = words as f32 / lines as f32;
        if words_per_line > 5.0 && words_per_line < 15.0 {
            score += 0.1;
        }
    }

    score.max(0.0_f32).min(1.0_f32)
}

fn calculate_anxiety(input: &str, avg_line_length: usize) -> f32 {
    let mut score: f32 = 0.3;

    // Very long lines might indicate complexity
    if avg_line_length > 120 {
        score += 0.3;
    } else if avg_line_length > 100 {
        score += 0.2;
    }

    // Nested structures increase anxiety
    let braces = input.matches('{').count() + input.matches('}').count();
    let parens = input.matches('(').count() + input.matches(')').count();
    let brackets = input.matches('[').count() + input.matches(']').count();

    let nesting_level = braces + parens + brackets;
    if nesting_level > 20 {
        score += 0.3;
    } else if nesting_level > 10 {
        score += 0.2;
    }

    score.max(0.0_f32).min(1.0_f32)
}

/// Cleanup emotional coder instance
///
/// # Safety
///
/// - This function is thread-safe (uses internal synchronization)
/// - After calling this, the instance can be re-initialized with `emotional_coder_init()`
#[no_mangle]
pub extern "C" fn emotional_coder_cleanup() {
    match EMOTIONAL_CODER_INSTANCE.lock() {
        Ok(mut guard) => {
            *guard = None;
            tracing::info!("✅ Emotional coder instance cleaned up");
        }
        Err(e) => {
            tracing::error!("emotional_coder_cleanup: mutex poisoned: {}", e);
        }
    }
}
