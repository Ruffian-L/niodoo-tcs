//! Real AI inference engine backed by vLLM with lexical emotion analysis fallback.

use crate::feeling_model::EmotionalAnalysis;
use crate::vllm_bridge::VLLMBridge;
use anyhow::{anyhow, Context, Result as AnyResult};
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::env;
use std::sync::Arc;
use tokio::runtime::Runtime;
use tracing::{info, warn};

const EMOTION_DIM: usize = 5; // joy, sadness, anger, fear, surprise

/// Configuration for the AI inference engine.
#[derive(Clone, Debug)]
pub struct AIInferenceConfig {
    pub endpoint: String,
    pub api_key: Option<String>,
    pub max_tokens: usize,
    pub temperature: f64,
    pub top_p: f64,
    pub preface_prompt: String,
    pub model_name: String,
}

impl Default for AIInferenceConfig {
    fn default() -> Self {
        let endpoint = format!(
            "http://{}:{}",
            env::var("VLLM_HOST").unwrap_or_else(|_| "127.0.0.1".to_string()),
            env::var("VLLM_PORT").unwrap_or_else(|_| "8000".to_string())
        );

        Self {
            endpoint,
            api_key: env::var("VLLM_API_KEY").ok(),
            max_tokens: env::var("NIODOO_MAX_TOKENS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(256),
            temperature: env::var("NIODOO_TEMPERATURE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.65),
            top_p: env::var("NIODOO_TOP_P")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.9),
            preface_prompt: env::var("NIODOO_AI_PREFACE").unwrap_or_else(|_| {
                "You are Niodoo, an empathetic cognitive architecture. Engage with care, name emotions, and offer grounded next steps.".to_string()
            }),
            model_name: env::var("NIODOO_MODEL_NAME")
                .unwrap_or_else(|_| "qwen2.5-7b-instruct".to_string()),
        }
    }
}

/// Result returned by the AI inference engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIInferenceResult {
    pub output: String,
    pub confidence: f64,
    pub model_name: String,
}

impl Default for AIInferenceResult {
    fn default() -> Self {
        Self {
            output: String::new(),
            confidence: 0.0,
            model_name: "unknown".to_string(),
        }
    }
}

/// Primary AI inference engine.
pub struct AIInferenceEngine {
    pub model_name: String,
    pub confidence: f64,
    config: AIInferenceConfig,
    mode: EngineMode,
}

impl Clone for AIInferenceEngine {
    fn clone(&self) -> Self {
        Self {
            model_name: self.model_name.clone(),
            confidence: self.confidence,
            config: self.config.clone(),
            mode: match &self.mode {
                EngineMode::Vllm(bridge) => EngineMode::Vllm(bridge.clone()),
                EngineMode::Offline(generator) => EngineMode::Offline(generator.clone()),
            },
        }
    }
}

enum EngineMode {
    Vllm(Arc<VLLMBridge>),
    Offline(OfflineGenerator),
}

impl EngineMode {
    fn generate(
        &self,
        prompt: &str,
        user_input: &str,
        config: &AIInferenceConfig,
    ) -> AnyResult<String> {
        match self {
            EngineMode::Vllm(bridge) => call_vllm_blocking(bridge.clone(), prompt, config),
            EngineMode::Offline(generator) => Ok(generator.generate(user_input)),
        }
    }
}

fn call_vllm_blocking(
    bridge: Arc<VLLMBridge>,
    prompt: &str,
    config: &AIInferenceConfig,
) -> AnyResult<String> {
    let fut = bridge.generate(prompt, config.max_tokens, config.temperature, config.top_p);

    if let Ok(handle) = tokio::runtime::Handle::try_current() {
        handle
            .block_on(fut)
            .map_err(|e| anyhow!("vLLM generation failed: {e}"))
    } else {
        let runtime = Runtime::new().context("failed to create Tokio runtime for AI inference")?;
        runtime
            .block_on(fut)
            .map_err(|e| anyhow!("vLLM generation failed: {e}"))
    }
}

impl AIInferenceEngine {
    /// Create an engine with environment-driven defaults.
    pub fn new_default() -> Self {
        Self::new_with_config(AIInferenceConfig::default())
    }

    /// Create an engine with explicit configuration.
    pub fn new_with_config(config: AIInferenceConfig) -> Self {
        match VLLMBridge::connect(&config.endpoint, config.api_key.clone()) {
            Ok(bridge) => {
                info!(
                    %config.endpoint,
                    model = %config.model_name,
                    "AI inference using live vLLM backend"
                );
                Self {
                    model_name: config.model_name.clone(),
                    confidence: 0.72,
                    config,
                    mode: EngineMode::Vllm(Arc::new(bridge)),
                }
            }
            Err(err) => {
                warn!(
                    error = %err,
                    "vLLM backend unavailable; falling back to lexical offline generator"
                );
                Self {
                    model_name: format!("{}-offline", config.model_name),
                    confidence: 0.48,
                    config,
                    mode: EngineMode::Offline(OfflineGenerator::default()),
                }
            }
        }
    }

    fn build_prompt(&self, input: &str) -> String {
        if input.trim().is_empty() {
            return format!(
                "{}\n\nThe user remained quiet. Offer a gentle check-in that keeps the door open.",
                self.config.preface_prompt
            );
        }

        format!(
            "{preface}\n\nUser shared:\n\"{summary}\"\n\nPlease respond as Niodoo: validate the emotion, mirror key language, and give one grounded next action.",
            preface = self.config.preface_prompt,
            summary = input.trim()
        )
    }

    /// Generate a response. Falls back to the offline summariser if vLLM is unavailable.
    pub fn generate(&self, input: &str) -> AIInferenceResult {
        let trimmed_input = input.trim();
        let prompt = self.build_prompt(trimmed_input);
        let generation = self
            .mode
            .generate(&prompt, trimmed_input, &self.config)
            .map(|text| text.trim().to_string())
            .unwrap_or_else(|err| {
                warn!(error = %err, "AI inference failed; using emergency offline synthesis");
                OfflineGenerator::default().generate(trimmed_input)
            });

        let confidence = calibrate_confidence(trimmed_input, &generation, self.confidence);

        AIInferenceResult {
            output: generation,
            confidence,
            model_name: self.model_name.clone(),
        }
    }

    /// Analyze the emotional tone of the input asynchronously.
    pub async fn detect_emotion(&self, input: &str) -> AnyResult<EmotionalAnalysis> {
        let text = input.to_string();
        let confidence = self.confidence as f32;

        let join_result = tokio::task::spawn_blocking(move || -> AnyResult<EmotionalAnalysis> {
            let lexicon = &*EMOTION_LEXICON;
            let mut analysis = lexicon.analyze(&text);
            lexicon.scale_in_place(&mut analysis, confidence);
            lexicon.finalize(&mut analysis);
            Ok(analysis)
        })
        .await;

        match join_result {
            Ok(inner) => inner,
            Err(join_err) => Err(anyhow!(join_err)),
        }
    }
}

#[derive(Clone, Default)]
struct OfflineGenerator;

impl OfflineGenerator {
    fn generate(&self, user_input: &str) -> String {
        let content = user_input.trim();
        if content.is_empty() {
            return "I'm here and paying attention. Share whatever feels safe and we'll take it one breath at a time.".to_string();
        }

        let sentences = split_sentences(content);
        let tokens = tokenize(content);
        let frequencies = term_frequencies(&tokens);

        let mut sentence_scores: Vec<(usize, f64)> = sentences
            .iter()
            .enumerate()
            .map(|(idx, sentence)| {
                let words = tokenize(sentence);
                let score: f64 = words
                    .iter()
                    .map(|word| frequencies.get(word).copied().unwrap_or(0.0))
                    .sum();
                (idx, score + 0.1)
            })
            .collect();

        sentence_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        let mut selected_indices: Vec<usize> = sentence_scores
            .into_iter()
            .take(sentences.len().min(3))
            .map(|(idx, _)| idx)
            .collect();
        selected_indices.sort_unstable();

        let summary_lines: Vec<String> = selected_indices
            .into_iter()
            .map(|idx| format!("• {}", sentences[idx].trim()))
            .collect();

        let mut keywords: Vec<(String, f64)> = frequencies.into_iter().collect();
        keywords.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        let keyword_list: Vec<String> =
            keywords.into_iter().take(5).map(|(word, _)| word).collect();

        let emotional_read = EMOTION_LEXICON.analyze(content);
        let tone = if emotional_read.emotional_intensity > 0.6 {
            "I can feel how charged this is."
        } else if emotional_read.emotional_intensity > 0.3 {
            "There's a steady pulse here and it's worth honoring."
        } else {
            "This feels gentle but still important."
        };

        let action_plan = build_action_plan(&emotional_read, keyword_list.first());

        let header = if keyword_list.is_empty() {
            "Here's what I heard:".to_string()
        } else {
            format!("Key themes I heard: {}", keyword_list.join(", "))
        };

        let summary = if summary_lines.is_empty() {
            "• I want to acknowledge that sharing this is a big step.".to_string()
        } else {
            summary_lines.join("\n")
        };

        format!(
            "{tone}\n{header}\n{summary}\n\nAction idea: {action}",
            tone = tone,
            header = header,
            summary = summary,
            action = action_plan
        )
    }
}

fn build_action_plan(analysis: &EmotionalAnalysis, keyword: Option<&String>) -> String {
    let anchor = keyword
        .map(|k| format!(" around \"{}\"", k))
        .unwrap_or_default();

    if analysis.sadness > 0.35 {
        format!(
            "Name one comfort you can reach for in the next hour and let someone trusted know you need it{}.",
            anchor
        )
    } else if analysis.anger > 0.35 {
        format!(
            "Channel that energy into a boundary statement—write it down, even if you never send it{}.",
            anchor
        )
    } else if analysis.fear > 0.35 {
        format!(
            "Map the smallest safe experiment you can try; prove to yourself that movement is possible{}.",
            anchor
        )
    } else if analysis.joy > 0.4 {
        format!(
            "Immortalize this spark—capture what nourished you so we can replay it later{}.",
            anchor
        )
    } else {
        format!(
            "Commit to one five-minute step that nudges the situation forward and note how it shifts your body state{}.",
            anchor
        )
    }
}

fn calibrate_confidence(user_input: &str, response: &str, base: f64) -> f64 {
    let prompt_len = user_input.split_whitespace().count() as f64;
    let response_len = response.split_whitespace().count() as f64;
    let ratio = if prompt_len < f64::EPSILON {
        1.0
    } else {
        (response_len / prompt_len).clamp(0.25, 2.5)
    };

    let lexical_diversity = {
        let tokens = tokenize(response);
        if tokens.is_empty() {
            0.0
        } else {
            let unique: HashSet<_> = tokens.into_iter().collect();
            unique.len() as f64 / response_len.max(1.0)
        }
    };

    let overlap = {
        let prompt_tokens: HashSet<_> = tokenize(user_input).into_iter().collect();
        if prompt_tokens.is_empty() {
            0.5
        } else {
            let response_tokens: HashSet<_> = tokenize(response).into_iter().collect();
            prompt_tokens.intersection(&response_tokens).count() as f64 / prompt_tokens.len() as f64
        }
    };

    let punctuation_bonus = if response.contains('!') { 0.03 } else { 0.0 }
        - if response.trim_end().ends_with('?') {
            0.04
        } else {
            0.0
        };

    (0.35 + base * 0.3 + ratio * 0.1 + lexical_diversity * 0.1 + overlap * 0.12 + punctuation_bonus)
        .clamp(0.1, 0.97)
}

fn split_sentences(text: &str) -> Vec<&str> {
    text.split(|c| matches!(c, '.' | '!' | '?' | ';'))
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect()
}

fn term_frequencies(tokens: &[String]) -> HashMap<String, f64> {
    if tokens.is_empty() {
        return HashMap::new();
    }

    let mut counts: HashMap<String, usize> = HashMap::new();
    for token in tokens {
        *counts.entry(token.clone()).or_insert(0) += 1;
    }

    let total = tokens.len() as f64;
    counts
        .into_iter()
        .map(|(token, count)| (token, count as f64 / total))
        .collect()
}

fn tokenize(text: &str) -> Vec<String> {
    text.split_whitespace()
        .filter_map(normalize_token)
        .collect()
}

fn normalize_token(token: &str) -> Option<String> {
    let cleaned = token
        .trim_matches(|c: char| !c.is_alphanumeric())
        .to_lowercase();
    if cleaned.is_empty() {
        None
    } else {
        Some(cleaned)
    }
}

struct EmotionLexicon {
    weights: HashMap<&'static str, [f32; EMOTION_DIM]>,
    negators: HashSet<&'static str>,
    amplifiers: HashSet<&'static str>,
}

impl EmotionLexicon {
    fn analyze(&self, text: &str) -> EmotionalAnalysis {
        let tokens = tokenize(text);
        if tokens.is_empty() {
            return EmotionalAnalysis {
                joy: 0.0,
                sadness: 0.0,
                anger: 0.0,
                fear: 0.0,
                surprise: 0.0,
                emotional_intensity: 0.0,
                dominant_emotion: "neutral".to_string(),
            };
        }

        let mut totals = [0.0f32; EMOTION_DIM];
        let mut activations = 0usize;

        for (idx, token) in tokens.iter().enumerate() {
            if let Some(weights) = self.weights.get(token.as_str()) {
                let mut scale = 1.0f32;

                if idx > 0 && self.negators.contains(tokens[idx - 1].as_str()) {
                    scale *= -1.0;
                }
                if idx > 0 && self.amplifiers.contains(tokens[idx - 1].as_str()) {
                    scale *= 1.4;
                }
                if idx > 1 && self.amplifiers.contains(tokens[idx - 2].as_str()) {
                    scale *= 1.2;
                }

                for i in 0..EMOTION_DIM {
                    totals[i] += weights[i] * scale;
                }
                activations += 1;
            }
        }

        let positive_totals: Vec<f32> = totals.iter().map(|v| v.max(0.0)).collect();
        let sum_positive: f32 = positive_totals.iter().sum();

        let mut analysis = EmotionalAnalysis {
            joy: 0.0,
            sadness: 0.0,
            anger: 0.0,
            fear: 0.0,
            surprise: 0.0,
            emotional_intensity: if activations == 0 {
                0.0
            } else {
                (sum_positive / activations as f32).clamp(0.0, 1.0)
            },
            dominant_emotion: "neutral".to_string(),
        };

        if sum_positive > 0.0 {
            analysis.joy = positive_totals[0] / sum_positive;
            analysis.sadness = positive_totals[1] / sum_positive;
            analysis.anger = positive_totals[2] / sum_positive;
            analysis.fear = positive_totals[3] / sum_positive;
            analysis.surprise = positive_totals[4] / sum_positive;
        }

        analysis
    }

    fn scale_in_place(&self, analysis: &mut EmotionalAnalysis, confidence: f32) {
        let scale = confidence.clamp(0.25, 1.35);
        analysis.joy = (analysis.joy * scale).clamp(0.0, 1.0);
        analysis.sadness = (analysis.sadness * scale).clamp(0.0, 1.0);
        analysis.anger = (analysis.anger * scale).clamp(0.0, 1.0);
        analysis.fear = (analysis.fear * scale).clamp(0.0, 1.0);
        analysis.surprise = (analysis.surprise * scale).clamp(0.0, 1.0);
        analysis.emotional_intensity = (analysis.emotional_intensity * scale).clamp(0.0, 1.0);
    }

    fn finalize(&self, analysis: &mut EmotionalAnalysis) {
        let mut components = [
            ("joy", analysis.joy),
            ("sadness", analysis.sadness),
            ("anger", analysis.anger),
            ("fear", analysis.fear),
            ("surprise", analysis.surprise),
        ];

        let sum: f32 = components.iter().map(|(_, value)| value).sum();
        if sum > 1.0 && sum.is_finite() {
            let normaliser = 1.0 / sum;
            for (_, value) in &mut components {
                *value *= normaliser;
            }
            analysis.joy = components[0].1;
            analysis.sadness = components[1].1;
            analysis.anger = components[2].1;
            analysis.fear = components[3].1;
            analysis.surprise = components[4].1;
        }

        if let Some((label, value)) = components
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
        {
            analysis.dominant_emotion = if *value > 0.05 {
                (*label).to_string()
            } else {
                "neutral".to_string()
            };
        }
    }
}

static EMOTION_LEXICON: Lazy<EmotionLexicon> = Lazy::new(|| EmotionLexicon {
    weights: {
        let mut map = HashMap::new();
        map.insert("joy", [0.9, 0.0, 0.0, 0.0, 0.3]);
        map.insert("happy", [0.85, 0.0, 0.0, 0.0, 0.2]);
        map.insert("grateful", [0.8, 0.0, 0.0, 0.0, 0.1]);
        map.insert("calm", [0.6, 0.0, 0.0, 0.1, 0.0]);
        map.insert("relieved", [0.7, 0.0, 0.0, 0.0, 0.2]);
        map.insert("sad", [0.0, 0.9, 0.0, 0.0, 0.1]);
        map.insert("hurt", [0.0, 0.85, 0.0, 0.0, 0.0]);
        map.insert("alone", [0.0, 0.8, 0.0, 0.1, 0.0]);
        map.insert("mourning", [0.0, 0.95, 0.0, 0.0, 0.0]);
        map.insert("angry", [0.0, 0.0, 0.9, 0.0, 0.1]);
        map.insert("furious", [0.0, 0.0, 1.0, 0.0, 0.1]);
        map.insert("frustrated", [0.0, 0.0, 0.8, 0.0, 0.1]);
        map.insert("resentful", [0.0, 0.0, 0.85, 0.0, 0.0]);
        map.insert("fear", [0.0, 0.0, 0.0, 0.95, 0.0]);
        map.insert("anxious", [0.0, 0.1, 0.0, 0.9, 0.1]);
        map.insert("worried", [0.0, 0.2, 0.0, 0.85, 0.1]);
        map.insert("panic", [0.0, 0.0, 0.1, 1.0, 0.1]);
        map.insert("surprised", [0.2, 0.0, 0.0, 0.0, 0.9]);
        map.insert("shocked", [0.0, 0.0, 0.0, 0.1, 1.0]);
        map.insert("curious", [0.4, 0.0, 0.0, 0.0, 0.8]);
        map.insert("delighted", [1.0, 0.0, 0.0, 0.0, 0.4]);
        map.insert("peaceful", [0.75, 0.0, 0.0, 0.0, 0.1]);
        map.insert("hopeful", [0.8, 0.0, 0.0, 0.0, 0.2]);
        map.insert("disappointed", [0.0, 0.8, 0.0, 0.1, 0.0]);
        map.insert("bitter", [0.0, 0.4, 0.8, 0.0, 0.0]);
        map.insert("betrayed", [0.0, 0.5, 0.85, 0.0, 0.0]);
        map.insert("ashamed", [0.0, 0.7, 0.1, 0.5, 0.0]);
        map.insert("lonely", [0.0, 0.85, 0.0, 0.2, 0.0]);
        map.insert("overwhelmed", [0.0, 0.3, 0.1, 0.85, 0.2]);
        map.insert("excited", [0.85, 0.0, 0.0, 0.0, 0.95]);
        map.insert("inspired", [0.9, 0.0, 0.0, 0.0, 0.5]);
        map
    },
    negators: {
        let items = ["not", "never", "no", "hardly", "rarely", "without"];
        items.into_iter().collect()
    },
    amplifiers: {
        let items = ["very", "deeply", "extremely", "so", "incredibly", "truly"];
        items.into_iter().collect()
    },
});

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offline_generator_handles_empty_input() {
        let generator = OfflineGenerator::default();
        let output = generator.generate("");
        assert!(output.contains("I'm here"));
    }

    #[test]
    fn emotion_lexicon_detects_joy() {
        let analysis = EMOTION_LEXICON.analyze("I feel so happy and inspired today");
        assert!(analysis.joy > analysis.sadness);
        assert_eq!(analysis.dominant_emotion, "joy");
    }
}
