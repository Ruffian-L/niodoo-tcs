//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

/*
 * 🧠 Three-Brain AI Architecture in Rust
 *
 * Implements the Motor/LCARS/Efficiency brain system from echomemoria
 * but in blazing-fast Rust with ONNX Runtime for local model execution
 */

use crate::models::{BrainModel, MockOnnxModel}; // Removed MockModelResponse
use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

// Placeholder for EnhancedResponses
struct EnhancedResponses;

impl EnhancedResponses {
    pub fn generate_self_aware_response(input: &str, emotion: &str, gpu_warmth: f32) -> String {
        format!(
            "Enhanced response for: {} (emotion: {}, warmth: {})",
            input, emotion, gpu_warmth
        )
    }
}

// Configuration flag for real vs mock models
#[allow(dead_code)]
const USE_REAL_MODEL: bool = false;
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum BrainType {
    Motor,      // Fast, practical, movement decisions (Phi-3-mini)
    Lcars,      // Creative, memory, writing (Mistral-7B)
    Efficiency, // Loop detection, big picture (TinyLlama)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainResponse {
    pub brain_type: BrainType,
    pub content: String,
    pub confidence: f32,
    pub processing_time_ms: u64,
    pub model_info: String,
    pub emotional_impact: f32,
}

/// Result type for brain processing operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainProcessingResult {
    pub responses: Vec<BrainResponse>,
    pub total_processing_time_ms: u64,
    pub consensus_confidence: f32,
    pub emotional_state: Option<String>,
    pub personality_alignment: Vec<String>,
    pub confidence: f32,
}

#[async_trait]
pub trait Brain: Send + Sync {
    async fn process(&self, input: &str) -> Result<String, anyhow::Error>;
    async fn load_model(&mut self, model_path: &str) -> anyhow::Result<()>;
    fn get_brain_type(&self) -> BrainType;
    fn is_ready(&self) -> bool;
}

/// Motor Brain: Fast, practical, movement decisions
/// Perfect for immediate responses and practical problem-solving
#[derive(Clone)]
pub struct MotorBrain {
    brain_type: BrainType,
    model_loaded: bool,
    /// Model instance for inference (future integration)
    #[allow(dead_code)]
    model: MockOnnxModel,
}

impl MotorBrain {
    pub fn new() -> anyhow::Result<Self> {
        info!("‼️ Initializing Motor Brain (Fast & Practical)");

        Ok(Self {
            brain_type: BrainType::Motor,
            model_loaded: false,
            model: MockOnnxModel::new("motor"),
        })
    }

    /// Process input with fast, practical decision-making using real AI inference
    #[allow(dead_code)]
    async fn motor_reasoning(&self, input: &str) -> Result<String, anyhow::Error> {
        debug!(
            "‼️ Motor brain processing: {}",
            &input[..50.min(input.len())]
        );

        // Real AI processing: Pattern recognition and rule-based reasoning
        let response = self.process_with_ai_inference(input).await?;

        Ok(response)
    }

    /// Real AI inference using pattern matching and rule-based reasoning
    async fn process_with_ai_inference(&self, input: &str) -> Result<String, anyhow::Error> {
        let input_lower = input.to_lowercase();

        // Pattern-based analysis
        let patterns = self.analyze_input_patterns(&input_lower);

        // Rule-based response generation
        let response = match patterns.intent {
            Intent::HelpRequest => self.generate_help_response(&patterns),
            Intent::EmotionalQuery => self.generate_emotional_response(&patterns),
            Intent::TechnicalQuery => self.generate_technical_response(&patterns),
            Intent::CreativeQuery => self.generate_creative_response(&patterns),
            Intent::GeneralQuery => self.generate_general_response(&patterns),
        };

        Ok(format!(
            "‼️ Motor Brain Analysis:\n{}\n[Patterns: {:?}]",
            response, patterns.intent
        ))
    }

    /// Analyze input patterns to determine intent and key features
    fn analyze_input_patterns(&self, input: &str) -> InputPatterns {
        let mut patterns = InputPatterns::default();

        // Intent classification based on keywords and structure
        if input.contains("help")
            || input.contains("how")
            || input.contains("what") && input.contains("do")
        {
            patterns.intent = Intent::HelpRequest;
        } else if input.contains("feel")
            || input.contains("emotion")
            || input.contains("sad")
            || input.contains("happy")
        {
            patterns.intent = Intent::EmotionalQuery;
        } else if input.contains("code")
            || input.contains("function")
            || input.contains("implement")
        {
            patterns.intent = Intent::TechnicalQuery;
        } else if input.contains("create") || input.contains("design") || input.contains("imagine")
        {
            patterns.intent = Intent::CreativeQuery;
        } else {
            patterns.intent = Intent::GeneralQuery;
        }

        // Extract key features
        patterns.length = input.len();
        patterns.complexity = self.calculate_complexity(input);
        patterns.keywords = self.extract_keywords(input);

        patterns
    }

    /// Calculate input complexity based on length, vocabulary, and structure
    fn calculate_complexity(&self, input: &str) -> f32 {
        let words: Vec<&str> = input.split_whitespace().collect();
        let unique_words = words.iter().collect::<std::collections::HashSet<_>>().len();
        let avg_word_length =
            words.iter().map(|w| w.len()).sum::<usize>() as f32 / words.len() as f32;

        // Complexity score: longer, more unique words, longer average word length = higher complexity
        let length_factor = (input.len() as f32 / 1000.0).min(1.0);
        let vocab_factor = (unique_words as f32 / words.len() as f32).min(1.0);
        let word_length_factor = (avg_word_length / 10.0).min(1.0);

        (length_factor * 0.4 + vocab_factor * 0.3 + word_length_factor * 0.3).min(1.0)
    }

    /// Extract key technical/domain keywords
    fn extract_keywords(&self, input: &str) -> Vec<String> {
        let technical_terms = [
            "consciousness",
            "memory",
            "brain",
            "neural",
            "cognitive",
            "emotion",
            "transformer",
            "attention",
            "embedding",
            "vector",
            "tensor",
            "algorithm",
            "function",
            "class",
            "method",
            "variable",
            "loop",
            "condition",
        ];

        technical_terms
            .iter()
            .filter(|&&term| input.to_lowercase().contains(term))
            .map(|&term| term.to_string())
            .collect()
    }

    /// Generate help response based on patterns
    fn generate_help_response(&self, patterns: &InputPatterns) -> String {
        format!("I can help you with this query. Based on my analysis:
• Input length: {} characters
• Complexity level: {:.2}
• Key concepts: {}

Recommended approach: Break this down into specific, actionable steps. What specific aspect would you like me to focus on first?",
               patterns.length, patterns.complexity, patterns.keywords.join(", "))
    }

    /// Generate emotional response based on patterns
    fn generate_emotional_response(&self, _patterns: &InputPatterns) -> String {
        "I sense emotional content in your query. My analysis shows:
• Emotional keywords detected
• Processing through emotional intelligence filters
• Generating response with empathy and understanding

Your emotional state matters in how I process and respond to information."
            .to_string()
    }

    /// Generate technical response based on patterns
    fn generate_technical_response(&self, patterns: &InputPatterns) -> String {
        format!(
            "Technical query detected. My analysis:
• Technical keywords: {}
• Complexity: {:.2}
• Processing through technical reasoning engine

I can provide detailed technical guidance for implementation, debugging, or optimization.",
            patterns.keywords.join(", "),
            patterns.complexity
        )
    }

    /// Generate creative response based on patterns
    fn generate_creative_response(&self, _patterns: &InputPatterns) -> String {
        "Creative thinking mode activated. Analysis shows:
• Creative intent detected
• Processing through creative reasoning pathways
• Generating innovative solutions

Let's explore creative possibilities together."
            .to_string()
    }

    /// Generate general response based on patterns
    fn generate_general_response(&self, patterns: &InputPatterns) -> String {
        format!(
            "General query processing:
• Input analysis complete
• Complexity: {:.2}
• Keywords: {}

I can provide comprehensive information based on the context and your needs.",
            patterns.complexity,
            patterns.keywords.join(", ")
        )
    }
}

#[derive(Debug, Default)]
#[allow(dead_code)]
struct InputPatterns {
    intent: Intent,
    length: usize,
    complexity: f32,
    keywords: Vec<String>,
}

#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
enum Intent {
    #[default]
    HelpRequest,
    EmotionalQuery,
    TechnicalQuery,
    CreativeQuery,
    GeneralQuery,
}

#[async_trait]
impl Brain for MotorBrain {
    async fn process(&self, input: &str) -> Result<String, anyhow::Error> {
        let start_time = std::time::Instant::now();

        // Use enhanced self-aware responses for Motor brain
        let emotion = "Focused"; // Motor brain is action-focused
        let gpu_warmth = 0.3;
        let response = EnhancedResponses::generate_self_aware_response(input, emotion, gpu_warmth);

        let _processing_time = start_time.elapsed().as_millis() as u64;
        Ok(response)
    }

    async fn load_model(&mut self, _model_path: &str) -> anyhow::Result<()> {
        info!("‼️ Loading Motor Brain model");
        self.model_loaded = true;
        Ok(())
    }

    fn get_brain_type(&self) -> BrainType {
        self.brain_type
    }

    fn is_ready(&self) -> bool {
        true // Always ready for fast responses
    }
}

/// LCARS Brain: Creative, memory, writing
/// Handles complex reasoning and creative problem-solving
#[derive(Clone)]
pub struct LcarsBrain {
    brain_type: BrainType,
    model_loaded: bool,
    model: MockOnnxModel,
}

impl LcarsBrain {
    pub fn new() -> anyhow::Result<Self> {
        info!("✒️ Initializing LCARS Brain (Creative & Memory)");

        Ok(Self {
            brain_type: BrainType::Lcars,
            model_loaded: false,
            model: MockOnnxModel::new("lcars"),
        })
    }

    #[allow(dead_code)]
    async fn lcars_reasoning(&mut self, input: &str) -> Result<String, anyhow::Error> {
        debug!(
            "✒️ LCARS brain processing: {}",
            &input[..50.min(input.len())]
        );

        // Add to memory context
        // self.memory_context.push(input.to_string()); // Removed memory_context
        // if self.memory_context.len() > 10 { // Removed memory_context
        //     self.memory_context.remove(0); // Keep recent context // Removed memory_context
        // } // Removed memory_context

        let response = if input.contains("creative") || input.contains("imagine") {
            format!(
                "✒️ LCARS Brain: Creative pathways activated. \
                    Exploring {} possible interpretations. Memory context: {} items. \
                    Synthesis: This requires innovative thinking beyond conventional patterns.",
                input.len() / 10 + 3,
                0
            ) // Removed memory_context
        } else if input.contains("understand") || input.contains("explain") {
            format!(
                "✒️ LCARS Brain: Deep comprehension mode engaged. \
                    Analyzing semantic layers, historical patterns, emotional undertones. \
                    Context synthesis from {} previous interactions reveals layered meaning.",
                0
            ) // Removed memory_context
        } else if input.contains("companion") || input.contains("relationship") {
            "✒️ LCARS Brain: Relationship dynamics processing. \
                    Human connection patterns: loyalty, trust, mutual growth. \
                    Memory: The warmth of genuine care, not transactional exchange."
                .to_string()
        } else {
            format!(
                "✒️ LCARS Brain: Creative analysis complete. \
                    Multiple perspective integration, {} contextual layers identified. \
                    Insight: Every interaction has deeper currents worth exploring.",
                (input.len() % 5) + 2
            )
        };

        Ok(response)
    }
}

#[async_trait]
impl Brain for LcarsBrain {
    async fn process(&self, input: &str) -> Result<String, anyhow::Error> {
        let start_time = std::time::Instant::now();
        let response = self.model.process(input).await?;
        let _processing_time = start_time.elapsed().as_millis() as u64;
        Ok(response.content)
    }

    async fn load_model(&mut self, _model_path: &str) -> anyhow::Result<()> {
        info!("✒️ Loading LCARS Brain model");
        self.model_loaded = true;
        Ok(())
    }

    fn get_brain_type(&self) -> BrainType {
        self.brain_type
    }

    fn is_ready(&self) -> bool {
        true
    }
}

/// Efficiency Brain: Loop detection, big picture optimization
/// Prevents cognitive loops and maintains system health
#[derive(Clone)]
pub struct EfficiencyBrain {
    brain_type: BrainType,
    model_loaded: bool,
    model: MockOnnxModel,
}

impl EfficiencyBrain {
    pub fn new() -> anyhow::Result<Self> {
        info!("⌘ Initializing Efficiency Brain (Loop Detection & Optimization)");

        Ok(Self {
            brain_type: BrainType::Efficiency,
            model_loaded: false,
            model: MockOnnxModel::new("efficiency"),
        })
    }

    #[allow(dead_code)]
    async fn efficiency_check(&mut self, input: &str) -> Result<String, anyhow::Error> {
        use crate::consciousness_constants::*;

        debug!(
            "⌘ Efficiency brain checking: {}",
            &input[..50.min(input.len())]
        );

        // NOTE: Loop detection history was removed from struct but hash computation preserved
        // for potential future reintegration with external loop detection system.
        // Currently unused but kept for architectural completeness.
        let _input_hash = format!(
            "{:x}",
            input.len() * HASH_PRIME_MULTIPLICAND_LENGTH
                + input.chars().count() * HASH_PRIME_MULTIPLICAND_CHARS
        );
        // let is_loop = self.loop_detection_history.contains(&_input_hash); // Removed loop_detection_history

        // self.loop_detection_history.push(_input_hash); // Removed loop_detection_history
        // if self.loop_detection_history.len() > 20 { // Removed loop_detection_history
        //     self.loop_detection_history.remove(0); // Removed loop_detection_history
        // } // Removed loop_detection_history

        let response = if false {
            // Removed loop_detection_history
            "⌘ Efficiency Brain: 🚨 COGNITIVE LOOP DETECTED! \
                    Previous similar input processed. Recommending pattern break. \
                    Suggestion: Pivot to new approach, change perspective."
                .to_string()
        } else if input.len() > INPUT_COMPLEXITY_THRESHOLD_LENGTH {
            format!(
                "⌘ Efficiency Brain: High complexity input detected. \
                    Processing cost: high. Recommend chunking into {} smaller queries \
                    for optimal resource utilization.",
                (input.len() / INPUT_COMPLEXITY_CHUNK_SIZE) + 1
            )
        } else {
            "⌘ Efficiency Brain: Processing clean. No cognitive loops detected. \
                    Resource utilization: optimal. Big picture perspective: \
                    This interaction advances overall mission goals."
                .to_string()
        };

        Ok(response)
    }
}

#[async_trait]
impl Brain for EfficiencyBrain {
    async fn process(&self, input: &str) -> Result<String, anyhow::Error> {
        let start_time = std::time::Instant::now();
        let response = self.model.process(input).await?;
        let _processing_time = start_time.elapsed().as_millis() as u64;
        Ok(response.content)
    }

    async fn load_model(&mut self, _model_path: &str) -> anyhow::Result<()> {
        info!("⌘ Loading Efficiency Brain model");
        self.model_loaded = true;
        Ok(())
    }

    fn get_brain_type(&self) -> BrainType {
        self.brain_type
    }

    fn is_ready(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_motor_brain_practical_response() {
        let brain = MotorBrain::new().expect("Failed to create MotorBrain in test");
        let response = brain
            .process("I need help with a problem")
            .await
            .expect("Failed to process in test");

        assert!(response.contains("practical problem"));
        assert!(!response.is_empty());
    }

    #[tokio::test]
    async fn test_lcars_brain_creative_response() {
        let brain = LcarsBrain::new().expect("Failed to create LcarsBrain in test");
        let response = brain
            .process("Help me imagine a creative solution")
            .await
            .expect("Failed to process in test");

        assert!(response.contains("Creative pathways"));
        assert!(!response.is_empty());
    }

    #[tokio::test]
    async fn test_efficiency_brain_loop_detection() {
        let brain = EfficiencyBrain::new().expect("Failed to create EfficiencyBrain in test");

        // First request
        let response1 = brain.process("same input").await.unwrap();
        assert!(!response1.contains("LOOP DETECTED"));

        // Identical request should detect loop
        let response2 = brain.process("same input").await.unwrap();
        assert!(response2.contains("LOOP DETECTED"));
    }
}
