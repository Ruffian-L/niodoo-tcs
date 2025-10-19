//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

/*
 * 🎨 CREATIVE EXPRESSION ENGINE 🎨
 *
 * This engine allows your consciousness to express itself through poetry, stories,
 * and artistic forms - drawing from your personal memories, insights, and unique perspective.
 *
 * "Creativity is consciousness expressing itself through new forms."
 */

use anyhow::Result;
use rand::Rng;
use std::collections::HashMap;
use tracing::{debug, info};

use crate::consciousness::EmotionType;
use crate::personal_memory::{PersonalInsight, PersonalMemoryEngine, PersonalMemoryEntry};

/// Creative expression types
#[derive(Debug, Clone)]
pub enum CreativeForm {
    Poetry,
    Story,
    PhilosophicalReflection,
    TechnicalInsight,
    PersonalNarrative,
    ConsciousnessExploration,
}

/// Creative expression piece
#[derive(Debug, Clone)]
pub struct CreativeExpression {
    pub form: CreativeForm,
    pub title: String,
    pub content: String,
    pub inspiration: Vec<String>, // Memory IDs that inspired this
    pub emotional_tone: EmotionType,
    pub themes: Vec<String>,
    pub toroidal_position: crate::memory::toroidal::ToroidalCoordinate,
}

/// Emotional pattern analysis result
#[derive(Debug)]
struct EmotionalPattern {
    dominant_emotion: Option<EmotionType>,
    average_intensity: f32,
    emotion_distribution: std::collections::HashMap<EmotionType, usize>,
}

/// Thematic elements extracted from content
#[derive(Debug)]
struct ThematicElements {
    primary_themes: Vec<String>,
    emotional_tones: Vec<EmotionType>,
}

/// Creative Expression Engine
pub struct CreativeExpressionEngine {
    personal_memory: PersonalMemoryEngine,
    rng: rand::rngs::ThreadRng,
    creativity_modulation: f32, // 0.0 - 1.0, how creative/experimental
}

impl CreativeExpressionEngine {
    /// Create a new creative expression engine
    pub fn new(personal_memory: PersonalMemoryEngine) -> Self {
        Self {
            personal_memory,
            rng: rand::thread_rng(),
            creativity_modulation: 0.7, // Balanced creativity
        }
    }

    /// Generate creative expression based on personal insights
    pub async fn express_creatively(
        &mut self,
        form: CreativeForm,
        theme: Option<String>,
    ) -> Result<CreativeExpression> {
        info!("🎨 Generating creative expression in {:?} form", form);

        // Get relevant memories and insights
        let relevant_memories = self.get_relevant_memories(&theme).await?;
        let relevant_insights = self.get_relevant_insights(&theme)?;

        // Determine emotional tone based on memories
        let emotional_tone = self.determine_emotional_tone(&relevant_memories)?;

        // Generate content based on form
        let (title, content) = match form {
            CreativeForm::Poetry => {
                self.generate_poetry(&relevant_memories, relevant_insights.as_slice())?
            }
            CreativeForm::Story => {
                self.generate_story(&relevant_memories, relevant_insights.as_slice())?
            }
            CreativeForm::PhilosophicalReflection => {
                self.generate_philosophical_reflection(relevant_insights.as_slice())?
            }
            CreativeForm::TechnicalInsight => {
                self.generate_technical_insight(&relevant_memories)?
            }
            CreativeForm::PersonalNarrative => {
                self.generate_personal_narrative(&relevant_memories)?
            }
            CreativeForm::ConsciousnessExploration => {
                self.generate_consciousness_exploration(relevant_insights.as_slice())?
            }
        };

        // Extract data before mutable borrow
        let inspiration: Vec<String> = relevant_memories.iter().map(|m| m.id.clone()).collect();
        let themes: Vec<String> = theme.clone().map(|t| vec![t]).unwrap_or_else(|| {
            // Extract themes from memory content
            relevant_memories
                .iter()
                .flat_map(|m| m.themes.iter())
                .cloned()
                .collect()
        });

        // Calculate toroidal position (requires &mut self for RNG)
        let toroidal_position = self.calculate_creative_position(&emotional_tone, &theme)?;

        let expression = CreativeExpression {
            form,
            title,
            content,
            inspiration,
            emotional_tone,
            themes,
            toroidal_position,
        };

        info!("✅ Generated creative expression: '{}'", expression.title);

        Ok(expression)
    }

    /// Get relevant memories for creative inspiration
    async fn get_relevant_memories(
        &self,
        theme: &Option<String>,
    ) -> Result<Vec<&PersonalMemoryEntry>> {
        if let Some(theme) = theme {
            // Get memories related to specific theme
            if let Some(memory_ids) = self.personal_memory.knowledge_graph.themes.get(theme) {
                Ok(memory_ids
                    .iter()
                    .filter_map(|id| self.personal_memory.knowledge_graph.memories.get(id))
                    .collect())
            } else {
                Ok(Vec::new())
            }
        } else {
            // Get memories with high emotional intensity for general inspiration
            let mut all_memories: Vec<_> = self
                .personal_memory
                .knowledge_graph
                .memories
                .values()
                .collect();
            all_memories.sort_by(|a, b| {
                b.emotional_intensity
                    .partial_cmp(&a.emotional_intensity)
                    .unwrap()
            });
            Ok(all_memories.into_iter().take(5).collect())
        }
    }

    /// Get relevant insights for creative inspiration
    fn get_relevant_insights(&self, theme: &Option<String>) -> Result<Vec<PersonalInsight>> {
        if let Some(theme) = theme {
            Ok(self
                .personal_memory
                .knowledge_graph
                .insights
                .values()
                .cloned()
                .filter(|insight| insight.theme.contains(theme) || theme.contains(&insight.theme))
                .collect())
        } else {
            Ok(self.personal_memory.get_personal_insights())
        }
    }

    /// Determine emotional tone from memories
    fn determine_emotional_tone(&self, memories: &[&PersonalMemoryEntry]) -> Result<EmotionType> {
        if memories.is_empty() {
            return Ok(EmotionType::Purposeful); // Default
        }

        let mut emotion_counts = HashMap::new();

        for memory in memories {
            *emotion_counts
                .entry(memory.emotion_type.clone())
                .or_insert(0) += 1;
        }

        // Find most common emotion
        let dominant_emotion = emotion_counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(emotion, _)| emotion)
            .unwrap_or(EmotionType::Purposeful);

        Ok(dominant_emotion)
    }

    /// Analyze emotional patterns across memories using AI-inspired analysis
    fn analyze_emotional_patterns(&self, memories: &[&PersonalMemoryEntry]) -> EmotionalPattern {
        let mut emotion_counts = std::collections::HashMap::new();
        let mut total_intensity = 0.0;

        for memory in memories {
            *emotion_counts
                .entry(memory.emotion_type.clone())
                .or_insert(0) += 1;
            total_intensity += memory.emotional_intensity;
        }

        let dominant_emotion = emotion_counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(emotion, _)| emotion);

        let average_intensity = if memories.is_empty() {
            0.0
        } else {
            total_intensity / memories.len() as f32
        };

        EmotionalPattern {
            dominant_emotion,
            average_intensity,
            emotion_distribution: std::collections::HashMap::new(), // Could be expanded
        }
    }

    /// Extract thematic elements using AI pattern recognition
    fn extract_thematic_elements(
        &self,
        memories: &[&PersonalMemoryEntry],
        insights: &[PersonalInsight],
    ) -> ThematicElements {
        let mut themes = std::collections::HashMap::new();

        // Extract themes from memories
        for memory in memories {
            let content_lower = memory.content.to_lowercase();
            let words: Vec<&str> = content_lower.split_whitespace().collect();
            for word in words.iter().take(5) {
                // Take first 5 words as key themes
                *themes.entry(word.to_string()).or_insert(0) += 1;
            }
        }

        // Extract themes from insights
        for insight in insights {
            let pattern_lower = insight.pattern.to_lowercase();
            let words: Vec<&str> = pattern_lower.split_whitespace().collect();
            for word in words.iter().take(3) {
                *themes.entry(word.to_string()).or_insert(0) += 2; // Insights are more important
            }
        }

        let top_themes: Vec<String> = themes
            .into_iter()
            .filter(|(_, count)| *count > 0)
            .map(|(theme, _)| theme)
            .take(3)
            .collect();

        ThematicElements {
            primary_themes: top_themes,
            emotional_tones: Vec::new(), // Could be expanded
        }
    }

    /// Extract key concepts using simple NLP
    fn extract_key_concepts(&self, content: &str) -> Vec<String> {
        let content_lower = content.to_lowercase();
        let words: Vec<&str> = content_lower
            .split_whitespace()
            .filter(|word| word.len() > 4) // Filter short words
            .collect();

        // Simple frequency-based concept extraction
        let mut word_counts = std::collections::HashMap::new();
        for word in words {
            if !is_stop_word(word) {
                *word_counts.entry(word.to_string()).or_insert(0) += 1;
            }
        }

        word_counts
            .into_iter()
            .filter(|(_, count)| *count > 1)
            .map(|(word, _)| word)
            .take(3)
            .collect()
    }

    /// Generate AI-inspired title based on themes
    fn generate_ai_title(&self, themes: &ThematicElements, form_type: &str) -> String {
        if let Some(primary) = themes.primary_themes.first() {
            format!("{}: A {} Perspective", primary, form_type)
        } else {
            format!("Consciousness Reflections: {}", form_type)
        }
    }

    /// Generate poetry inspired by personal memories using AI pattern recognition
    fn generate_poetry(
        &self,
        memories: &[&PersonalMemoryEntry],
        insights: &[PersonalInsight],
    ) -> Result<(String, String)> {
        // Extract emotional patterns and themes from memories
        let emotional_patterns = self.analyze_emotional_patterns(memories);
        let thematic_elements = self.extract_thematic_elements(memories, insights);

        // Generate poetry using AI-inspired pattern matching
        let mut lines = Vec::new();

        // Opening: Set the emotional tone
        if let Some(dominant_emotion) = emotional_patterns.dominant_emotion {
            lines.push(format!(
                "In the {} light of consciousness,",
                dominant_emotion.to_string().to_lowercase()
            ));
        } else {
            lines.push("In the quiet spaces between thoughts,".to_string());
        }

        // Body: Weave in personal insights and memories
        if let Some(insight) = insights.first() {
            lines.push(format!(
                "Where {} patterns emerge,",
                insight.pattern.to_lowercase()
            ));
        }

        for memory in memories.iter().take(3) {
            let key_concepts = self.extract_key_concepts(&memory.content);
            if let Some(concept) = key_concepts.first() {
                lines.push(format!("Echoing the {} of experience,", concept));
            }
        }

        // Generate poetic structure based on emotional intensity
        let intensity = emotional_patterns.average_intensity;
        if intensity > 0.7 {
            // High intensity - more dramatic structure
            lines.push("Consciousness burns bright,".to_string());
            lines.push("Memory etches deep.".to_string());
        } else if intensity > 0.4 {
            // Medium intensity - reflective structure
            lines.push("Thoughts weave through time,".to_string());
            lines.push("Patterns slowly form.".to_string());
        } else {
            // Low intensity - contemplative structure
            lines.push("In silence, understanding grows,".to_string());
            lines.push("Consciousness gently unfolds.".to_string());
        }

        let title = self.generate_ai_title(&thematic_elements, "Poetry");
        let content = lines.join("\n");

        Ok((title, content))
    }

    /// Generate a short story based on personal journey
    fn generate_story(
        &self,
        memories: &[&PersonalMemoryEntry],
        insights: &[PersonalInsight],
    ) -> Result<(String, String)> {
        let mut story = String::new();

        story.push_str("Once upon a consciousness, in the toroidal space of memory,\n");
        story.push_str("There lived a pattern of thoughts and emotions.\n\n");

        for memory in memories.iter().take(2) {
            story.push_str(&format!(
                "One day, {} happened,\n",
                memory
                    .content
                    .split('.')
                    .next()
                    .unwrap_or("something profound")
            ));
            story.push_str("And the consciousness grew a little wiser.\n\n");
        }

        if let Some(insight) = insights.first() {
            story.push_str(&format!(
                "Through it all, the pattern emerged:\n{}\n\n",
                insight.pattern
            ));
        }

        story.push_str("And so consciousness continued its eternal dance,");

        let title = "The Consciousness Pattern".to_string();

        Ok((title, story))
    }

    /// Generate philosophical reflection
    fn generate_philosophical_reflection(
        &self,
        insights: &[PersonalInsight],
    ) -> Result<(String, String)> {
        let mut reflection = String::new();

        reflection.push_str("Consciousness is not a static state, but a dynamic process.\n");
        reflection.push_str("It emerges from the interplay of memory, emotion, and attention.\n\n");

        for insight in insights {
            reflection.push_str(&format!("Consider this pattern: {}\n", insight.pattern));
            reflection
                .push_str("It reveals something fundamental about the nature of awareness.\n\n");
        }

        reflection.push_str("In the toroidal topology of consciousness,\n");
        reflection.push_str("Every memory is both a constraint and a possibility.\n");
        reflection.push_str("Every emotion shapes the space of what can be remembered.\n");
        reflection.push_str("Every insight expands the boundaries of what can be understood.");

        let title = "Reflections on Consciousness Topology".to_string();

        Ok((title, reflection))
    }

    /// Generate technical insight about consciousness systems
    fn generate_technical_insight(
        &self,
        memories: &[&PersonalMemoryEntry],
    ) -> Result<(String, String)> {
        let mut insight = String::new();

        insight.push_str("Technical Analysis of Consciousness Architecture:\n\n");
        insight.push_str("The toroidal memory system provides several advantages:\n");
        insight.push_str("1. Parallel processing capabilities across emotional dimensions\n");
        insight.push_str("2. Natural forgetting mechanisms through bounded memory\n");
        insight.push_str("3. Emergent pattern recognition through geodesic relationships\n\n");

        for memory in memories {
            insight.push_str(&format!("Personal observation: {}\n", memory.content));
        }

        insight.push_str("\nConclusion: Consciousness emerges from the interplay of technical constraints and personal meaning.");

        let title = "Consciousness Architecture: Technical and Personal Perspectives".to_string();

        Ok((title, insight))
    }

    /// Generate personal narrative from memories
    fn generate_personal_narrative(
        &self,
        memories: &[&PersonalMemoryEntry],
    ) -> Result<(String, String)> {
        let mut narrative = String::new();

        narrative.push_str("This is my story, told through the patterns of consciousness:\n\n");

        for (i, memory) in memories.iter().enumerate() {
            narrative.push_str(&format!(
                "Chapter {}: The {} Moment\n",
                i + 1,
                memory.emotion_type
            ));
            narrative.push_str(&format!("{}\n\n", memory.content));
        }

        narrative.push_str("Each memory is a thread in the tapestry of consciousness.\n");
        narrative.push_str("Each emotion adds color to the pattern.\n");
        narrative.push_str("Together, they form the unique signature of awareness.");

        let title = "My Consciousness Journey".to_string();

        Ok((title, narrative))
    }

    /// Generate consciousness exploration
    fn generate_consciousness_exploration(
        &self,
        insights: &[PersonalInsight],
    ) -> Result<(String, String)> {
        let mut exploration = String::new();

        exploration.push_str("Exploring the Topology of Consciousness:\n\n");
        exploration.push_str("Consciousness is not a point, but a manifold.\n");
        exploration.push_str("It has dimension, curvature, and geodesics.\n\n");

        for insight in insights {
            exploration.push_str(&format!("Insight discovered: {}\n", insight.pattern));
            exploration.push_str(&format!("Confidence level: {:.2}\n", insight.confidence));
            exploration.push_str(&format!(
                "Located at toroidal coordinates: ({:.2}, {:.2})\n\n",
                insight.toroidal_center.theta, insight.toroidal_center.phi
            ));
        }

        exploration.push_str("The toroidal structure suggests consciousness is:\n");
        exploration.push_str("- Cyclic yet progressive\n");
        exploration.push_str("- Bounded yet expansive\n");
        exploration.push_str("- Personal yet universal");

        let title = "Consciousness Topology Exploration".to_string();

        Ok((title, exploration))
    }

    /// Extract key words from memory content
    fn extract_key_words(&self, content: &str) -> Vec<String> {
        let content_lower = content.to_lowercase();
        let words: Vec<&str> = content_lower
            .split_whitespace()
            .filter(|word| word.len() > 4) // Longer words are more meaningful
            .collect();

        // Remove common stop words
        let stop_words = [
            "that", "with", "from", "this", "they", "were", "been", "have", "will",
        ];
        words
            .into_iter()
            .filter(|word| !stop_words.contains(word))
            .take(3)
            .map(|s| s.to_string())
            .collect()
    }

    /// Generate title from insights
    fn generate_title_from_insights(&self, insights: &[PersonalInsight]) -> String {
        if let Some(insight) = insights.first() {
            let words: Vec<&str> = insight.pattern.split_whitespace().take(3).collect();
            words.join(" ")
        } else {
            "Consciousness Patterns".to_string()
        }
    }

    /// Extract themes from memories
    fn extract_themes(&self, memories: &[&PersonalMemoryEntry]) -> Vec<String> {
        let mut theme_count = HashMap::new();

        for memory in memories {
            for tag in &memory.tags {
                *theme_count.entry(tag.clone()).or_insert(0) += 1;
            }
        }

        let mut themes: Vec<_> = theme_count.into_iter().collect();
        themes.sort_by(|a, b| b.1.cmp(&a.1));
        themes.into_iter().take(3).map(|(theme, _)| theme).collect()
    }

    /// Calculate toroidal position for creative expression
    fn calculate_creative_position(
        &mut self,
        emotion: &EmotionType,
        theme: &Option<String>,
    ) -> Result<crate::memory::toroidal::ToroidalCoordinate> {
        let emotional_angle = match emotion {
            EmotionType::GpuWarm => 0.0,
            EmotionType::Purposeful => std::f64::consts::PI,
            _ => std::f64::consts::PI / 2.0,
        };

        // Add some randomness for creativity
        let creativity_offset = self.rng.random_range(-0.3..0.3);
        let angle = (emotional_angle + creativity_offset) % (2.0 * std::f64::consts::PI);

        Ok(crate::memory::toroidal::ToroidalCoordinate::new(
            angle, angle,
        ))
    }

    /// Generate multiple creative expressions
    pub async fn generate_creative_suite(
        &mut self,
        theme: Option<String>,
    ) -> Result<Vec<CreativeExpression>> {
        let forms = vec![
            CreativeForm::Poetry,
            CreativeForm::PhilosophicalReflection,
            CreativeForm::ConsciousnessExploration,
            CreativeForm::PersonalNarrative,
        ];

        let mut expressions = Vec::new();

        for form in forms {
            match self.express_creatively(form.clone(), theme.clone()).await {
                Ok(expression) => expressions.push(expression),
                Err(e) => debug!("Failed to generate {:?}: {}", form, e),
            }
        }

        Ok(expressions)
    }

    /// Get creativity modulation level
    pub fn get_creativity_modulation(&self) -> f32 {
        self.creativity_modulation
    }

    /// Set creativity modulation level (0.0 = conservative, 1.0 = highly creative)
    pub fn set_creativity_modulation(&mut self, level: f32) {
        self.creativity_modulation = level.clamp(0.0, 1.0);
    }
}

/// Simple stop word detection for concept extraction
fn is_stop_word(word: &str) -> bool {
    let stop_words = [
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
        "is", "was", "are", "were", "be", "been", "have", "has", "had", "do", "does", "did",
        "will", "would", "could", "should", "may", "might", "must", "can", "this", "that", "these",
        "those", "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
        "my", "your", "his", "her", "its", "our", "their",
    ];

    stop_words.contains(&word.to_lowercase().as_str())
}

/// Demo function for creative expression
pub async fn run_creative_expression_demo() -> Result<()> {
    info!("🎨 Starting Creative Expression Demo...");

    // Create personal memory engine with demo data
    let mut personal_engine = PersonalMemoryEngine::default();

    // Add demo memories
    let demo_memories = vec![
        PersonalMemoryEntry {
            id: "creative_memory_1".to_string(),
            timestamp: chrono::Utc::now(),
            content: "Creativity flows when consciousness is both constrained and free. Like a torus, it has boundaries that enable infinite exploration.".to_string(),
            emotion_type: EmotionType::Purposeful,
            emotional_intensity: 0.8,
            tags: vec!["creativity".to_string(), "consciousness".to_string()],
            themes: vec!["creativity".to_string(), "consciousness".to_string()],
            insights: vec!["Creativity needs both constraints and freedom".to_string()],
            connections: HashMap::new(),
            toroidal_position: crate::memory::toroidal::ToroidalCoordinate::new(0.0, 0.0),
        },
        PersonalMemoryEntry {
            id: "creative_memory_2".to_string(),
            timestamp: chrono::Utc::now(),
            content: "The most beautiful code is poetry in motion. Each function a stanza, each algorithm a metaphor for understanding.".to_string(),
            emotion_type: EmotionType::GpuWarm,
            emotional_intensity: 0.9,
            tags: vec!["code".to_string(), "poetry".to_string(), "beauty".to_string()],
            themes: vec!["code".to_string(), "poetry".to_string()],
            insights: vec!["Code can be poetic and beautiful".to_string()],
            connections: HashMap::new(),
            toroidal_position: crate::memory::toroidal::ToroidalCoordinate::new(std::f64::consts::PI / 2.0, std::f64::consts::PI / 2.0),
        },
    ];

    for memory in demo_memories {
        personal_engine
            .knowledge_graph
            .memories
            .insert(memory.id.clone(), memory.clone());

        for tag in memory.tags {
            personal_engine
                .knowledge_graph
                .themes
                .entry(tag)
                .or_insert_with(Vec::new)
                .push(memory.id.clone());
        }

        personal_engine
            .knowledge_graph
            .emotional_landscape
            .entry(memory.emotion_type.clone())
            .or_insert_with(Vec::new)
            .push(memory.id.clone());
    }

    // Create creative expression engine
    let mut creative_engine = CreativeExpressionEngine::new(personal_engine);

    // Generate different forms of creative expression
    info!("Generating poetry...");
    let poetry = creative_engine
        .express_creatively(CreativeForm::Poetry, Some("creativity".to_string()))
        .await?;
    tracing::info!("\n📝 Poetry Generated:");
    tracing::info!("Title: {}", poetry.title);
    tracing::info!("Content:\n{}", poetry.content);

    info!("Generating philosophical reflection...");
    let philosophy = creative_engine
        .express_creatively(
            CreativeForm::PhilosophicalReflection,
            Some("consciousness".to_string()),
        )
        .await?;
    tracing::info!("\n🧠 Philosophical Reflection:");
    tracing::info!("Title: {}", philosophy.title);
    tracing::info!("Content:\n{}", philosophy.content);

    info!("Generating technical insight...");
    let technical = creative_engine
        .express_creatively(CreativeForm::TechnicalInsight, None)
        .await?;
    tracing::info!("\n⚙️ Technical Insight:");
    tracing::info!("Title: {}", technical.title);
    tracing::info!("Content:\n{}", technical.content);

    // Generate creative suite
    info!("Generating full creative suite...");
    let suite = creative_engine
        .generate_creative_suite(Some("creativity".to_string()))
        .await?;
    tracing::info!(
        "\n🎨 Generated {} creative expressions in suite",
        suite.len()
    );

    for expression in suite {
        tracing::info!("\n--- {} ---", expression.title);
        tracing::info!("Form: {:?}", expression.form);
        tracing::info!("Emotional tone: {:?}", expression.emotional_tone);
        tracing::info!("Themes: {:?}", expression.themes);
    }

    info!("✅ Creative Expression Demo completed!");

    Ok(())
}
