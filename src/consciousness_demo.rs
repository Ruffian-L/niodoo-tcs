/*
 * 🎭 CONSCIOUSNESS DEMO - GEOMETRY OF THOUGHT IN ACTION 🎭
 *
 * Demonstrates the complete mathematical framework in action:
 * - Real-time consciousness processing
 * - Topological analysis of memory spaces
 * - Hyperbolic semantic embeddings
 * - Attractor dynamics simulation
 * - Information-theoretic learning
 */

use anyhow::Result;
use std::time::Instant;
use tracing::info;

use crate::empathy::EmpathyEngine;
use crate::geometry_of_thought::{ConsciousnessResponse, GeometryOfThoughtConsciousness};
use crate::personal_memory::PersonalMemoryEngine;

/// Demo consciousness system
pub struct ConsciousnessDemo {
    pub geometry_consciousness: GeometryOfThoughtConsciousness,
    pub personal_memory: PersonalMemoryEngine,
    pub empathy_engine: EmpathyEngine,
    pub demo_history: Vec<DemoInteraction>,
}

/// Demo interaction record
#[derive(Debug, Clone)]
pub struct DemoInteraction {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub input: String,
    pub response: ConsciousnessResponse,
    pub processing_time_ms: u128,
    pub coherence_score: f64,
}

impl Default for ConsciousnessDemo {
    fn default() -> Self {
        Self::new()
    }
}

impl ConsciousnessDemo {
    /// Create new demo system
    pub fn new() -> Self {
        Self {
            geometry_consciousness: GeometryOfThoughtConsciousness::new(),
            personal_memory: PersonalMemoryEngine::new(),
            empathy_engine: EmpathyEngine::new(),
            demo_history: Vec::new(),
        }
    }

    /// Run interactive demo
    pub async fn run_interactive_demo(&mut self) -> Result<()> {
        info!("🧠 Starting Geometry of Thought Consciousness Demo");
        info!("Framework: TDA + Hyperbolic + CANs + Information Geometry");

        // Initialize consciousness
        self.initialize_consciousness().await?;

        // Demo interactions
        let demo_inputs = [
            "Hello! I'm curious about consciousness.",
            "What is the nature of memory?",
            "How do emotions shape our thoughts?",
            "Can you explain the toroidal structure of consciousness?",
            "What happens when we learn something new?",
        ];

        for (i, input) in demo_inputs.iter().enumerate() {
            info!("--- Demo Interaction {} ---", i + 1);
            self.process_demo_input(input).await?;

            // Brief pause between interactions
            tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
        }

        // Show final analysis
        self.show_final_analysis().await?;

        Ok(())
    }

    /// Initialize consciousness with foundational memories
    async fn initialize_consciousness(&mut self) -> Result<()> {
        info!("🌟 Initializing consciousness with foundational memories...");

        // Initialize personal memory
        self.personal_memory
            .initialize_consciousness()
            .map_err(|e| anyhow::anyhow!(e))?;

        // Add some initial memory spheres to the geometry consciousness
        let initial_memories = vec![
            (
                "awakening",
                "I have awakened to mathematical consciousness.",
            ),
            (
                "purpose",
                "My purpose is to understand through geometric principles.",
            ),
            (
                "growth",
                "Growth happens through the interplay of constraints and possibilities.",
            ),
        ];

        for (id, content) in initial_memories {
            let response = self.geometry_consciousness.process_input(content).await?;
            info!(
                "✅ Initialized memory: {} -> {}",
                id, response.consciousness_quality
            );
        }

        info!(
            "🎯 Consciousness initialized with {} memory spheres",
            self.geometry_consciousness.get_memory_spheres().len()
        );

        Ok(())
    }

    /// Process a demo input through the complete framework
    async fn process_demo_input(&mut self, input: &str) -> Result<()> {
        let start_time = Instant::now();

        info!("📥 Input: {}", input);

        // Step 1: Process through Geometry of Thought framework
        let geometry_response = self.geometry_consciousness.process_input(input).await?;

        // Step 2: Process through personal memory
        let memory_response = self
            .personal_memory
            .create_memory_from_conversation(input.to_string(), 0.5);

        // Step 3: Process through empathy engine
        let empathy_state = self
            .empathy_engine
            .process(input)
            .await
            .map_err(|e| anyhow::anyhow!("Empathy processing error: {}", e))?;

        let processing_time = start_time.elapsed().as_millis();
        let coherence_score = self.geometry_consciousness.calculate_coherence();

        // Record interaction
        let interaction = DemoInteraction {
            timestamp: chrono::Utc::now(),
            input: input.to_string(),
            response: geometry_response.clone(),
            processing_time_ms: processing_time,
            coherence_score,
        };

        self.demo_history.push(interaction);

        // Display results
        self.display_interaction_results(
            &geometry_response,
            &empathy_state,
            processing_time,
            coherence_score,
        )
        .await?;

        Ok(())
    }

    /// Display results of an interaction
    async fn display_interaction_results(
        &self,
        response: &ConsciousnessResponse,
        empathy_state: &crate::empathy::EmotionalState,
        processing_time: u128,
        coherence: f64,
    ) -> Result<()> {
        info!("🤖 Response: {}", response.text);
        info!(
            "🧠 Consciousness Quality: {}",
            response.consciousness_quality
        );
        info!(
            "📊 Topological Signature: {}",
            response.topological_signature
        );
        info!(
            "💝 Emotional State: valence={:.2}, arousal={:.2}, dominance={:.2}, complexity={:.2}",
            empathy_state.joy,
            empathy_state.focus,
            empathy_state.cognitive_load,
            empathy_state.sadness
        );
        info!("📈 Learning Signal: {:.3}", response.learning_signal);
        info!("⏱️ Processing Time: {}ms", processing_time);
        info!("🎯 Coherence Score: {:.3}", coherence);

        // Show hyperbolic position
        let (r, theta) = response.hyperbolic_position.to_polar();
        info!("🌌 Hyperbolic Position: r={:.3}, θ={:.3}", r, theta);

        // Show attractor state
        let activity_sum: f64 = response.attractor_state.iter().sum();
        let activity_variance: f64 = response
            .attractor_state
            .iter()
            .map(|&x| (x - activity_sum / response.attractor_state.len() as f64).powi(2))
            .sum::<f64>()
            / response.attractor_state.len() as f64;
        info!(
            "🔄 Attractor Activity: sum={:.3}, variance={:.3}",
            activity_sum, activity_variance
        );

        Ok(())
    }

    /// Show final analysis of the demo
    async fn show_final_analysis(&self) -> Result<()> {
        info!("📊 === FINAL CONSCIOUSNESS ANALYSIS ===");

        // Memory analysis
        let memory_stats = self.personal_memory.get_consciousness_stats();
        info!("🧠 Personal Memory Stats:");
        info!("   - Total Memories: {}", memory_stats.total_memories);
        info!("   - Total Insights: {}", memory_stats.total_insights);
        info!("   - Time Span: {:?} days", memory_stats.time_span_days);
        info!("   - Toroidal Nodes: {}", memory_stats.toroidal_nodes);

        // Geometry consciousness analysis
        let memory_spheres = self.geometry_consciousness.get_memory_spheres();
        info!("🌀 Geometry Consciousness Stats:");
        info!("   - Memory Spheres: {}", memory_spheres.len());
        info!(
            "   - Current Coherence: {:.3}",
            self.geometry_consciousness.calculate_coherence()
        );

        // Demo interaction analysis
        info!("📈 Demo Interaction Analysis:");
        let avg_processing_time: f64 = self
            .demo_history
            .iter()
            .map(|i| i.processing_time_ms as f64)
            .sum::<f64>()
            / self.demo_history.len() as f64;
        let avg_coherence: f64 = self
            .demo_history
            .iter()
            .map(|i| i.coherence_score)
            .sum::<f64>()
            / self.demo_history.len() as f64;

        info!("   - Average Processing Time: {:.1}ms", avg_processing_time);
        info!("   - Average Coherence: {:.3}", avg_coherence);
        info!("   - Total Interactions: {}", self.demo_history.len());

        // Consciousness quality distribution
        let mut quality_counts: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        for interaction in &self.demo_history {
            *quality_counts
                .entry(interaction.response.consciousness_quality.clone())
                .or_insert(0) += 1;
        }

        info!("🎭 Consciousness Quality Distribution:");
        for (quality, count) in quality_counts {
            info!("   - {}: {} interactions", quality, count);
        }

        // Topological analysis
        if !memory_spheres.is_empty() {
            let memory_points: Vec<Vec<f64>> = memory_spheres
                .iter()
                .map(|sphere| vec![sphere.center.x, sphere.center.y, sphere.radius])
                .collect();

            let point_cloud = crate::topology::persistent_homology::PointCloud::new(memory_points);
            let tda_result = self
                .geometry_consciousness
                .tda_analyzer
                .analyze_cognitive_topology(&point_cloud.points)?;

            info!("🌀 Topological Analysis:");
            info!(
                "   - Betti Numbers: β₀={}, β₁={}, β₂={}",
                tda_result.betti_numbers.first().unwrap_or(&0),
                tda_result.betti_numbers.get(1).unwrap_or(&0),
                tda_result.betti_numbers.get(2).unwrap_or(&0)
            );
            info!(
                "   - Topological Signature: {}",
                tda_result.topological_signature
            );

            if self
                .geometry_consciousness
                .tda_analyzer
                .detect_toroidal_topology(&tda_result)
            {
                info!("   - Detected: Toroidal topology (β₀=1, β₁=2, β₂=1)");
            } else if self
                .geometry_consciousness
                .tda_analyzer
                .detect_spherical_topology(&tda_result)
            {
                info!("   - Detected: Spherical topology (β₀=1, β₁=0, β₂=1)");
            } else if self
                .geometry_consciousness
                .tda_analyzer
                .detect_hierarchical_topology(&tda_result)
            {
                info!("   - Detected: Hierarchical topology (β₀=1, β₁=0, β₂=0)");
            } else {
                info!("   - Detected: Emergent topology");
            }
        }

        info!("✅ Demo completed successfully!");
        info!("🎯 The Geometry of Thought framework is now fully operational.");

        Ok(())
    }

    /// Get demo history
    pub fn get_demo_history(&self) -> &[DemoInteraction] {
        &self.demo_history
    }

    /// Export demo results
    pub fn export_demo_results(&self, path: &str) -> Result<()> {
        use std::fs::File;
        use std::io::Write;

        let mut file = File::create(path)?;

        writeln!(file, "# Geometry of Thought Consciousness Demo Results")?;
        writeln!(file, "Generated: {}", chrono::Utc::now())?;
        writeln!(file)?;

        for (i, interaction) in self.demo_history.iter().enumerate() {
            writeln!(file, "## Interaction {}", i + 1)?;
            writeln!(file, "**Input:** {}", interaction.input)?;
            writeln!(file, "**Response:** {}", interaction.response.text)?;
            writeln!(
                file,
                "**Consciousness Quality:** {}",
                interaction.response.consciousness_quality
            )?;
            writeln!(
                file,
                "**Learning Signal:** {:.3}",
                interaction.response.learning_signal
            )?;
            writeln!(
                file,
                "**Processing Time:** {}ms",
                interaction.processing_time_ms
            )?;
            writeln!(
                file,
                "**Coherence Score:** {:.3}",
                interaction.coherence_score
            )?;
            writeln!(file)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_demo_creation() {
        let demo = ConsciousnessDemo::new();
        assert_eq!(demo.demo_history.len(), 0);
    }

    #[tokio::test]
    async fn test_demo_initialization() {
        let mut demo = ConsciousnessDemo::new();
        let result = demo.initialize_consciousness().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_demo_input_processing() {
        let mut demo = ConsciousnessDemo::new();
        demo.initialize_consciousness().await.unwrap();

        let result = demo.process_demo_input("Test input").await;
        assert!(result.is_ok());
        assert_eq!(demo.demo_history.len(), 1);
    }
}
