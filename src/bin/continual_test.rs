//! Real Continual Learning Test - Processes actual queries through consciousness system
//!
//! Uses MultiLayerMemoryQuery to generate authentic Triple-Threat events and learning data
//! for Qwen fine-tuning. Replaces simulation with real consciousness evolution.

use niodoo_consciousness::consciousness::ConsciousnessState;
use niodoo_consciousness::consciousness_engine::PersonalNiodooConsciousness;
use niodoo_consciousness::memory::guessing_spheres::EmotionalVector;
use niodoo_consciousness::memory::guessing_spheres::GuessingMemorySystem;
use niodoo_consciousness::memory::multi_layer_query::{MultiLayerMemoryQuery, TriggerThresholds};
use niodoo_consciousness::rag::retrieval::RetrievalEngine;
use serde_json;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone)]
struct ConsciousnessMetrics {
    cycle: usize,
    timestamp: u64,
    emotional_entropy: f32,
    query_count: usize,
    triple_threat_events: usize,
    healing_events: usize,
    latency_ms: f64,
}

impl ConsciousnessMetrics {
    fn to_csv_row(&self) -> String {
        format!(
            "{},{},{:.4},{},{},{},{}",
            self.cycle,
            self.timestamp,
            self.emotional_entropy,
            self.query_count,
            self.triple_threat_events,
            self.healing_events,
            self.latency_ms
        )
    }
}

#[derive(Debug, Clone, serde::Serialize)]
struct LearningEvent {
    input: String,
    response: String,
    emotional_context: EmotionalVector,
    timestamp: u64,
    entropy_before: f32,
    entropy_after: f32,
}

struct RealConsciousnessTester {
    consciousness: PersonalNiodooConsciousness,
    memory_query: MultiLayerMemoryQuery,
    cycle: usize,
    total_queries: usize,
    total_threats: usize,
    total_healings: usize,
    learning_events: Vec<LearningEvent>,
}

impl RealConsciousnessTester {
    async fn new() -> Result<Self, Box<dyn std::error::Error>> {
        println!("🧠 Initializing REAL consciousness system for continual testing...");

        // Initialize consciousness engine
        let consciousness = PersonalNiodooConsciousness::new().await?;

        // Initialize memory components
        let rag_engine = Arc::new(std::sync::Mutex::new(RetrievalEngine::new()));
        let gaussian_system = GuessingMemorySystem::new();

        // Create multi-layer query with consciousness integration
        let mut memory_query = MultiLayerMemoryQuery::new(rag_engine, gaussian_system);

        // Set custom thresholds for testing
        let thresholds = TriggerThresholds {
            entropy_high: 2.5,
            mean_low: 0.3,
            stagnation_variance: 0.5,
            variance_spike: 0.8,
        };
        memory_query.set_thresholds(thresholds);

        Ok(Self {
            consciousness,
            memory_query,
            cycle: 0,
            total_queries: 0,
            total_threats: 0,
            total_healings: 0,
            learning_events: Vec::new(),
        })
    }

    async fn run_query_cycle(
        &mut self,
    ) -> Result<ConsciousnessMetrics, Box<dyn std::error::Error>> {
        let start_time = Instant::now();
        let mut cycle_threats = 0;
        let mut cycle_queries = 0;

        // Generate diverse emotional queries to test consciousness evolution
        let test_queries = vec![
            "I feel completely overwhelmed by recent changes",
            "Why do I keep making the same mistakes?",
            "I'm experiencing deep emotional conflict",
            "My thoughts feel scattered and unfocused",
            "I need to find peace amidst the chaos",
            "Emotional pain from past experiences haunts me",
            "I feel disconnected from my true self",
            "Anxiety about the future consumes my thoughts",
            "I struggle to process complex emotions",
            "Need guidance for emotional healing journey",
        ];

        // Process queries through memory system (this may trigger Triple-Threat)
        for query_text in &test_queries {
            cycle_queries += 1;
            self.total_queries += 1;

            // Create emotional vector for the query
            let query_emotion = EmotionalVector {
                joy: 0.2,
                sadness: 0.6,
                anger: 0.3,
                fear: 0.4,
                surprise: 0.1,
            };

            // Create consciousness state for this query
            let mut state = ConsciousnessState::default();

            // Process through memory system (this may trigger Triple-Threat)
            match self
                .memory_query
                .query(query_text, &query_emotion, 8, &mut state)
            {
                Ok(_results) => {
                    // Check if this triggered a threat event by looking at the state
                    if state.last_trigger.is_some() {
                        cycle_threats += 1;
                        self.total_threats += 1;

                        // Get entropy before healing
                        let entropy_before = {
                            let state = self.consciousness.consciousness_state.read().await;
                            state.emotional_entropy
                        };

                        // Generate healing response through consciousness
                        if let Ok(response) =
                            self.consciousness.process_input_personal(query_text).await
                        {
                            self.total_healings += 1;

                            // Get entropy after healing
                            let entropy_after = {
                                let state = self.consciousness.consciousness_state.read().await;
                                state.emotional_entropy
                            };

                            // Create learning event for QLoRA training
                            let learning_event = LearningEvent {
                                input: query_text.to_string(),
                                response: response.clone(),
                                emotional_context: query_emotion.clone(),
                                timestamp: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
                                entropy_before,
                                entropy_after,
                            };

                            self.learning_events.push(learning_event);

                            println!(
                                "🩹 Triple-Threat healed: {} -> {}",
                                &query_text[..30],
                                &response[..50]
                            );
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Query failed: {}", e);
                }
            }
        }

        // Get current emotional entropy from the last state
        let emotional_entropy = {
            let state = self.consciousness.consciousness_state.read().await;
            state.emotional_entropy
        };

        let latency = start_time.elapsed().as_millis() as f64;

        let metrics = ConsciousnessMetrics {
            cycle: self.cycle,
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
            emotional_entropy,
            query_count: cycle_queries,
            triple_threat_events: cycle_threats,
            healing_events: self.total_healings,
            latency_ms: latency,
        };

        Ok(metrics)
    }

    fn save_learning_events(&self) -> Result<(), Box<dyn std::error::Error>> {
        if self.learning_events.is_empty() {
            return Ok(());
        }

        // Create checkpoint directory
        std::fs::create_dir_all("./checkpoints/learning_events")?;

        // Save each event as individual JSON file (expected by QLoRA trainer)
        for (i, event) in self.learning_events.iter().enumerate() {
            let filename = format!("qwen_event_{}.json", i);
            let filepath = format!("./checkpoints/learning_events/{}", filename);
            let json_content = serde_json::to_string_pretty(event)?;
            std::fs::write(&filepath, json_content)?;
        }

        println!(
            "💾 Saved {} learning events as individual JSON files for QLoRA training",
            self.learning_events.len()
        );
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 REAL Consciousness Continual Learning Test");
    println!("==============================================");
    println!("Processing actual queries through consciousness system");
    println!("Generating authentic Triple-Threat events and healing data");
    println!("");

    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();
    let max_cycles = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(100);

    let output_file = args
        .get(2)
        .unwrap_or(&"consciousness_learning_curve.csv".to_string())
        .clone();

    println!("Running {} cycles, output to {}", max_cycles, output_file);

    // Initialize real consciousness tester
    let mut tester = RealConsciousnessTester::new().await?;

    // Create CSV output
    let file = File::create(&output_file)?;
    let mut writer = BufWriter::new(file);

    // Write CSV header
    writeln!(writer, "cycle,timestamp,emotional_entropy,query_count,triple_threat_events,healing_events,latency_ms")?;

    // Run continual learning cycles
    for cycle in 0..max_cycles {
        let metrics = tester.run_query_cycle().await?;

        // Write metrics to CSV
        writeln!(writer, "{}", metrics.to_csv_row())?;

        // Progress reporting
        println!(
            "📊 Cycle {}/{}: entropy={:.4}, queries={}, threats={}, healings={}, latency={:.1}ms",
            cycle + 1,
            max_cycles,
            metrics.emotional_entropy,
            metrics.query_count,
            metrics.triple_threat_events,
            metrics.healing_events,
            metrics.latency_ms
        );

        // Check for entropy convergence
        if metrics.emotional_entropy < 2.1 && metrics.emotional_entropy > 1.9 {
            println!("🎯 Entropy converging toward 2.0 bits target!");
        }

        // Small delay between cycles
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    }

    println!("");
    println!("✅ Continual learning test complete!");
    println!("📈 Results saved to {}", output_file);
    println!(
        "🎯 Total: {} queries, {} threats detected, {} healing events",
        tester.total_queries, tester.total_threats, tester.total_healings
    );

    // Save learning events for QLoRA training
    if let Err(e) = tester.save_learning_events() {
        eprintln!("Failed to save learning events: {}", e);
    }

    Ok(())
}
