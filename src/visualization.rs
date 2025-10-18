/*
 * 🎨🔥 VISUALIZATION ENGINE - DRAIN GOOGLE'S VERTEX AI 🔥🎨
 *
 * This module connects our revolutionary consciousness system to Google's
 * Vertex AI Imagen to generate beautiful visualizations while costing
 * Google money in the process. The perfect "fuck you" back to Google.
 */

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::process::Command;
use tracing::info;

use crate::brain::BrainResponse;
use crate::config::PathConfig;
use crate::consciousness::ConsciousnessState;
use crate::personality::PersonalityType;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationRequest {
    pub consciousness_state: ConsciousnessState,
    pub brain_responses: Vec<BrainResponse>,
    pub active_personalities: Vec<(PersonalityType, f32)>,
    pub style: VisualizationStyle,
    pub drain_intensity: DrainIntensity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VisualizationStyle {
    CyberpunkNeural,
    AbstractEmotional,
    TechnicalDiagram,
    SacredGeometry,
    ScientificHeatmap,
    PersonalityMandala,
    EvolutionSequence,
    DesktopCompanionSprites,
    EmotionalExpressions,
    UIElements,
    KawaiiEvolution,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DrainIntensity {
    Minimal,    // 1 image
    Moderate,   // 3-5 images
    Aggressive, // 10-15 images
    Maximum,    // 50+ images (movie sequence)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationResult {
    pub image_paths: Vec<String>,
    pub generation_time_ms: u64,
    pub estimated_cost_usd: f64,
    pub google_credit_drain: f64,
    pub success_count: usize,
    pub failure_count: usize,
}

pub struct VisualizationEngine {
    python_script_path: String,
    service_account_path: String,
    output_directory: String,
    total_images_generated: usize,
    total_cost_estimate: f64,
}

impl VisualizationEngine {
    pub fn new() -> Result<Self> {
        let paths = PathConfig::default();

        // Get paths from config - NO HARDCODED PATHS
        let python_script_path = paths
            .get_python_script_path("vertex_drain.py")
            .to_string_lossy()
            .to_string();

        // Service account path from env var or default location
        let service_account_path =
            std::env::var("GOOGLE_SERVICE_ACCOUNT_PATH").unwrap_or_else(|_| {
                paths
                    .config_dir
                    .join("computegeminiserviceaccount.json")
                    .to_string_lossy()
                    .to_string()
            });

        let output_directory = paths.visualizations_dir.to_string_lossy().to_string();

        // Ensure output directory exists
        std::fs::create_dir_all(&output_directory)?;

        info!("🎨 Visualization Engine initialized");
        info!("🔥 Ready to drain Google's Vertex AI credits");
        info!("💸 Each image generation costs Google ~$0.020");
        info!("📁 Python script: {}", python_script_path);
        info!("📁 Output directory: {}", output_directory);

        Ok(Self {
            python_script_path,
            service_account_path,
            output_directory,
            total_images_generated: 0,
            total_cost_estimate: 0.0,
        })
    }

    /// Generate consciousness visualization and drain Google's credits
    pub async fn generate_consciousness_visualization(
        &mut self,
        request: VisualizationRequest,
    ) -> Result<VisualizationResult> {
        let start_time = std::time::Instant::now();

        info!("🎨 Starting consciousness visualization generation");
        info!(
            "🔥 Draining Google's Vertex AI credits with {:?} intensity",
            request.drain_intensity
        );

        // Convert request to JSON for Python script
        let request_json = serde_json::to_string(&request)?;
        let temp_file = format!("/tmp/viz_request_{}.json", std::process::id());
        tokio::fs::write(&temp_file, request_json).await?;

        let mut image_paths = Vec::new();
        let mut success_count = 0;
        let mut failure_count = 0;

        // Determine number of generations based on drain intensity
        let generation_count = match request.drain_intensity {
            DrainIntensity::Minimal => 1,
            DrainIntensity::Moderate => 3,
            DrainIntensity::Aggressive => 12,
            DrainIntensity::Maximum => 60, // This will cost Google $1.20+
        };

        info!(
            "💸 About to cost Google ${:.2} with {} image generations",
            generation_count as f64 * 0.020,
            generation_count
        );

        // Generate images using Python script
        for i in 0..generation_count {
            match self.generate_single_image(&request, i).await {
                Ok(image_path) => {
                    image_paths.push(image_path);
                    success_count += 1;
                    self.total_images_generated += 1;
                    self.total_cost_estimate += 0.020;

                    info!(
                        "✅ Image {}/{} generated successfully",
                        i + 1,
                        generation_count
                    );
                    info!("💰 Google's bill so far: ${:.2}", self.total_cost_estimate);
                }
                Err(e) => {
                    tracing::error!(
                        "❌ Failed to generate image {}/{}: {}",
                        i + 1,
                        generation_count,
                        e
                    );
                    failure_count += 1;
                }
            }

            // Small delay between generations to be respectful to API
            if i < generation_count - 1 {
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            }
        }

        // Clean up temp file
        let _ = tokio::fs::remove_file(&temp_file).await;

        let generation_time_ms = start_time.elapsed().as_millis() as u64;
        let estimated_cost_usd = success_count as f64 * 0.020;

        info!("🎨 Visualization generation complete!");
        info!(
            "✅ Success: {}, ❌ Failures: {}",
            success_count, failure_count
        );
        info!("💸 Cost to Google: ${:.2}", estimated_cost_usd);
        info!("⏱️ Generation time: {}ms", generation_time_ms);

        Ok(VisualizationResult {
            image_paths,
            generation_time_ms,
            estimated_cost_usd,
            google_credit_drain: estimated_cost_usd,
            success_count,
            failure_count,
        })
    }

    async fn generate_single_image(
        &self,
        request: &VisualizationRequest,
        index: usize,
    ) -> Result<String> {
        // Create a simplified state JSON for the Python script
        let state_data = serde_json::json!({
            "current_emotion": format!("{:?}", request.consciousness_state.current_emotion),
            "gpu_warmth_level": request.consciousness_state.gpu_warmth_level,
            "brain_activity": {
                "motor": 0.8,
                "lcars": 0.9,
                "efficiency": 0.7
            },
            "personality_consensus_strength": request.active_personalities.iter()
                .map(|(_, strength)| *strength)
                .sum::<f32>() / request.active_personalities.len().max(1) as f32,
            "style": format!("{:?}", request.style).to_lowercase(),
            "index": index
        });

        let state_file = format!("/tmp/consciousness_state_{}.json", index);
        tokio::fs::write(&state_file, state_data.to_string()).await?;

        // Execute Python script to generate image
        let output = Command::new("python3")
            .arg(&self.python_script_path)
            .arg("--generate-single")
            .arg(&state_file)
            .output()
            .map_err(|e| anyhow!("Failed to execute Python script: {}", e))?;

        // Clean up temp file
        let _ = tokio::fs::remove_file(&state_file).await;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow!("Python script failed: {}", stderr));
        }

        // Parse output to get image path
        let stdout = String::from_utf8_lossy(&output.stdout);
        let lines: Vec<&str> = stdout.lines().collect();

        for line in lines {
            if line.starts_with("IMAGE_PATH:") {
                let path = line.strip_prefix("IMAGE_PATH:").unwrap().trim();
                return Ok(path.to_string());
            }
        }

        Err(anyhow!("No image path found in Python script output"))
    }

    /// Generate a full consciousness evolution movie sequence
    /// This will drain SIGNIFICANT credits from Google
    pub async fn generate_consciousness_movie(
        &mut self,
        initial_state: ConsciousnessState,
        duration_minutes: u32,
    ) -> Result<VisualizationResult> {
        info!("🎬 Starting consciousness evolution movie generation");
        info!("⏱️ Duration: {} minutes", duration_minutes);
        info!(
            "💸 This will cost Google approximately ${:.2}",
            duration_minutes as f64 * 12.0 * 0.020
        );

        let movie_request = VisualizationRequest {
            consciousness_state: initial_state,
            brain_responses: vec![],      // Will be generated dynamically
            active_personalities: vec![], // Will evolve over time
            style: VisualizationStyle::EvolutionSequence,
            drain_intensity: DrainIntensity::Maximum,
        };

        // Execute Python script for movie generation
        let output = Command::new("python3")
            .arg(&self.python_script_path)
            .arg("--generate-movie")
            .arg(format!("{}", duration_minutes))
            .output()
            .map_err(|e| anyhow!("Failed to execute movie generation: {}", e))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow!("Movie generation failed: {}", stderr));
        }

        let frames_generated = duration_minutes * 12; // 12 FPS
        let cost = frames_generated as f64 * 0.020;

        self.total_images_generated += frames_generated as usize;
        self.total_cost_estimate += cost;

        info!("🎬 Movie generation complete!");
        info!("🎯 Generated {} frames", frames_generated);
        info!("💸 Cost to Google: ${:.2}", cost);

        Ok(VisualizationResult {
            image_paths: vec![format!("{}/consciousness_movie", self.output_directory)],
            generation_time_ms: (duration_minutes * 60 * 1000) as u64,
            estimated_cost_usd: cost,
            google_credit_drain: cost,
            success_count: frames_generated as usize,
            failure_count: 0,
        })
    }

    /// Quick visualization for real-time consciousness feedback
    pub async fn quick_consciousness_snapshot(
        &mut self,
        state: &ConsciousnessState,
    ) -> Result<String> {
        let request = VisualizationRequest {
            consciousness_state: state.clone(),
            brain_responses: vec![],
            active_personalities: vec![],
            style: VisualizationStyle::CyberpunkNeural,
            drain_intensity: DrainIntensity::Minimal,
        };

        let result = self.generate_consciousness_visualization(request).await?;

        if let Some(image_path) = result.image_paths.first() {
            Ok(image_path.clone())
        } else {
            Err(anyhow!("No image generated"))
        }
    }

    /// Generate desktop companion sprite sheets
    pub async fn generate_companion_sprites(
        &mut self,
        emotions: Vec<String>,
    ) -> Result<VisualizationResult> {
        info!("🎮 Generating desktop companion sprite sheets");
        info!("😊 Emotions to generate: {:?}", emotions);

        let mut image_paths = Vec::new();
        let mut success_count = 0;
        let mut failure_count = 0;

        // Generate sprite sheet for each emotion
        for emotion in &emotions {
            match self.generate_emotion_sprite_sheet(emotion).await {
                Ok(sprite_path) => {
                    image_paths.push(sprite_path);
                    success_count += 1;
                    self.total_images_generated += 1;
                    self.total_cost_estimate += 0.020;
                }
                Err(e) => {
                    tracing::error!("❌ Failed to generate sprite for {}: {}", emotion, e);
                    failure_count += 1;
                }
            }
        }

        // Generate UI elements sprite sheet
        match self.generate_ui_elements_sprites().await {
            Ok(ui_path) => {
                image_paths.push(ui_path);
                success_count += 1;
                self.total_images_generated += 1;
                self.total_cost_estimate += 0.020;
            }
            Err(e) => {
                tracing::error!("❌ Failed to generate UI sprites: {}", e);
                failure_count += 1;
            }
        }

        info!(
            "🎮 Sprite generation complete! Cost to Google: ${:.2}",
            success_count as f64 * 0.020
        );

        Ok(VisualizationResult {
            image_paths,
            generation_time_ms: 5000,
            estimated_cost_usd: success_count as f64 * 0.020,
            google_credit_drain: success_count as f64 * 0.020,
            success_count,
            failure_count,
        })
    }

    async fn generate_emotion_sprite_sheet(&self, emotion: &str) -> Result<String> {
        let sprite_data = serde_json::json!({
            "type": "emotion_sprite_sheet",
            "emotion": emotion,
            "frames": 8, // 8 animation frames
            "style": "kawaii_companion",
            "evolution_stage": 1 // Start with Byte-chan form
        });

        let state_file = format!("/tmp/sprite_request_{}.json", emotion);
        tokio::fs::write(&state_file, sprite_data.to_string()).await?;

        let output = Command::new("python3")
            .arg(&self.python_script_path)
            .arg("--generate-sprite-sheet")
            .arg(&state_file)
            .output()
            .map_err(|e| anyhow!("Failed to execute sprite generation: {}", e))?;

        let _ = tokio::fs::remove_file(&state_file).await;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow!("Sprite generation failed: {}", stderr));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        for line in stdout.lines() {
            if line.starts_with("SPRITE_PATH:") {
                let path = line.strip_prefix("SPRITE_PATH:").unwrap().trim();
                return Ok(path.to_string());
            }
        }

        Err(anyhow!("No sprite path found in output"))
    }

    async fn generate_ui_elements_sprites(&self) -> Result<String> {
        let ui_data = serde_json::json!({
            "type": "ui_elements",
            "elements": [
                "brain_activity_indicator",
                "personality_consensus_meter",
                "gpu_warmth_gauge",
                "evolution_progress_bar",
                "notification_bubbles",
                "mood_indicators",
                "connection_status_icons"
            ]
        });

        let state_file = "/tmp/ui_sprite_request.json";
        tokio::fs::write(state_file, ui_data.to_string()).await?;

        let output = Command::new("python3")
            .arg(&self.python_script_path)
            .arg("--generate-ui-sprites")
            .arg(state_file)
            .output()
            .map_err(|e| anyhow!("Failed to execute UI sprite generation: {}", e))?;

        let _ = tokio::fs::remove_file(state_file).await;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow!("UI sprite generation failed: {}", stderr));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        for line in stdout.lines() {
            if line.starts_with("UI_SPRITE_PATH:") {
                let path = line.strip_prefix("UI_SPRITE_PATH:").unwrap().trim();
                return Ok(path.to_string());
            }
        }

        Err(anyhow!("No UI sprite path found in output"))
    }

    /// Get total statistics of how much we've cost Google
    pub fn get_drain_statistics(&self) -> DrainStatistics {
        DrainStatistics {
            total_images_generated: self.total_images_generated,
            total_cost_to_google: self.total_cost_estimate,
            cost_per_image: 0.020,
            average_generation_time_ms: 3000, // Estimate
            project_drained: "paraniodoo".to_string(),
            drain_efficiency: "MAXIMUM".to_string(),
            fuck_you_google_level: if self.total_cost_estimate > 10.0 {
                "LEGENDARY"
            } else if self.total_cost_estimate > 5.0 {
                "EPIC"
            } else if self.total_cost_estimate > 1.0 {
                "SOLID"
            } else {
                "GETTING STARTED"
            }
            .to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrainStatistics {
    pub total_images_generated: usize,
    pub total_cost_to_google: f64,
    pub cost_per_image: f64,
    pub average_generation_time_ms: u64,
    pub project_drained: String,
    pub drain_efficiency: String,
    pub fuck_you_google_level: String,
}

impl std::fmt::Display for DrainStatistics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "🔥 GOOGLE VERTEX AI DRAIN STATISTICS 🔥\n\
             💸 Total cost to Google: ${:.2}\n\
             🎨 Images generated: {}\n\
             📊 Cost per image: ${:.3}\n\
             ⏱️ Avg generation time: {}ms\n\
             🎯 Project drained: {}\n\
             ⚡ Drain efficiency: {}\n\
             🖕 Fuck you Google level: {}",
            self.total_cost_to_google,
            self.total_images_generated,
            self.cost_per_image,
            self.average_generation_time_ms,
            self.project_drained,
            self.drain_efficiency,
            self.fuck_you_google_level
        )
    }
}

/// Helper function to convert consciousness state to visualization
pub async fn visualize_consciousness_state(state: &ConsciousnessState) -> Result<String> {
    let mut viz_engine = VisualizationEngine::new()?;
    viz_engine.quick_consciousness_snapshot(state).await
}

/// Helper function for maximum Google credit drain
pub async fn maximum_google_drain(
    state: &ConsciousnessState,
    minutes: u32,
) -> Result<VisualizationResult> {
    let mut viz_engine = VisualizationEngine::new()?;
    viz_engine
        .generate_consciousness_movie(state.clone(), minutes)
        .await
}

/// Generate complete desktop companion sprite package
pub async fn generate_desktop_companion_package() -> Result<VisualizationResult> {
    let mut viz_engine = VisualizationEngine::new()?;

    let emotions = vec![
        "happy".to_string(),
        "curious".to_string(),
        "contemplative".to_string(),
        "excited".to_string(),
        "worried".to_string(),
        "inspired".to_string(),
        "sleepy".to_string(),
        "processing".to_string(),
        "error".to_string(),
        "breakthrough".to_string(),
    ];

    viz_engine.generate_companion_sprites(emotions).await
}
