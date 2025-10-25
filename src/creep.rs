// Copyright (c) 2025 Jason Van Pham (ruffian-l on GitHub) @ The Niodoo Collaborative
// Licensed under the MIT License - See LICENSE file for details
// Attribution required for all derivative works

//! Creep agents for spreading embeddings across code diffs

use candle_core::{Tensor, Device};
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};
use anyhow::{Result, anyhow};
use std::collections::HashMap;

/// Creep agent for spreading embeddings across code structures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreepAgent {
    pub id: usize,
    pub position: Vec<f32>,           // Current position in embedding space
    pub embedding: Vec<f32>,          // Agent's embedding representation
    pub coverage: DMatrix<bool>,      // Coverage map of explored areas
    pub energy: f32,                  // Agent energy (0.0-1.0)
    pub spread_rate: f32,             // How fast this agent spreads
}

impl CreepAgent {
    /// Create a new creep agent
    pub fn new(id: usize, position: Vec<f32>, embedding: Vec<f32>, coverage_size: usize) -> Self {
        let coverage = DMatrix::from_element(coverage_size, coverage_size, false);

        Self {
            id,
            position,
            embedding,
            coverage,
            energy: 1.0,
            spread_rate: 0.1,
        }
    }

    /// Update agent position based on code structure and other agents
    pub fn update_position(&mut self, target_embedding: &[f32], other_agents: &[CreepAgent]) {
        // Attraction to target embedding
        let attraction = self.calculate_attraction(target_embedding);

        // Repulsion from other agents
        let repulsion = self.calculate_repulsion(other_agents);

        // Combine forces
        let mut new_position = self.position.clone();
        for i in 0..new_position.len() {
            let force = attraction[i] + repulsion[i];
            new_position[i] += force * self.spread_rate;
            new_position[i] = new_position[i].clamp(-1.0, 1.0);
        }

        self.position = new_position;
        self.energy = (self.energy - 0.01).max(0.0); // Energy drain
    }

    /// Calculate attraction force towards target embedding
    fn calculate_attraction(&self, target: &[f32]) -> Vec<f32> {
        let mut forces = Vec::new();

        for (i, (&current, &target_val)) in self.position.iter().zip(target.iter()).enumerate() {
            let distance = target_val - current;
            forces.push(distance * 0.1); // Attraction strength
        }

        forces
    }

    /// Calculate repulsion force from other agents
    fn calculate_repulsion(&self, other_agents: &[CreepAgent]) -> Vec<f32> {
        let mut forces = vec![0.0; self.position.len()];
        let repulsion_radius = 0.2;

        for agent in other_agents {
            if agent.id == self.id {
                continue;
            }

            let distance = self.calculate_distance(&agent.position);
            if distance < repulsion_radius && distance > 0.0 {
                let repulsion_strength = (repulsion_radius - distance) / repulsion_radius;
                for (i, (&current, &other)) in self.position.iter().zip(agent.position.iter()).enumerate() {
                    forces[i] += (current - other) * repulsion_strength * 0.05;
                }
            }
        }

        forces
    }

    /// Calculate Euclidean distance to another position
    fn calculate_distance(&self, other: &[f32]) -> f32 {
        self.position.iter().zip(other.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    /// Mark coverage area as explored
    pub fn mark_coverage(&mut self, x: usize, y: usize) {
        if x < self.coverage.nrows() && y < self.coverage.ncols() {
            self.coverage[(x, y)] = true;
        }
    }

    /// Get coverage percentage
    pub fn coverage_percentage(&self) -> f32 {
        let total_cells = self.coverage.len();
        let covered_cells = self.coverage.iter().filter(|&&covered| covered).count();
        covered_cells as f32 / total_cells as f32
    }
}

/// Spawn creep agents from code diff using BERT embeddings
pub async fn spawn_from_diff(diff_content: &str, num_agents: usize, model: &BertModel) -> Result<Vec<CreepAgent>> {
    if num_agents == 0 {
        return Ok(Vec::new());
    }

    // Split diff into chunks for agent spawning
    let chunks = split_diff_into_chunks(diff_content, num_agents);
    let mut agents = Vec::new();

    for (i, chunk) in chunks.into_iter().enumerate() {
        // Generate embedding for this chunk
        let embedding = generate_bert_embedding(&chunk, model).await?;

        // Create agent at random position near the embedding
        let position = add_random_offset(&embedding, 0.1);

        let mut agent = CreepAgent::new(i, position, embedding, 100); // 100x100 coverage grid
        agent.energy = 1.0;
        agent.spread_rate = 0.05 + (i as f32 * 0.01); // Vary spread rates

        agents.push(agent);
    }

    Ok(agents)
}

/// Split diff into roughly equal chunks for agent spawning
fn split_diff_into_chunks(diff: &str, num_chunks: usize) -> Vec<String> {
    let lines: Vec<&str> = diff.lines().collect();
    if lines.is_empty() || num_chunks == 0 {
        return vec![diff.to_string()];
    }

    let chunk_size = (lines.len() as f32 / num_chunks as f32).ceil() as usize;
    let mut chunks = Vec::new();

    for i in 0..num_chunks {
        let start = i * chunk_size;
        let end = (start + chunk_size).min(lines.len());
        let chunk_lines: Vec<&str> = lines[start..end].iter().cloned().collect();
        chunks.push(chunk_lines.join("\n"));
    }

    chunks
}

/// Generate BERT embedding for text chunk
async fn generate_bert_embedding(text: &str, model: &BertModel) -> Result<Vec<f32>> {
    let device = Device::Cpu;

    // Create stub tokenizer since hf-hub feature not enabled
    let mut vocab = std::collections::HashMap::new();
    vocab.insert("[PAD]".to_string(), 0);
    vocab.insert("[UNK]".to_string(), 100);
    vocab.insert("[CLS]".to_string(), 101);
    vocab.insert("[SEP]".to_string(), 102);
    vocab.insert("[MASK]".to_string(), 103);

    // Simple tokenization: split by whitespace and map to vocab
    let tokens: Vec<&str> = text.split_whitespace().collect();
    let mut input_ids: Vec<u32> = vec![101]; // [CLS]
    for token in tokens.iter().take(510) { // Limit to reasonable length
        let id = vocab.get(*token).copied().unwrap_or(100); // [UNK] if not found
        input_ids.push(id);
    }
    input_ids.push(102); // [SEP]

    // Create attention mask (all 1s)
    let attention_mask: Vec<u32> = vec![1; input_ids.len()];

    // Convert to tensors
    let input_tensor = Tensor::new(input_ids.as_slice(), &device)?;
    let attention_tensor = Tensor::new(attention_mask.as_slice(), &device)?;

    // Forward pass through BERT
    let outputs = model.forward(&input_tensor.unsqueeze(0)?, &attention_tensor.unsqueeze(0)?, None)?;

    // Extract [CLS] token embedding (first token, last hidden state)
    let cls_embedding = outputs.get(0)?  // Last hidden state
        .get(0)?                         // First sequence
        .get(0)?                         // First token ([CLS])
        .to_vec1::<f32>()?;

    Ok(cls_embedding)
}

/// Add random offset to embedding for agent diversity
fn add_random_offset(embedding: &[f32], offset_scale: f32) -> Vec<f32> {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    embedding.iter()
        .map(|&x| x + rng.gen::<f32>() * 2.0 * offset_scale - offset_scale)
        .map(|x| x.clamp(-1.0, 1.0))
        .collect()
}

/// Spread activation using graph Laplacian heat diffusion
pub fn spread_activation(agents: &mut [CreepAgent], graph_laplacian: &DMatrix<f64>, timestep: f32) -> Result<()> {
    if agents.is_empty() {
        return Ok(());
    }

    // Create activation vector from agent positions
    let num_nodes = graph_laplacian.nrows();
    let mut activation = DVector::zeros(num_nodes);

    // Map agent positions to graph nodes (simplified mapping)
    for agent in agents.iter() {
        if let Some(node_idx) = map_position_to_node(&agent.position, num_nodes) {
            activation[node_idx] = agent.energy as f64;
        }
    }

    // Heat diffusion: dA/dt = -L * A (Laplacian operator)
    let diffusion = graph_laplacian * &activation;
    let new_activation = &activation - timestep as f64 * &diffusion;

    // Update agent positions based on new activation
    for agent in agents.iter_mut() {
        if let Some(node_idx) = map_position_to_node(&agent.position, num_nodes) {
            let new_energy = new_activation[node_idx] as f32;
            agent.energy = new_energy.max(0.0).min(1.0);

            // Update position based on activation gradient
            update_agent_position_from_activation(agent, &new_activation, node_idx);
        }
    }

    Ok(())
}

/// Map embedding position to graph node index
fn map_position_to_node(position: &[f32], num_nodes: usize) -> Option<usize> {
    if position.is_empty() {
        return None;
    }

    // Simple mapping based on first dimension (could be more sophisticated)
    let normalized = (position[0] + 1.0) / 2.0; // Convert from [-1, 1] to [0, 1]
    let node_idx = (normalized * num_nodes as f32) as usize;
    Some(node_idx.min(num_nodes - 1))
}

/// Update agent position based on activation gradient
fn update_agent_position_from_activation(agent: &mut CreepAgent, activation: &DVector<f64>, node_idx: usize) {
    // Calculate gradient (simple finite difference)
    let gradient = if node_idx > 0 && node_idx < activation.len() - 1 {
        (activation[node_idx + 1] - activation[node_idx - 1]) / 2.0
    } else {
        0.0
    };

    // Update position in direction of gradient
    if let Some(first_dim) = agent.position.first_mut() {
        *first_dim += gradient as f32 * 0.01;
        *first_dim = first_dim.clamp(-1.0, 1.0);
    }
}

/// Cache review results for incremental processing
pub fn cache_review(agents: &[CreepAgent]) -> HashMap<String, Vec<f32>> {
    let mut cache = HashMap::new();

    for agent in agents {
        let key = format!("agent_{}", agent.id);
        cache.insert(key, agent.position.clone());
    }

    cache
}

/// Calculate coverage statistics for all agents
pub fn calculate_coverage_stats(agents: &[CreepAgent]) -> CoverageStats {
    let total_coverage: f32 = agents.iter().map(|a| a.coverage_percentage()).sum();
    let average_coverage = if agents.is_empty() { 0.0 } else { total_coverage / agents.len() as f32 };

    let total_energy: f32 = agents.iter().map(|a| a.energy).sum();
    let average_energy = if agents.is_empty() { 0.0 } else { total_energy / agents.len() as f32 };

    CoverageStats {
        total_coverage,
        average_coverage,
        total_energy,
        average_energy,
        active_agents: agents.iter().filter(|a| a.energy > 0.1).count(),
    }
}

/// Coverage statistics for creep agent swarm
#[derive(Debug, Clone)]
pub struct CoverageStats {
    pub total_coverage: f32,
    pub average_coverage: f32,
    pub total_energy: f32,
    pub average_energy: f32,
    pub active_agents: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_creep_agent_creation() {
        let position = vec![0.1, 0.2, 0.3];
        let embedding = vec![0.4, 0.5, 0.6];
        let agent = CreepAgent::new(0, position.clone(), embedding.clone(), 50);

        assert_eq!(agent.id, 0);
        assert_eq!(agent.position, position);
        assert_eq!(agent.embedding, embedding);
        assert_eq!(agent.energy, 1.0);
        assert_eq!(agent.coverage.nrows(), 50);
    }

    #[test]
    fn test_chunk_splitting() {
        let diff = "line1\nline2\nline3\nline4\nline5";
        let chunks = split_diff_into_chunks(diff, 2);

        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].contains("line1"));
        assert!(chunks[0].contains("line2"));
        assert!(chunks[1].contains("line4"));
        assert!(chunks[1].contains("line5"));
    }

    #[test]
    fn test_position_mapping() {
        let position = vec![0.5]; // Midpoint
        let node_idx = map_position_to_node(&position, 100);
        assert_eq!(node_idx, Some(50)); // Should map to middle

        let position_edge = vec![-0.9]; // Near edge
        let node_idx_edge = map_position_to_node(&position_edge, 100);
        assert_eq!(node_idx_edge, Some(5)); // Should map near beginning
    }

    #[test]
    fn test_coverage_stats() {
        let agent1 = CreepAgent::new(0, vec![0.0], vec![0.0], 10);
        let agent2 = CreepAgent::new(1, vec![0.0], vec![0.0], 10);

        // Mark some coverage
        agent1.mark_coverage(0, 0);
        agent1.mark_coverage(1, 1);

        let agents = vec![agent1, agent2];
        let stats = calculate_coverage_stats(&agents);

        assert_eq!(stats.active_agents, 2);
        assert!(stats.average_coverage > 0.0);
    }
}






