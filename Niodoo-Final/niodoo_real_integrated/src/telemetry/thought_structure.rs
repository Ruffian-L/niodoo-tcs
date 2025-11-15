//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

//! Thought Structure Builder
//!
//! Builds hierarchical thought trees from pipeline execution to represent
//! the AI's reasoning process.

use crate::telemetry::{
    DecisionOption, DecisionPoint, ReasoningStep, ReasoningType, ThoughtNode, ThoughtNodeType,
    ThoughtTree,
};
use petgraph::prelude::*;
use std::collections::HashMap;
use uuid::Uuid;

/// Builder for constructing thought trees from pipeline execution
pub struct ThoughtTreeBuilder {
    nodes: Vec<ThoughtNode>,
    reasoning_steps: Vec<ReasoningStep>,
    decision_points: Vec<DecisionPoint>,
    confidence_path: Vec<f32>,
    graph: DiGraph<Uuid, f32>, // Graph of node IDs with edge weights (confidence)
    node_map: HashMap<Uuid, usize>, // Map from UUID to graph node index
}

impl ThoughtTreeBuilder {
    /// Create a new thought tree builder
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            reasoning_steps: Vec::new(),
            decision_points: Vec::new(),
            confidence_path: Vec::new(),
            graph: DiGraph::new(),
            node_map: HashMap::new(),
        }
    }

    /// Add an observation node
    pub fn add_observation(
        &mut self,
        source: String,
        content: String,
        confidence: f32,
        raw_data: Option<String>,
        parent_id: Option<Uuid>,
    ) -> Uuid {
        let node_id = Uuid::new_v4();
        let timestamp = chrono::Utc::now().to_rfc3339();

        let node = ThoughtNode {
            node_id,
            node_type: ThoughtNodeType::Observation { source, raw_data },
            content,
            confidence,
            confidence_interval: None,
            pruned: false,
            pruning_rationale: None,
            timestamp,
            parent_id,
            children: Vec::new(),
            metadata: HashMap::new(),
        };

        self.add_node_to_graph(node_id, parent_id, confidence);
        self.nodes.push(node);
        self.confidence_path.push(confidence);
        node_id
    }

    /// Add a reasoning step
    pub fn add_reasoning(
        &mut self,
        reasoning_type: ReasoningType,
        premises: Vec<String>,
        conclusion: String,
        confidence: f32,
        parent_id: Option<Uuid>,
    ) -> (Uuid, Uuid) {
        let node_id = Uuid::new_v4();
        let step_id = Uuid::new_v4();
        let timestamp = chrono::Utc::now().to_rfc3339();

        let reasoning_step = ReasoningStep {
            step_id,
            step_type: reasoning_type.clone(),
            premises: premises.clone(),
            conclusion: conclusion.clone(),
            confidence,
        };

        let node = ThoughtNode {
            node_id,
            node_type: ThoughtNodeType::Reasoning {
                reasoning_type,
                chain: premises,
            },
            content: conclusion,
            confidence,
            confidence_interval: None,
            pruned: false,
            pruning_rationale: None,
            timestamp,
            parent_id,
            children: Vec::new(),
            metadata: HashMap::new(),
        };

        self.add_node_to_graph(node_id, parent_id, confidence);
        self.nodes.push(node);
        self.reasoning_steps.push(reasoning_step);
        self.confidence_path.push(confidence);
        (node_id, step_id)
    }

    /// Add a decision point
    pub fn add_decision(
        &mut self,
        description: String,
        options: Vec<DecisionOption>,
        chosen: usize,
        rationale: String,
        confidence: f32,
        parent_id: Option<Uuid>,
    ) -> (Uuid, Uuid) {
        let node_id = Uuid::new_v4();
        let decision_id = Uuid::new_v4();
        let timestamp = chrono::Utc::now().to_rfc3339();

        // Separate chosen and pruned options
        let mut chosen_options = Vec::new();
        let mut pruned_options = Vec::new();
        for (idx, mut opt) in options.into_iter().enumerate() {
            if idx == chosen {
                opt.pruned = false;
                chosen_options.push(opt);
            } else {
                opt.pruned = true;
                opt.pruning_rationale = Some(format!("Not chosen (score: {:.3})", opt.score));
                pruned_options.push(opt);
            }
        }

        let decision_point = DecisionPoint {
            decision_id,
            description: description.clone(),
            options: chosen_options.clone(),
            chosen,
            pruned_options: pruned_options.clone(),
            rationale: rationale.clone(),
            confidence,
        };

        let node = ThoughtNode {
            node_id,
            node_type: ThoughtNodeType::Decision {
                options: chosen_options,
                chosen,
                rationale,
            },
            content: description,
            confidence,
            confidence_interval: None,
            pruned: false,
            pruning_rationale: None,
            timestamp,
            parent_id,
            children: Vec::new(),
            metadata: HashMap::new(),
        };

        self.add_node_to_graph(node_id, parent_id, confidence);
        self.nodes.push(node);
        self.decision_points.push(decision_point);
        self.confidence_path.push(confidence);
        (node_id, decision_id)
    }

    /// Add an action node
    pub fn add_action(
        &mut self,
        action_type: String,
        content: String,
        parameters: HashMap<String, String>,
        confidence: f32,
        parent_id: Option<Uuid>,
    ) -> Uuid {
        let node_id = Uuid::new_v4();
        let timestamp = chrono::Utc::now().to_rfc3339();

        let node = ThoughtNode {
            node_id,
            node_type: ThoughtNodeType::Action {
                action_type,
                parameters,
            },
            content,
            confidence,
            confidence_interval: None,
            pruned: false,
            pruning_rationale: None,
            timestamp,
            parent_id,
            children: Vec::new(),
            metadata: HashMap::new(),
        };

        self.add_node_to_graph(node_id, parent_id, confidence);
        self.nodes.push(node);
        self.confidence_path.push(confidence);
        node_id
    }

    /// Add a memory node
    pub fn add_memory(
        &mut self,
        memory_id: String,
        content: String,
        retrieval_score: f32,
        impact: f32,
        parent_id: Option<Uuid>,
    ) -> Uuid {
        let node_id = Uuid::new_v4();
        let timestamp = chrono::Utc::now().to_rfc3339();
        let confidence = (retrieval_score + impact) / 2.0;

        let node = ThoughtNode {
            node_id,
            node_type: ThoughtNodeType::Memory {
                memory_id,
                retrieval_score,
                impact,
            },
            content,
            confidence,
            confidence_interval: None,
            pruned: false,
            pruning_rationale: None,
            timestamp,
            parent_id,
            children: Vec::new(),
            metadata: HashMap::new(),
        };

        self.add_node_to_graph(node_id, parent_id, confidence);
        self.nodes.push(node);
        self.confidence_path.push(confidence);
        node_id
    }

    /// Add node to graph structure
    fn add_node_to_graph(&mut self, node_id: Uuid, parent_id: Option<Uuid>, weight: f32) {
        let node_idx = self.graph.add_node(node_id);
        self.node_map.insert(node_id, node_idx.index());

        if let Some(parent_id) = parent_id {
            if let Some(&parent_idx) = self.node_map.get(&parent_id) {
                self.graph
                    .add_edge(NodeIndex::new(parent_idx), node_idx, weight);
            }
        }
    }

    /// Build the final thought tree
    pub fn build(self) -> ThoughtTree {
        let root_node = self.nodes.first().cloned();
        ThoughtTree {
            root_node,
            nodes: self.nodes,
            reasoning_steps: self.reasoning_steps,
            decision_points: self.decision_points,
            confidence_path: self.confidence_path,
        }
    }
}

impl Default for ThoughtTreeBuilder {
    fn default() -> Self {
        Self::new()
    }
}
