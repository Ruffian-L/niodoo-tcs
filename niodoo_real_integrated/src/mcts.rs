//! Monte Carlo Tree Search implementation for NIODOO.
//!
//! This module provides the foundational MCTS data structures and algorithms
//! for exploring reasoning paths through retrieval-augmented generation.

use crate::torus::PadGhostState;
use std::fmt;
use std::time::{Duration, Instant};

/// Actions available in the MCTS decision tree.
///
/// Each action represents a distinct strategy for processing a query:
/// - Retrieve: Query ERAG to fetch relevant documents
/// - Decompose: Break the query into sub-problems (sub-prompts)
/// - DirectAnswer: Skip retrieval and answer directly from model knowledge
/// - Explore: Retrieve from distant regions of the embedding space
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MctsAction {
    /// Query the ERAG system to retrieve relevant documents.
    Retrieve,
    /// Decompose the query into multiple sub-questions.
    Decompose,
    /// Answer directly without retrieval.
    DirectAnswer,
    /// Explore distant regions of the embedding space.
    Explore,
}

impl fmt::Display for MctsAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MctsAction::Retrieve => write!(f, "Retrieve"),
            MctsAction::Decompose => write!(f, "Decompose"),
            MctsAction::DirectAnswer => write!(f, "DirectAnswer"),
            MctsAction::Explore => write!(f, "Explore"),
        }
    }
}

/// Statistics collected during adaptive MCTS search.
///
/// Tracks the progress and performance of the search process.
#[derive(Debug, Clone)]
pub struct AdaptiveSearchStats {
    /// Total number of simulations completed
    pub total_simulations: usize,
    /// Total elapsed time in milliseconds
    pub elapsed_time_ms: u64,
    /// Number of nodes visited
    pub nodes_visited: usize,
    /// Maximum depth reached
    pub max_depth: usize,
    /// Average reward across all simulations
    pub average_reward: f64,
    /// Best action found (index into actions)
    pub best_action_idx: usize,
    /// Best action's UCB1 score
    pub best_action_score: f64,
}

/// A node in the Monte Carlo Tree Search tree.
///
/// Each node represents a state in the decision tree, storing:
/// - The action taken to reach this state
/// - The emotional/reasoning state (PAD+ghost projection)
/// - Parent-child relationships
/// - Visit counts and accumulated rewards for UCB1 calculation
#[derive(Debug, Clone)]
pub struct MctsNode {
    /// The action taken to reach this node
    pub action: MctsAction,

    /// The emotional/reasoning state at this node
    pub state: PadGhostState,

    /// Parent node (if any)
    pub parent: Option<Box<MctsNode>>,

    /// Child nodes in the decision tree
    pub children: Vec<MctsNode>,

    /// Number of times this node has been visited
    pub visits: usize,

    /// Cumulative reward from all visits
    pub total_reward: f64,
}

impl MctsNode {
    /// Create a new MCTS node.
    ///
    /// # Arguments
    /// * `action` - The action taken to reach this state
    /// * `state` - The PAD+ghost emotional state
    /// * `parent` - Optional parent node
    ///
    /// Returns a new node with visits=0 and total_reward=0.0
    pub fn new(action: MctsAction, state: PadGhostState, parent: Option<Box<MctsNode>>) -> Self {
        Self {
            action,
            state,
            parent,
            children: Vec::new(),
            visits: 0,
            total_reward: 0.0,
        }
    }

    /// Add a child node to this node.
    pub fn add_child(&mut self, child: MctsNode) {
        self.children.push(child);
    }

    /// Get the average reward (exploitation term).
    ///
    /// Returns 0.0 for unvisited nodes.
    pub fn avg_reward(&self) -> f64 {
        if self.visits == 0 {
            0.0
        } else {
            self.total_reward / self.visits as f64
        }
    }

    /// Calculate the UCB1 (Upper Confidence Bound) score for this node.
    ///
    /// UCB1 formula: Q(n) / N(n) + c * sqrt(ln(N(parent)) / N(n))
    ///
    /// Where:
    /// - Q(n) = total_reward
    /// - N(n) = visits count
    /// - N(parent) = parent visits count
    /// - c = exploration_c (typically sqrt(2) ≈ 1.414)
    ///
    /// Unvisited nodes (visits == 0) return infinity to ensure exploration.
    ///
    /// # Arguments
    /// * `exploration_c` - Exploration parameter (typically sqrt(2))
    ///
    /// # Returns
    /// UCB1 score, or f64::INFINITY for unvisited nodes
    pub fn ucb1(&self, exploration_c: f64) -> f64 {
        // Unvisited nodes get infinite score to ensure they are explored
        if self.visits == 0 {
            return f64::INFINITY;
        }

        // Exploitation term: average reward
        let exploitation = self.avg_reward();

        // Exploration term requires parent visit count
        let exploration = if let Some(ref parent) = self.parent {
            let parent_visits = parent.visits as f64;
            if parent_visits > 0.0 {
                exploration_c * (parent_visits.ln() / self.visits as f64).sqrt()
            } else {
                0.0
            }
        } else {
            // Root node has no exploration bonus
            0.0
        };

        exploitation + exploration
    }

    /// Select the best child node using UCB1.
    ///
    /// Iterates through all children and returns the one with the highest UCB1 score.
    /// If no children exist, returns None.
    ///
    /// # Arguments
    /// * `exploration_c` - Exploration parameter (typically sqrt(2))
    ///
    /// # Returns
    /// A reference to the child with the highest UCB1 score, or None if no children exist
    pub fn best_child(&self, exploration_c: f64) -> Option<&MctsNode> {
        if self.children.is_empty() {
            return None;
        }

        self.children.iter().max_by(|a, b| {
            let a_score = a.ucb1(exploration_c);
            let b_score = b.ucb1(exploration_c);
            // Use partial_cmp for f64 comparison, handling NaN gracefully
            a_score
                .partial_cmp(&b_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Update this node with a new visit and reward.
    ///
    /// # Arguments
    /// * `reward` - The reward value to add
    pub fn update(&mut self, reward: f64) {
        self.visits += 1;
        self.total_reward += reward;
    }

    /// Get depth of this node in the tree.
    ///
    /// Root nodes have depth 0.
    pub fn depth(&self) -> usize {
        let mut depth = 0;
        let mut current = self.parent.as_ref();
        while current.is_some() {
            depth += 1;
            current = current.and_then(|p| p.parent.as_ref());
        }
        depth
    }

    /// Check if this is a leaf node (no children).
    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    /// Clear all children and grandchildren.
    pub fn prune_children(&mut self) {
        self.children.clear();
    }

    /// Perform adaptive MCTS search with time budget.
    ///
    /// Runs MCTS simulations until either the time budget is exhausted or
    /// 100 simulations are completed, whichever comes first.
    ///
    /// Each simulation performs the standard MCTS phases:
    /// 1. **Selection**: Traverse tree following UCB1 until reaching a leaf
    /// 2. **Expansion**: Create children for visited leaf nodes
    /// 3. **Simulation**: Evaluate leaf node with reward function
    /// 4. **Backpropagation**: Update leaf node with reward
    ///
    /// # Arguments
    /// * `max_time_ms` - Maximum time budget in milliseconds
    /// * `exploration_c` - Exploration parameter for UCB1 (typically sqrt(2))
    /// * `reward_fn` - Callback function to compute reward for a path
    ///
    /// # Returns
    /// AdaptiveSearchStats containing search metrics and best action info
    pub fn search_adaptive<F>(
        &mut self,
        max_time_ms: u64,
        exploration_c: f64,
        mut reward_fn: F,
    ) -> AdaptiveSearchStats
    where
        F: FnMut(&MctsNode) -> f64,
    {
        let start_time = Instant::now();
        let time_limit = Duration::from_millis(max_time_ms);
        let max_simulations = 100;

        let mut simulation_count = 0;
        let mut total_nodes_visited = 1; // Count root
        let mut max_depth_reached = 0;
        let mut cumulative_reward = 0.0;

        // Ensure root is visited at least once
        if self.visits == 0 {
            self.update(0.0);
        }

        // Run simulations until time limit or max simulations reached
        while simulation_count < max_simulations && start_time.elapsed() < time_limit {
            // Selection & Expansion phase: traverse and expand tree
            let (reward, depth, nodes_added) = self.simulate_one(exploration_c, &mut reward_fn);

            cumulative_reward += reward;
            total_nodes_visited += nodes_added;
            max_depth_reached = max_depth_reached.max(depth);
            simulation_count += 1;
        }

        let mut elapsed_ms = start_time.elapsed().as_millis() as u64;
        if simulation_count > 0 && elapsed_ms == 0 {
            elapsed_ms = 1;
        }

        // Find best child by visit count (exploitation)
        let mut best_action_idx = 0;
        let mut best_visits = 0usize;

        for (idx, child) in self.children.iter().enumerate() {
            if child.visits > best_visits {
                best_visits = child.visits;
                best_action_idx = idx;
            }
        }

        // Calculate best action score by UCB1 as secondary metric
        let best_action_score = if self.children.is_empty() {
            0.0
        } else {
            self.children[best_action_idx].ucb1(exploration_c)
        };

        AdaptiveSearchStats {
            total_simulations: simulation_count,
            elapsed_time_ms: elapsed_ms,
            nodes_visited: total_nodes_visited,
            max_depth: max_depth_reached,
            average_reward: if simulation_count > 0 {
                cumulative_reward / simulation_count as f64
            } else {
                0.0
            },
            best_action_idx,
            best_action_score,
        }
    }

    /// Perform a single MCTS simulation (one iteration).
    ///
    /// Returns (reward, depth_reached, nodes_added)
    fn simulate_one<F>(&mut self, exploration_c: f64, reward_fn: &mut F) -> (f64, usize, usize)
    where
        F: FnMut(&MctsNode) -> f64,
    {
        let mut nodes_added = 0;

        // Selection phase: select or create a child to explore
        if self.is_leaf() && self.visits > 0 {
            // Expand this leaf: create all 4 action children
            for action in &[
                MctsAction::Retrieve,
                MctsAction::Decompose,
                MctsAction::DirectAnswer,
                MctsAction::Explore,
            ] {
                let new_state = self.state.clone();
                self.add_child(MctsNode::new(*action, new_state, None));
                nodes_added += 1;
            }
        }

        // If we now have children, pick the best one
        if !self.is_leaf() {
            if let Some(best_child) = self.children.iter_mut().max_by(|a, b| {
                a.ucb1(exploration_c)
                    .partial_cmp(&b.ucb1(exploration_c))
                    .unwrap_or(std::cmp::Ordering::Equal)
            }) {
                // Recursively simulate child and get reward
                let (child_reward, child_depth, child_nodes) =
                    best_child.simulate_one(exploration_c, reward_fn);
                best_child.update(child_reward);
                return (child_reward, child_depth + 1, nodes_added + child_nodes);
            }
        }

        // Terminal case: evaluate leaf node
        let reward = reward_fn(self);
        self.update(reward);
        (reward, 0, nodes_added)
    }
}
