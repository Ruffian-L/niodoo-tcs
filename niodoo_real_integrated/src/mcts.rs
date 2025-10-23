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

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a test state with default values.
    fn test_state() -> PadGhostState {
        PadGhostState {
            pad: [0.5; 7],
            entropy: 0.75,
            mu: [0.0; 7],
            sigma: [0.1; 7],
        }
    }

    #[test]
    fn test_node_creation() {
        let state = test_state();
        let node = MctsNode::new(MctsAction::Retrieve, state.clone(), None);

        assert_eq!(node.action, MctsAction::Retrieve);
        assert_eq!(node.visits, 0);
        assert_eq!(node.total_reward, 0.0);
        assert!(node.parent.is_none());
        assert!(node.children.is_empty());
    }

    #[test]
    fn test_node_update() {
        let state = test_state();
        let mut node = MctsNode::new(MctsAction::DirectAnswer, state, None);

        node.update(0.8);
        assert_eq!(node.visits, 1);
        assert_eq!(node.total_reward, 0.8);

        node.update(0.6);
        assert_eq!(node.visits, 2);
        assert_eq!(node.total_reward, 1.4);
    }

    #[test]
    fn test_avg_reward() {
        let state = test_state();
        let mut node = MctsNode::new(MctsAction::Decompose, state, None);

        assert_eq!(node.avg_reward(), 0.0);

        node.update(0.5);
        assert_eq!(node.avg_reward(), 0.5);

        node.update(0.9);
        assert_eq!(node.avg_reward(), 0.7);
    }

    #[test]
    fn test_ucb1_unvisited() {
        let state = test_state();
        let node = MctsNode::new(MctsAction::Retrieve, state, None);

        let score = node.ucb1(1.414);
        assert_eq!(score, f64::INFINITY);
    }

    #[test]
    fn test_ucb1_with_parent() {
        let state = test_state();
        let parent_state = test_state();

        // Create parent node
        let mut parent = MctsNode::new(MctsAction::Retrieve, parent_state, None);
        parent.update(1.0); // Parent visited once
        parent.update(1.0); // Parent visited twice

        // Create child node
        let mut child = MctsNode::new(MctsAction::Decompose, state, Some(Box::new(parent)));
        child.update(0.8); // Child visited once
        child.update(0.6); // Child visited twice

        let exploration_c = 1.414; // sqrt(2)

        // UCB1 = 0.7 (avg reward) + 1.414 * sqrt(ln(2) / 2)
        // ln(2) ≈ 0.693, sqrt(0.693 / 2) ≈ 0.589
        // UCB1 ≈ 0.7 + 1.414 * 0.589 ≈ 0.7 + 0.833 ≈ 1.533
        let score = child.ucb1(exploration_c);

        // Allow small floating point error
        assert!(score > 1.5 && score < 1.6);
    }

    #[test]
    fn test_ucb1_root_node() {
        let state = test_state();
        let mut node = MctsNode::new(MctsAction::Explore, state, None);

        node.update(0.5);
        node.update(0.7);

        let exploration_c = 1.414;
        let score = node.ucb1(exploration_c);

        // Root node has no parent, so exploration term is 0
        // Score should equal avg_reward
        assert!((score - 0.6).abs() < 0.001);
    }

    #[test]
    fn test_best_child_selection() {
        let parent_state = test_state();
        let mut parent = MctsNode::new(MctsAction::Retrieve, parent_state, None);
        parent.update(1.0);
        parent.update(1.0);
        parent.update(1.0);

        let exploration_c = 1.414;

        // Create child 1: unvisited (should have infinite UCB1)
        let child1_state = test_state();
        let child1 = MctsNode::new(
            MctsAction::Decompose,
            child1_state,
            Some(Box::new(parent.clone())),
        );

        // Create child 2: visited with decent reward
        let child2_state = test_state();
        let mut child2 = MctsNode::new(
            MctsAction::DirectAnswer,
            child2_state,
            Some(Box::new(parent.clone())),
        );
        child2.update(0.7);
        child2.update(0.8);

        // Create child 3: visited with poor reward
        let child3_state = test_state();
        let mut child3 = MctsNode::new(
            MctsAction::Explore,
            child3_state,
            Some(Box::new(parent.clone())),
        );
        child3.update(0.2);
        child3.update(0.1);

        let mut root = MctsNode::new(MctsAction::Retrieve, test_state(), None);
        root.update(1.0);
        root.update(1.0);
        root.update(1.0);

        root.add_child(child1);
        root.add_child(child2);
        root.add_child(child3);

        let best = root.best_child(exploration_c);
        assert!(best.is_some());

        // The unvisited child should be selected (infinite UCB1)
        let best_node = best.unwrap();
        assert_eq!(best_node.action, MctsAction::Decompose);
    }

    #[test]
    fn test_depth() {
        let root = MctsNode::new(MctsAction::Retrieve, test_state(), None);
        assert_eq!(root.depth(), 0);

        let child1 = MctsNode::new(
            MctsAction::Decompose,
            test_state(),
            Some(Box::new(root.clone())),
        );
        assert_eq!(child1.depth(), 1);

        let child2 = MctsNode::new(
            MctsAction::DirectAnswer,
            test_state(),
            Some(Box::new(child1.clone())),
        );
        assert_eq!(child2.depth(), 2);
    }

    #[test]
    fn test_is_leaf() {
        let mut node = MctsNode::new(MctsAction::Retrieve, test_state(), None);
        assert!(node.is_leaf());

        let child = MctsNode::new(MctsAction::Decompose, test_state(), None);
        node.add_child(child);
        assert!(!node.is_leaf());
    }

    #[test]
    fn test_prune_children() {
        let mut node = MctsNode::new(MctsAction::Retrieve, test_state(), None);
        node.add_child(MctsNode::new(MctsAction::Decompose, test_state(), None));
        node.add_child(MctsNode::new(MctsAction::DirectAnswer, test_state(), None));

        assert_eq!(node.children.len(), 2);

        node.prune_children();
        assert_eq!(node.children.len(), 0);
        assert!(node.is_leaf());
    }

    #[test]
    fn test_node_tree_structure() {
        // Build a small tree manually
        let root_state = test_state();
        let mut root = MctsNode::new(MctsAction::Retrieve, root_state, None);

        // Level 1 children
        let child1_state = test_state();
        let mut child1 = MctsNode::new(MctsAction::Decompose, child1_state, None);

        let child2_state = test_state();
        let child2 = MctsNode::new(MctsAction::DirectAnswer, child2_state, None);

        // Add children to root
        root.add_child(child1);
        root.add_child(child2);
        assert_eq!(root.children.len(), 2);

        // Verify structure
        assert!(!root.is_leaf());
        assert_eq!(root.depth(), 0);
    }

    #[test]
    fn test_ucb1_comparison() {
        let parent_state = test_state();
        let mut parent = MctsNode::new(MctsAction::Retrieve, parent_state, None);
        parent.update(1.0);
        parent.update(1.0);
        parent.update(1.0);
        parent.update(1.0);

        // Create high-reward child
        let mut high_reward_child = MctsNode::new(
            MctsAction::Decompose,
            test_state(),
            Some(Box::new(parent.clone())),
        );
        high_reward_child.update(1.0);
        high_reward_child.update(1.0);
        high_reward_child.update(1.0);
        high_reward_child.update(1.0);

        // Create low-reward child with fewer visits
        let mut low_reward_child = MctsNode::new(
            MctsAction::DirectAnswer,
            test_state(),
            Some(Box::new(parent.clone())),
        );
        low_reward_child.update(0.1);

        let c = 1.414;
        let high_score = high_reward_child.ucb1(c);
        let low_score = low_reward_child.ucb1(c);

        // High reward child should have higher score
        assert!(high_score > low_score);
    }

    #[test]
    fn test_search_adaptive_basic() {
        let mut root = MctsNode::new(MctsAction::Retrieve, test_state(), None);
        root.update(1.0); // Initialize root with one visit

        let exploration_c = 1.414;

        // Simple reward function: return constant reward
        let reward_fn = |_node: &MctsNode| -> f64 { 0.5 };

        // Run adaptive search with 100ms time limit
        let stats = root.search_adaptive(100, exploration_c, reward_fn);

        // Verify we completed at least one simulation
        assert!(stats.total_simulations > 0);
        assert!(stats.elapsed_time_ms <= 110); // Allow small tolerance
        assert!(stats.nodes_visited > 1);
        assert_eq!(stats.average_reward, 0.5);
    }

    #[test]
    fn test_search_adaptive_respects_simulation_limit() {
        let mut root = MctsNode::new(MctsAction::Retrieve, test_state(), None);
        root.update(1.0);

        let exploration_c = 1.414;
        let mut counter = 0;
        let reward_fn = |_node: &MctsNode| -> f64 {
            counter += 1;
            0.8
        };

        // Run with generous time budget
        let stats = root.search_adaptive(10000, exploration_c, reward_fn);

        // Should not exceed 100 simulations
        assert!(stats.total_simulations <= 100);
    }

    #[test]
    fn test_search_adaptive_statistics() {
        let mut root = MctsNode::new(MctsAction::Retrieve, test_state(), None);
        root.update(1.0);

        let exploration_c = 1.414;
        let reward_fn = |_node: &MctsNode| -> f64 { 0.7 };

        let stats = root.search_adaptive(500, exploration_c, reward_fn);

        // Verify all statistics are populated
        assert!(stats.total_simulations > 0);
        assert!(stats.elapsed_time_ms > 0);
        assert!(stats.nodes_visited > 0);
        assert!(stats.max_depth >= 0);
        assert!((stats.average_reward - 0.7).abs() < 0.001);
        assert!(stats.best_action_idx < 4); // One of 4 actions
    }

    #[test]
    fn test_search_adaptive_time_budget_respected() {
        let mut root = MctsNode::new(MctsAction::Retrieve, test_state(), None);
        root.update(1.0);

        let exploration_c = 1.414;
        let reward_fn = |_node: &MctsNode| -> f64 { 0.5 };

        let start = Instant::now();
        let stats = root.search_adaptive(50, exploration_c, reward_fn);
        let elapsed = start.elapsed().as_millis() as u64;

        // Actual elapsed should be close to budget (within reasonable tolerance)
        assert!(elapsed <= 100); // 50ms budget + 50ms tolerance
        assert!(stats.elapsed_time_ms <= 100);
    }

    #[test]
    fn test_search_adaptive_with_varying_rewards() {
        let mut root = MctsNode::new(MctsAction::Retrieve, test_state(), None);
        root.update(1.0);

        let exploration_c = 1.414;
        let mut call_count = 0;

        let reward_fn = |_node: &MctsNode| -> f64 {
            call_count += 1;
            if call_count % 2 == 0 {
                0.9
            } else {
                0.3
            }
        };

        let stats = root.search_adaptive(200, exploration_c, reward_fn);

        // Average should be somewhere between 0.3 and 0.9
        assert!(stats.average_reward > 0.3 && stats.average_reward < 0.9);
        assert!(stats.total_simulations > 0);
    }
}
