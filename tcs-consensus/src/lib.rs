// Copyright 2025 Jason Van Pham (Niodoo)
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of Niodoo TCS.
// For commercial licensing, see LICENSE-COMMERCIAL.md

//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham
//!
//! Consensus and shared vocabulary scaffolding for the Topological Cognitive System.

use uuid::Uuid;

/// Placeholder token proposal structure.
#[derive(Debug, Clone)]
pub struct TokenProposal {
    pub id: Uuid,
    pub persistence_score: f32,
    pub emotional_coherence: f32,
}

/// Single-node threshold-based acceptance helper for prototype pipelines.
pub struct ThresholdConsensus {
    threshold: f32,
}

impl ThresholdConsensus {
    pub fn new(threshold: f32) -> Self {
        Self { threshold }
    }

    pub fn propose(&self, proposal: &TokenProposal) -> bool {
        proposal.persistence_score >= self.threshold
    }
}

#[deprecated = "Use ThresholdConsensus; this alias remains during the transition away from the ConsensusModule name."]
pub type ConsensusModule = ThresholdConsensus;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn honors_threshold() {
        let module = ThresholdConsensus::new(0.8);
        let accept = TokenProposal {
            id: Uuid::new_v4(),
            persistence_score: 0.85,
            emotional_coherence: 0.5,
        };
        let reject = TokenProposal {
            id: Uuid::new_v4(),
            persistence_score: 0.65,
            emotional_coherence: 0.5,
        };

        assert!(module.propose(&accept));
        assert!(!module.propose(&reject));
    }
}

pub mod hotstuff {
    use std::collections::HashSet;
    use std::error::Error;
    use std::fmt;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_PROPOSAL_ID: AtomicU64 = AtomicU64::new(1);

    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct FakeNode {
        pub id: usize,
        pub voting_power: u64,
    }

    impl FakeNode {
        pub fn new(id: usize) -> Self {
            Self {
                id,
                voting_power: 1,
            }
        }
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct Proposal {
        pub id: u64,
        pub proposer: usize,
        pub value: String,
        pub total_nodes: usize,
    }

    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    pub struct Vote {
        pub node_id: usize,
        pub proposal_id: u64,
        pub signature: String,
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct Commit {
        pub proposal_id: u64,
        pub value: String,
        pub voters: Vec<usize>,
    }

    #[derive(Debug, PartialEq, Eq)]
    pub enum HotStuffError {
        EmptyCommittee,
        UnknownProposer(usize),
        EmptyProposal,
        MismatchedClusterSize {
            proposal_nodes: usize,
            cluster_nodes: usize,
        },
        DuplicateVote(usize),
        InsufficientVotes {
            proposal_id: u64,
            needed: usize,
            actual: usize,
        },
    }

    impl fmt::Display for HotStuffError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                HotStuffError::EmptyCommittee => {
                    write!(f, "committee must contain at least one node")
                }
                HotStuffError::UnknownProposer(id) => write!(f, "unknown proposer node id {id}"),
                HotStuffError::EmptyProposal => write!(f, "proposal payload cannot be empty"),
                HotStuffError::MismatchedClusterSize {
                    proposal_nodes,
                    cluster_nodes,
                } => write!(
                    f,
                    "proposal expects {proposal_nodes} nodes but cluster has {cluster_nodes}"
                ),
                HotStuffError::DuplicateVote(node_id) => {
                    write!(f, "duplicate vote detected from node {node_id}")
                }
                HotStuffError::InsufficientVotes {
                    proposal_id,
                    needed,
                    actual,
                } => write!(
                    f,
                    "proposal {proposal_id} only collected {actual} votes, need {needed} for commit"
                ),
            }
        }
    }

    impl Error for HotStuffError {}

    fn ensure_non_empty(nodes: &Arc<[FakeNode]>) -> Result<(), HotStuffError> {
        if nodes.is_empty() {
            Err(HotStuffError::EmptyCommittee)
        } else {
            Ok(())
        }
    }

    pub fn supermajority_threshold(total_nodes: usize) -> usize {
        match total_nodes {
            0 => 0,
            _ => (2 * total_nodes) / 3 + 1,
        }
    }

    pub async fn propose(
        nodes: Arc<[FakeNode]>,
        proposer: usize,
        value: String,
    ) -> Result<Proposal, HotStuffError> {
        ensure_non_empty(&nodes)?;
        if !nodes.iter().any(|node| node.id == proposer) {
            return Err(HotStuffError::UnknownProposer(proposer));
        }
        let payload = value.trim().to_owned();
        if payload.is_empty() {
            return Err(HotStuffError::EmptyProposal);
        }
        let proposal_id = NEXT_PROPOSAL_ID.fetch_add(1, Ordering::Relaxed);
        Ok(Proposal {
            id: proposal_id,
            proposer,
            value: payload,
            total_nodes: nodes.len(),
        })
    }

    pub async fn vote(
        nodes: Arc<[FakeNode]>,
        proposal: &Proposal,
    ) -> Result<Vec<Vote>, HotStuffError> {
        ensure_non_empty(&nodes)?;
        if proposal.total_nodes != nodes.len() {
            return Err(HotStuffError::MismatchedClusterSize {
                proposal_nodes: proposal.total_nodes,
                cluster_nodes: nodes.len(),
            });
        }

        let mut seen = HashSet::with_capacity(nodes.len());
        let mut votes = Vec::with_capacity(nodes.len());

        for node in nodes.iter() {
            if !seen.insert(node.id) {
                return Err(HotStuffError::DuplicateVote(node.id));
            }
            votes.push(Vote {
                node_id: node.id,
                proposal_id: proposal.id,
                signature: format!("sig:node-{}:proposal-{}", node.id, proposal.id),
            });
        }

        Ok(votes)
    }

    pub async fn commit(
        nodes: Arc<[FakeNode]>,
        proposal: &Proposal,
        votes: &[Vote],
    ) -> Result<Commit, HotStuffError> {
        ensure_non_empty(&nodes)?;
        if proposal.total_nodes != nodes.len() {
            return Err(HotStuffError::MismatchedClusterSize {
                proposal_nodes: proposal.total_nodes,
                cluster_nodes: nodes.len(),
            });
        }

        let threshold = supermajority_threshold(nodes.len());
        let mut unique_voters = HashSet::with_capacity(votes.len());

        for vote in votes.iter() {
            if vote.proposal_id != proposal.id {
                continue;
            }
            if !unique_voters.insert(vote.node_id) {
                return Err(HotStuffError::DuplicateVote(vote.node_id));
            }
        }

        let actual = unique_voters.len();
        if actual < threshold {
            return Err(HotStuffError::InsufficientVotes {
                proposal_id: proposal.id,
                needed: threshold,
                actual,
            });
        }

        let mut voters: Vec<usize> = unique_voters.into_iter().collect();
        voters.sort_unstable();

        Ok(Commit {
            proposal_id: proposal.id,
            value: proposal.value.clone(),
            voters,
        })
    }
}

#[cfg(test)]
mod hotstuff_tests {
    use super::hotstuff::{FakeNode, commit, propose, supermajority_threshold, vote};
    use std::future::Future;
    use std::ptr;
    use std::sync::Arc;
    use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};
    use std::thread;

    fn block_on<F: Future>(future: F) -> F::Output {
        fn raw_waker() -> RawWaker {
            unsafe fn clone(_: *const ()) -> RawWaker {
                raw_waker()
            }
            unsafe fn wake(_: *const ()) {}
            unsafe fn wake_by_ref(_: *const ()) {}
            unsafe fn drop(_: *const ()) {}
            RawWaker::new(
                ptr::null(),
                &RawWakerVTable::new(clone, wake, wake_by_ref, drop),
            )
        }

        let waker = unsafe { Waker::from_raw(raw_waker()) };
        let mut future = Box::pin(future);

        loop {
            match future.as_mut().poll(&mut Context::from_waker(&waker)) {
                Poll::Ready(value) => return value,
                Poll::Pending => thread::yield_now(),
            }
        }
    }

    #[test]
    fn hotstuff_multithread_consensus() {
        let nodes: Arc<[FakeNode]> = (0..7).map(FakeNode::new).collect::<Vec<_>>().into();
        let rounds = 6usize;

        std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(rounds);

            for round in 0..rounds {
                let nodes_for_thread = Arc::clone(&nodes);
                handles.push(scope.spawn(move || {
                    let proposer = nodes_for_thread[round % nodes_for_thread.len()].id;
                    block_on(async move {
                        let cluster = nodes_for_thread;
                        let proposal = propose(cluster.clone(), proposer, format!("block-{round}"))
                            .await
                            .expect("proposal should succeed");
                        let votes = vote(cluster.clone(), &proposal)
                            .await
                            .expect("vote collection should succeed");
                        commit(cluster, &proposal, &votes)
                            .await
                            .expect("commit should reach quorum")
                    })
                }));
            }

            let commits: Vec<_> = handles
                .into_iter()
                .map(|handle| handle.join().expect("consensus thread panicked"))
                .collect();

            assert_eq!(commits.len(), rounds);
            let threshold = supermajority_threshold(nodes.len());
            for (round, commit) in commits.iter().enumerate() {
                assert_eq!(commit.value, format!("block-{round}"));
                assert!(commit.voters.len() >= threshold);
            }
        });
    }
}
