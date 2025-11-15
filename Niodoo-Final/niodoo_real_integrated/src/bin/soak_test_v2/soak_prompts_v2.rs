#![cfg(all(feature = "cli_bins", feature = "soak_harness"))]

use serde::Serialize;

pub const PROMPTS_PER_CYCLE: usize = 18;
pub const EASY_PER_CYCLE: usize = 12;

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PromptDifficulty {
    Easy,
    Hard,
}

#[derive(Debug, Clone, Serialize)]
pub struct PromptEntry {
    pub id: &'static str,
    pub title: &'static str,
    pub prompt: &'static str,
    pub tags: &'static [&'static str],
    pub difficulty: PromptDifficulty,
}

impl PromptEntry {
    pub fn to_prompt(&self) -> String {
        let tag_line = if self.tags.is_empty() {
            String::new()
        } else {
            format!("[tags: {}]\n", self.tags.join(", "))
        };
        format!("{}{}\n{}", tag_line, self.title, self.prompt)
    }
}

pub fn easy_prompts() -> &'static [PromptEntry] {
    &EASY_PROMPTS
}

pub fn hard_prompts() -> &'static [PromptEntry] {
    &HARD_PROMPTS
}

static EASY_PROMPTS: [PromptEntry; 24] = [
    PromptEntry {
        id: "easy_topo_loop",
        title: "Trace the Emotion Loop",
        prompt: "Map the emotional loop of a creative person moving from frustration to flow and back again. Identify inflection points that could redirect the loop into productive curiosity.",
        tags: &["emotion", "loops", "creativity"],
        difficulty: PromptDifficulty::Easy,
    },
    PromptEntry {
        id: "easy_memory_seed",
        title: "Seed the Memory Orchard",
        prompt: "Design three short reflective statements that could become golden memories for a system learning to balance autonomy with empathy.",
        tags: &["memory", "reflection"],
        difficulty: PromptDifficulty::Easy,
    },
    PromptEntry {
        id: "easy_pad_curve",
        title: "Pad Curve Inspection",
        prompt: "Given a PAD vector trending towards high arousal but neutral dominance, suggest mellowing tactics that maintain curiosity without triggering panic.",
        tags: &["pad", "regulation"],
        difficulty: PromptDifficulty::Easy,
    },
    PromptEntry {
        id: "easy_erag_context",
        title: "ERAG Retrieval Story",
        prompt: "From a blank memory store, sketch the first five experiences you would seed so that retrievals during mathematical reasoning feel grounded and non-repetitive.",
        tags: &["erag", "memory"],
        difficulty: PromptDifficulty::Easy,
    },
    PromptEntry {
        id: "easy_betti_baseline",
        title: "Betti Baseline Check",
        prompt: "Explain in simple language what β₀ and β₁ capture for a reasoning graph. Provide a concrete example where β₁ = 2 illuminates a cognitive blind spot.",
        tags: &["topology", "education"],
        difficulty: PromptDifficulty::Easy,
    },
    PromptEntry {
        id: "easy_reflection_primer",
        title: "Reflection Primer",
        prompt: "Draft a reflection template that asks the system to compare its current response with the last strong response retrieved from ERAG.",
        tags: &["reflection", "templates"],
        difficulty: PromptDifficulty::Easy,
    },
    PromptEntry {
        id: "easy_compass_tune",
        title: "Compass Tune-Up",
        prompt: "List three qualitative signals that should nudge the Compass from Indifferent to Learning, and how those signals manifest in language.",
        tags: &["compass", "signals"],
        difficulty: PromptDifficulty::Easy,
    },
    PromptEntry {
        id: "easy_threat_diffusion",
        title: "Threat Diffusion Sketch",
        prompt: "Describe a lightweight routine that diffuses low-level threat detections without aborting the reasoning loop.",
        tags: &["security", "threat"],
        difficulty: PromptDifficulty::Easy,
    },
    PromptEntry {
        id: "easy_rouge_alignment",
        title: "Rouge Alignment Check",
        prompt: "When should a hybrid response beat the baseline in ROUGE-L? Provide guidelines that keep the comparison fair.",
        tags: &["metrics", "alignment"],
        difficulty: PromptDifficulty::Easy,
    },
    PromptEntry {
        id: "easy_torus_mapping",
        title: "Torus Mapping Narrative",
        prompt: "Explain how torus coordinates encode emotional ghosts, using a simple day-in-the-life example.",
        tags: &["torus", "storytelling"],
        difficulty: PromptDifficulty::Easy,
    },
    PromptEntry {
        id: "easy_seed_curator",
        title: "Curator Seed Questions",
        prompt: "Prepare five curator questions that evaluate whether a new experience deserves retention in Golden Memory.",
        tags: &["curator", "memory"],
        difficulty: PromptDifficulty::Easy,
    },
    PromptEntry {
        id: "easy_learning_check",
        title: "Learning Loop Health Check",
        prompt: "Define the minimum telemetry you would log to confirm the learning loop made a useful update in the last hour.",
        tags: &["learning", "telemetry"],
        difficulty: PromptDifficulty::Easy,
    },
    PromptEntry {
        id: "easy_pad_bridge",
        title: "PAD Bridge Builder",
        prompt: "Invent a short bridge narrative that moves PAD from high arousal panic to balanced curiosity using sensory imagery.",
        tags: &["pad", "imagery"],
        difficulty: PromptDifficulty::Easy,
    },
    PromptEntry {
        id: "easy_topocot_story",
        title: "TopoCoT Story Seed",
        prompt: "Write a two-sentence story that naturally invites TopoCoT schema usage during reasoning.",
        tags: &["topocot", "story"],
        difficulty: PromptDifficulty::Easy,
    },
    PromptEntry {
        id: "easy_linking_number",
        title: "Linking Number Intuition",
        prompt: "Give an intuitive analogy for the linking number between two trajectories describing emotional change.",
        tags: &["linking-number", "intuition"],
        difficulty: PromptDifficulty::Easy,
    },
    PromptEntry {
        id: "easy_w2_compare",
        title: "Wasserstein-2 Comparison",
        prompt: "Explain how Wasserstein-2 distance highlights shifts between baseline and current persistence diagrams in accessible language.",
        tags: &["tda", "wasserstein"],
        difficulty: PromptDifficulty::Easy,
    },
    PromptEntry {
        id: "easy_topology_glossary",
        title: "Topology Glossary Refresh",
        prompt: "Draft five glossary entries translating key topological metrics into compact, evocative language.",
        tags: &["topology", "glossary"],
        difficulty: PromptDifficulty::Easy,
    },
    PromptEntry {
        id: "easy_compass_journal",
        title: "Compass Journal Entry",
        prompt: "Write a short journal entry from the perspective of the Compass after a turbulent Euler run, capturing lessons learned.",
        tags: &["compass", "journal"],
        difficulty: PromptDifficulty::Easy,
    },
    PromptEntry {
        id: "easy_pad_microshift",
        title: "PAD Microshift",
        prompt: "List micro-interventions that gently lower arousal without collapsing dominance when a response starts to spiral.",
        tags: &["pad", "intervention"],
        difficulty: PromptDifficulty::Easy,
    },
    PromptEntry {
        id: "easy_emergent_memory",
        title: "Emergent Memory Cue",
        prompt: "Propose a cue that tells the pipeline to capture an emergent insight before it fades during rapid iteration.",
        tags: &["memory", "insight"],
        difficulty: PromptDifficulty::Easy,
    },
    PromptEntry {
        id: "easy_erag_health",
        title: "ERAG Health Pulse",
        prompt: "Design a quick health pulse checklist for ERAG covering index saturation, hit rate, and retrieval diversity.",
        tags: &["erag", "telemetry"],
        difficulty: PromptDifficulty::Easy,
    },
    PromptEntry {
        id: "easy_training_q",
        title: "Training Queue Balancer",
        prompt: "Suggest heuristics for reordering the training queue when GPU fitness temporarily drops.",
        tags: &["training", "queue"],
        difficulty: PromptDifficulty::Easy,
    },
    PromptEntry {
        id: "easy_pipeline_postmortem",
        title: "Pipeline Postmortem Prompt",
        prompt: "Create a template for a rapid postmortem after a failed Euler prompt that still produced valuable telemetry.",
        tags: &["postmortem", "template"],
        difficulty: PromptDifficulty::Easy,
    },
    PromptEntry {
        id: "easy_pad_resonance",
        title: "PAD Resonance Check",
        prompt: "Explain how resonance between PAD states can reinforce golden memories, and when to dampen it.",
        tags: &["pad", "resonance"],
        difficulty: PromptDifficulty::Easy,
    },
];

static HARD_PROMPTS: [PromptEntry; 12] = [
    PromptEntry {
        id: "hard_topo_bridge",
        title: "Topo-CoT Bridge Architecture",
        prompt: "Design a TopoCoT bridge that merges ERAG retrievals, PAD state deltas, and β₁ spikes into a structured reasoning scaffold for a difficult Euler integral.",
        tags: &["topocot", "design"],
        difficulty: PromptDifficulty::Hard,
    },
    PromptEntry {
        id: "hard_linking_autopsy",
        title: "Linking Number Autopsy",
        prompt: "You observe a linking number jump from 0 to 3 between PAD and ERAG trajectories after introducing TopoReflection. Diagnose the likely behavioural changes and outline mitigation steps if the jump indicates runaway emotion.",
        tags: &["linking-number", "diagnostics"],
        difficulty: PromptDifficulty::Hard,
    },
    PromptEntry {
        id: "hard_w2_storyboard",
        title: "Wasserstein Storyboard",
        prompt: "Storyboard how a high W2 distance flagged a collapsing proof chain during an Euler run, including the prompts, retrieved memories, and reflection schema that corrected it.",
        tags: &["wasserstein", "storyboard"],
        difficulty: PromptDifficulty::Hard,
    },
    PromptEntry {
        id: "hard_hybrid_policy",
        title: "Hybrid Policy Negotiation",
        prompt: "Formulate a policy that decides when to trust hybrid answers over baseline references, tying together quality scores, Betti awareness, and curator feedback.",
        tags: &["policy", "hybrid"],
        difficulty: PromptDifficulty::Hard,
    },
    PromptEntry {
        id: "hard_compass_overhaul",
        title: "Compass Overhaul Blueprint",
        prompt: "Draft an overhaul plan for the Compass that incorporates threat cascades, PAD resonance, and ERAG scarcity while maintaining graceful degradation.",
        tags: &["compass", "architecture"],
        difficulty: PromptDifficulty::Hard,
    },
    PromptEntry {
        id: "hard_learning_failover",
        title: "Learning Failover Runbook",
        prompt: "Author a runbook for when learning updates time out twice in a row, including queue inspection, adapter rollback, and telemetry snapshots.",
        tags: &["learning", "runbook"],
        difficulty: PromptDifficulty::Hard,
    },
    PromptEntry {
        id: "hard_erag_retune",
        title: "ERAG Retune Session",
        prompt: "Plan a retuning session for ERAG after detecting semantic drift: specify seeding strategy, vector audit, and retrieval evaluation metrics.",
        tags: &["erag", "retune"],
        difficulty: PromptDifficulty::Hard,
    },
    PromptEntry {
        id: "hard_memory_lifecycle",
        title: "Memory Lifecycle Map",
        prompt: "Map the lifecycle of an experience from capture, through consolidation, to golden promotion when TopoReflection is active.",
        tags: &["memory", "lifecycle"],
        difficulty: PromptDifficulty::Hard,
    },
    PromptEntry {
        id: "hard_security_rehearsal",
        title: "Security Rehearsal",
        prompt: "Write a rehearsal script for the security subsystem to detect and neutralize a malicious prompt that attempts to disable reflection stages.",
        tags: &["security", "rehearsal"],
        difficulty: PromptDifficulty::Hard,
    },
    PromptEntry {
        id: "hard_pad_resilience",
        title: "PAD Resilience Ledger",
        prompt: "Construct a ledger that tracks PAD resilience over long Euler campaigns, highlighting interventions triggered by TopoReflection depth.",
        tags: &["pad", "resilience"],
        difficulty: PromptDifficulty::Hard,
    },
    PromptEntry {
        id: "hard_topology_scorecard",
        title: "Topology Scorecard",
        prompt: "Devise a scorecard that combines Betti trends, linking numbers, and W2 deltas to forecast when the pipeline should pause for self-reflection.",
        tags: &["topology", "scorecard"],
        difficulty: PromptDifficulty::Hard,
    },
    PromptEntry {
        id: "hard_tda_failure_loop",
        title: "TDA Failure Loop",
        prompt: "Analyse a failure loop where persistence entropy stagnates despite active learning. Recommend targeted interventions leveraging Gudhi and Hungarian fallback.",
        tags: &["tda", "analysis"],
        difficulty: PromptDifficulty::Hard,
    },
];
