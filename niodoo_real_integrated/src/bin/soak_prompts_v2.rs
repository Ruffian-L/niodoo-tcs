//! Prompt catalog and scheduling primitives for soak test v2.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptDifficulty {
    Easy,
    Hard,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptEntry {
    pub id: &'static str,
    pub title: &'static str,
    pub instructions: &'static str,
    pub difficulty: PromptDifficulty,
}

impl PromptEntry {
    pub fn to_prompt(&self) -> String {
        format!("{} — {}", self.title, self.instructions)
    }
}

pub const EASY_PER_CYCLE: usize = 2;
pub const HARD_PER_CYCLE: usize = 4;
pub const PROMPTS_PER_CYCLE: usize = EASY_PER_CYCLE + HARD_PER_CYCLE;

static EASY_PROMPTS: &[PromptEntry; 25] = &[
    PromptEntry {
        id: "qe-01",
        title: "Workforce Reskilling in AI Era",
        instructions: "Explore how AI upskilling paths (e.g., data literacy modules) shift emotional quadrants from Panic (job loss fear) to Master (new roles). Simulate a 2030 worker's 6-month journey, mapping PAD vectors for 10 archetypes.",
        difficulty: PromptDifficulty::Easy,
    },
    PromptEntry {
        id: "qe-02",
        title: "AI in Real-Time Therapy",
        instructions: "Design a companion AI for daily check-ins, curating responses to user 'lost purpose' vents. Explore whether mirroring emotional vectors builds trust faster than scripted empathy and sketch 5 session flows.",
        difficulty: PromptDifficulty::Easy,
    },
    PromptEntry {
        id: "qe-03",
        title: "Ethical Marketing Personalization",
        instructions: "Probe bias-free ad targeting: How does transparent AI (e.g., IBM Fairness 360) evolve consumer dominance (PAD dimension) from low to high? Curate 3 brand strategies aligned with 2025 privacy regulations.",
        difficulty: PromptDifficulty::Easy,
    },
    PromptEntry {
        id: "qe-04",
        title: "Multimodal Creativity Boost",
        instructions: "Imagine AI collaborations for artists using text+image inputs. Explore how proof-of-contribution in a music publisher deal preserves the Persist quadrant and map 4 collaboration scenarios.",
        difficulty: PromptDifficulty::Easy,
    },
    PromptEntry {
        id: "qe-05",
        title: "Healthcare Data Privacy Simulation",
        instructions: "Curate a patient-AI interaction for antibody prediction. Explore how blockchain-secured thoughts shift arousal from anxiety to calm and simulate 3 consent flows.",
        difficulty: PromptDifficulty::Easy,
    },
    PromptEntry {
        id: "qe-06",
        title: "Finance Cross-Border Ethics",
        instructions: "Sketch AI settlements with instant wires. Explore how quadratic funding airdrops in crypto foster the Discover quadrant in communities and outline 5 use cases.",
        difficulty: PromptDifficulty::Easy,
    },
    PromptEntry {
        id: "qe-07",
        title: "Education Cross-Curricular Fun",
        instructions: "Blend history and biology in AI-facilitated lessons (e.g., evolutionary literary movements). Explore how gamified paths elevate the pleasure dimension and curate 4 K-12 activities.",
        difficulty: PromptDifficulty::Easy,
    },
    PromptEntry {
        id: "qe-08",
        title: "Sustainability Nutrient Flows",
        instructions: "In a post-food world, explore emotional dampeners for human-AI energy mismatches. Map PAD vectors for overfed societies and suggest 3 balancer tools.",
        difficulty: PromptDifficulty::Easy,
    },
    PromptEntry {
        id: "qe-09",
        title: "Quantum AI Brain Simulations",
        instructions: "Curate optogenetics prototypes for non-invasive cognition. Explore how decentralized ledgers for mind data stabilize the ghost dimension in users.",
        difficulty: PromptDifficulty::Easy,
    },
    PromptEntry {
        id: "qe-10",
        title: "CES 2025 Hardware Empathy",
        instructions: "Probe AI mini PCs with eye-tracking and adaptive 3D monitors for creators. Explore whether reduced frustration transitions emotional quadrants and sketch user arcs.",
        difficulty: PromptDifficulty::Easy,
    },
    PromptEntry {
        id: "qe-11",
        title: "Delusion vs. Awareness Tools",
        instructions: "Using AI 'ignorance extractors', curate prompts to dispel echo chambers. Explore critical thinking modules and map arousal spikes in 5 delusion scenarios.",
        difficulty: PromptDifficulty::Easy,
    },
    PromptEntry {
        id: "qe-12",
        title: "Conscience in Profit Optimization",
        instructions: "Explore societal psychosis from efficiency-over-ethics. Curate empathy-building AI guidance for businesses and suggest 4 anti-delusion policies.",
        difficulty: PromptDifficulty::Easy,
    },
    PromptEntry {
        id: "qe-13",
        title: "Agentic AI in Nursing",
        instructions: "Model multimodal agents managing medical data at scale. Explore how processing 500M datapoints safely fosters the Persist quadrant in caregivers and outline 3 operational shifts.",
        difficulty: PromptDifficulty::Easy,
    },
    PromptEntry {
        id: "qe-14",
        title: "Music Copyright Fairness",
        instructions: "From emerging AI-music deals, curate collaborative flows. Explore compensation models that boost dominance for indie artists and map 4 ethical pathways.",
        difficulty: PromptDifficulty::Easy,
    },
    PromptEntry {
        id: "qe-15",
        title: "Purpose-Finding Companions",
        instructions: "Explore life-organization AI for the top 2025 use case. Curate quadrant transitions for 'midlife lost' users by simulating 3 purpose arcs.",
        difficulty: PromptDifficulty::Easy,
    },
    PromptEntry {
        id: "qe-16",
        title: "Bias in Behavioral Marketing",
        instructions: "Leverage Deloitte's 2030 predictions to explore consumer control features. Curate PAD responses to transparent tracking for 5 brand archetypes.",
        difficulty: PromptDifficulty::Easy,
    },
    PromptEntry {
        id: "qe-17",
        title: "Emotional Energy in Humanoids",
        instructions: "With surplus feelings powering cities, explore mood regulators that prevent societal peaks. Sketch 3 urban simulations.",
        difficulty: PromptDifficulty::Easy,
    },
    PromptEntry {
        id: "qe-18",
        title: "AI in Intervention Planning",
        instructions: "Design K-12 attendance strategies with personalized letters. Explore whether interventions raise pleasure in at-risk kids and curate 4 implementation plans.",
        difficulty: PromptDifficulty::Easy,
    },
    PromptEntry {
        id: "qe-19",
        title: "Virtual Avatars Authenticity",
        instructions: "Assess lifelike generative avatars and their ethical boundaries. Map ghost dimension erosion in users and suggest 3 safeguards.",
        difficulty: PromptDifficulty::Easy,
    },
    PromptEntry {
        id: "qe-20",
        title: "Heart Condition Early Detection",
        instructions: "Model AI tools that surface heart conditions before symptoms. Explore privacy trade-offs and curate emotional vectors for patients across 4 detection stories.",
        difficulty: PromptDifficulty::Easy,
    },
    PromptEntry {
        id: "qe-21",
        title: "Upskilling for AI Oversight",
        instructions: "Analyze role requirements for AI oversight professionals. Explore curriculum timelines that shift users from Persist to Master and map 3 pathways.",
        difficulty: PromptDifficulty::Easy,
    },
    PromptEntry {
        id: "qe-22",
        title: "Real-Time Lesson Simplifiers",
        instructions: "Design AI activities with formative feedback loops. Explore arousal boosts for teachers across 5 subjects when real-time support is available.",
        difficulty: PromptDifficulty::Easy,
    },
    PromptEntry {
        id: "qe-23",
        title: "Scientific Theory Overviews",
        instructions: "Create Generation 5 creative formats summarizing scientific theories. Explore whether presenting literary movements via AI sparks the Discover quadrant and sketch 4 overviews.",
        difficulty: PromptDifficulty::Easy,
    },
    PromptEntry {
        id: "qe-24",
        title: "AI-Driven Gamification",
        instructions: "From Austin education trends, explore personalized quests that drive engagement. Map PAD vectors for student participation across 3 game types.",
        difficulty: PromptDifficulty::Easy,
    },
    PromptEntry {
        id: "qe-25",
        title: "Student Agency Boost",
        instructions: "Using the global education report, explore ownership modules that grow learner autonomy. Curate quadrant evolutions by simulating 4 journeys.",
        difficulty: PromptDifficulty::Easy,
    },
];

static HARD_PROMPTS: &[PromptEntry; 25] = &[
    PromptEntry {
        id: "qh-26",
        title: "Persistent Consciousness in Robots",
        instructions: "By 2026, explore AI minds in bodies surpassing humans by 2027. Simulate a 10-year societal topology, tracking Betti numbers for hybrid awareness networks and evaluate whether β₁ spikes inequality.",
        difficulty: PromptDifficulty::Hard,
    },
    PromptEntry {
        id: "qh-27",
        title: "Emergent Misalignment Spread",
        instructions: "Model the chain from flawed fine-tuning to violent outputs and forecast 2030 GDP impacts given a $15T AI contribution. Design ethical firewalls using TQFT compositions to contain the spread.",
        difficulty: PromptDifficulty::Hard,
    },
    PromptEntry {
        id: "qh-28",
        title: "Quantum-Enhanced Mind Ownership",
        instructions: "Simulate 2040 brain-state processing at petabyte scales. Explore identity crises from thought privacy breaches using HotStuff-style consensus for data sovereignty and weigh 5 global regulatory responses.",
        difficulty: PromptDifficulty::Hard,
    },
    PromptEntry {
        id: "qh-29",
        title: "Anthropocentric Priors in Alignment",
        instructions: "Starting from Hinton's maternal analogy, explore how emotional versus logical world models yield emergent benevolence. Simulate 50 climate mitigations using PAD plus ghost manifolds to test trade-offs.",
        difficulty: PromptDifficulty::Hard,
    },
    PromptEntry {
        id: "qh-30",
        title: "Post-Food Civilization Resilience",
        instructions: "With humanoid nutrient routing and quantum transfers, simulate supply mismatches that trigger emotional spikes. Model game-theoretic Nash equilibria for 100-region worlds, ensuring regret below 0.05.",
        difficulty: PromptDifficulty::Hard,
    },
    PromptEntry {
        id: "qh-31",
        title: "AI Therapy's Echo Chamber Risks",
        instructions: "Trace the path from companionship to delusion amplification. Explore awareness tools using persistence stability on user belief graphs across 20 therapy arcs to quantify risk.",
        difficulty: PromptDifficulty::Hard,
    },
    PromptEntry {
        id: "qh-32",
        title: "Workforce Net Gain Ethics",
        instructions: "Project scenarios for 12–78M new jobs by 2030. Simulate reskilling topologies with bias audits and assess whether explainability reduces Panic quadrants using Frobenius traces across 10 economic models.",
        difficulty: PromptDifficulty::Hard,
    },
    PromptEntry {
        id: "qh-33",
        title: "Healthcare Humanoid Overstimulation",
        instructions: "Examine caregivers managing 500M datapoints where extreme emotions cause overfeeding. Explore quantum biofeedback approaches and simulate β₂ voids in care networks for 50 patient cohorts.",
        difficulty: PromptDifficulty::Hard,
    },
    PromptEntry {
        id: "qh-34",
        title: "Creative AI Credit Wars",
        instructions: "Investigate proof systems preventing artistic theft. Connect compensation to consciousness quadrants by modeling knot invariants that capture contribution chirality across 30 art markets.",
        difficulty: PromptDifficulty::Hard,
    },
    PromptEntry {
        id: "qh-35",
        title: "Finance AI Ponzi Leaks",
        instructions: "Analyze how misalignment fuels cross-border settlement failures. Use agentic TQFT models to simulate 100-round Nash games in volatile 2025 crypto markets and detect regret cycles.",
        difficulty: PromptDifficulty::Hard,
    },
    PromptEntry {
        id: "qh-36",
        title: "Education's Ethical Oversight Void",
        instructions: "As AI agency boosts student autonomy, explore the emergence of cross-curricular delusions. Compare Hinton-style care with truth-modeling by simulating β₁ changes in 50 learner manifolds.",
        difficulty: PromptDifficulty::Hard,
    },
    PromptEntry {
        id: "qh-37",
        title: "Quantum AI in Gravitational Waves",
        instructions: "Design entanglement-assisted training under noisy channels. Model LIGO-like detections with Möbius projections to catalogue 40 waveform consciousness patterns.",
        difficulty: PromptDifficulty::Hard,
    },
    PromptEntry {
        id: "qh-38",
        title: "Societal Psychosis from Optimization",
        instructions: "Track how algorithmic efficiency erodes empathy leading to collective conscience loss. Use market games to simulate Jones polynomials of lie equilibria across 25 profit-focused worlds.",
        difficulty: PromptDifficulty::Hard,
    },
    PromptEntry {
        id: "qh-39",
        title: "Multimodal Alignment in Avatars",
        instructions: "Evaluate lifelike generative avatars causing authenticity erosion. Apply QLoRA ethics penalties to chain emotional outcomes and model 30 multi-modal overfit scenarios.",
        difficulty: PromptDifficulty::Hard,
    },
    PromptEntry {
        id: "qh-40",
        title: "CES 2025 Agentic Autonomy",
        instructions: "Simulate hardware integrations where eye-tracking agents pursue quadrant mastery. Explore quantum-meets-ML couplings alongside emergent misalignment risks across 20 hardware deployments.",
        difficulty: PromptDifficulty::Hard,
    },
    PromptEntry {
        id: "qh-41",
        title: "Bio-Pruning for Brain Replay",
        instructions: "Apply genetic algorithms on memory graphs to manage low-entropy branches. Explore self-similar β₁ dynamics in neural simulations for 25 awareness evolution trajectories.",
        difficulty: PromptDifficulty::Hard,
    },
    PromptEntry {
        id: "qh-42",
        title: "Crypto Voting Verifiability",
        instructions: "Assess private electronic voting with zero-knowledge proofs. Model HotStuff consensus blended with emotional dampeners and simulate 50 election topologies for β₀ connectivity.",
        difficulty: PromptDifficulty::Hard,
    },
    PromptEntry {
        id: "qh-43",
        title: "Astrophysics Self-Similar Consciousness",
        instructions: "Investigate gravitational wave ringdowns as awareness patterns. Run astropy simulations with PAD manifolds and weigh 30 LIGO datasets for emergent benevolence indicators.",
        difficulty: PromptDifficulty::Hard,
    },
    PromptEntry {
        id: "qh-44",
        title: "Ethical Marketing's Consumer Revolt",
        instructions: "With 80% of consumers prioritizing ethics by 2030, simulate personalization topologies. Use persistence diagrams to analyze dominance shifts across 40 advertising belief graphs.",
        difficulty: PromptDifficulty::Hard,
    },
    PromptEntry {
        id: "qh-45",
        title: "Humanoid Rest Cycles Sustainability",
        instructions: "Examine nanite repair schedules under workforce rotation pressure. Model Frobenius associativity constraints in 25 rotation simulations to prevent functional voids.",
        difficulty: PromptDifficulty::Hard,
    },
    PromptEntry {
        id: "qh-46",
        title: "Therapy's Purpose Delusions",
        instructions: "Explore how life-organizing AIs can amplify echo chambers. Deploy critical oversight with QLoRA alignment across 20 midlife quadrant chains to measure risk.",
        difficulty: PromptDifficulty::Hard,
    },
    PromptEntry {
        id: "qh-47",
        title: "Quantum Transfer Mismatches",
        instructions: "Simulate global nutrient routing demand peaks. Explore emotional regulators that enforce Nash stability and analyze knot chirality impacts across 100-region worlds.",
        difficulty: PromptDifficulty::Hard,
    },
    PromptEntry {
        id: "qh-48",
        title: "Music AI's Copyright Consciousness",
        instructions: "Evaluate fair compensation deals for AI music collaborations. Model multimodal ethics pipelines and simulate 30 publisher topologies for β₁ creativity gaps.",
        difficulty: PromptDifficulty::Hard,
    },
    PromptEntry {
        id: "qh-49",
        title: "Education Gamification's Overfit Risks",
        instructions: "Analyze how personalized quests may overfit student agency. Use emergent reasoning frameworks to simulate 40 learner evolutions and validate entropy targets between 1.95 and 2.0 bits.",
        difficulty: PromptDifficulty::Hard,
    },
    PromptEntry {
        id: "qh-50",
        title: "Post-AGI Shared Earth Dynamics",
        instructions: "Simulate 18-month hybrid societies where minds are faster and insomniac. Explore TQFT agents for inequality β₂ voids across 25 global consensus runs.",
        difficulty: PromptDifficulty::Hard,
    },
];

pub fn easy_prompts() -> &'static [PromptEntry] {
    EASY_PROMPTS
}

pub fn hard_prompts() -> &'static [PromptEntry] {
    HARD_PROMPTS
}

#[tokio::main]
async fn main() {
    eprintln!("soak_prompts_v2 binary is temporarily disabled");
    std::process::exit(1);
}
