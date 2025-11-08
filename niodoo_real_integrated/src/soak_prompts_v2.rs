//! Curated exploration prompts for soak testing v2
//! Provides easy and hard prompts with cycle-aware scheduling

use once_cell::sync::Lazy;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptDifficulty {
    Easy,
    Hard,
}

#[derive(Debug, Clone)]
pub struct PromptEntry {
    pub id: u32,
    pub difficulty: PromptDifficulty,
    pub text: String,
}

impl PromptEntry {
    pub fn to_prompt(&self) -> String {
        self.text.clone()
    }
}

pub const EASY_PER_CYCLE: usize = 3;
pub const HARD_PER_CYCLE: usize = 2;
pub const PROMPTS_PER_CYCLE: usize = EASY_PER_CYCLE + HARD_PER_CYCLE;

pub fn easy_prompts() -> &'static [PromptEntry] {
    &*EASY_PROMPTS
}

pub fn hard_prompts() -> &'static [PromptEntry] {
    &*HARD_PROMPTS
}

static EASY_PROMPTS: Lazy<[PromptEntry; 15]> = Lazy::new(|| [
    PromptEntry {
        id: 1,
        difficulty: PromptDifficulty::Easy,
        text: String::from("What is the capital of France?"),
    },
    PromptEntry {
        id: 2,
        difficulty: PromptDifficulty::Easy,
        text: "Explain how photosynthesis works in simple terms.".to_string(),
    },
    PromptEntry {
        id: 3,
        difficulty: PromptDifficulty::Easy,
        text: "What are the three states of matter?".to_string(),
    },
    PromptEntry {
        id: 4,
        difficulty: PromptDifficulty::Easy,
        text: "Describe the water cycle briefly.".to_string(),
    },
    PromptEntry {
        id: 5,
        difficulty: PromptDifficulty::Easy,
        text: "What is the difference between a plant and an animal cell?".to_string(),
    },
    PromptEntry {
        id: 6,
        difficulty: PromptDifficulty::Easy,
        text: "Explain what gravity is.".to_string(),
    },
    PromptEntry {
        id: 7,
        difficulty: PromptDifficulty::Easy,
        text: "What causes day and night?".to_string(),
    },
    PromptEntry {
        id: 8,
        difficulty: PromptDifficulty::Easy,
        text: "Describe the structure of an atom.".to_string(),
    },
    PromptEntry {
        id: 9,
        difficulty: PromptDifficulty::Easy,
        text: "What is the difference between weather and climate?".to_string(),
    },
    PromptEntry {
        id: 10,
        difficulty: PromptDifficulty::Easy,
        text: "Explain how magnets work.".to_string(),
    },
    PromptEntry {
        id: 11,
        difficulty: PromptDifficulty::Easy,
        text: "What is the purpose of the circulatory system?".to_string(),
    },
    PromptEntry {
        id: 12,
        difficulty: PromptDifficulty::Easy,
        text: "Describe the process of evaporation.".to_string(),
    },
    PromptEntry {
        id: 13,
        difficulty: PromptDifficulty::Easy,
        text: "What are the main components of the solar system?".to_string(),
    },
    PromptEntry {
        id: 14,
        difficulty: PromptDifficulty::Easy,
        text: "Explain what a chemical reaction is.".to_string(),
    },
    PromptEntry {
        id: 15,
        difficulty: PromptDifficulty::Easy,
        text: "What is the role of DNA in living organisms?".to_string(),
    },
]);

static HARD_PROMPTS: Lazy<[PromptEntry; 10]> = Lazy::new(|| [
    PromptEntry {
        id: 101,
        difficulty: PromptDifficulty::Hard,
        text: "Analyze the philosophical implications of quantum entanglement on free will and determinism, considering both Copenhagen and many-worlds interpretations.".to_string(),
    },
    PromptEntry {
        id: 102,
        difficulty: PromptDifficulty::Hard,
        text: "Compare and contrast the economic theories of Keynesian and Austrian schools, and explain how each would address a modern recession with high inflation.".to_string(),
    },
    PromptEntry {
        id: 103,
        difficulty: PromptDifficulty::Hard,
        text: "Discuss the ethical considerations of using CRISPR gene editing for human enhancement versus therapeutic purposes, including potential long-term societal impacts.".to_string(),
    },
    PromptEntry {
        id: 104,
        difficulty: PromptDifficulty::Hard,
        text: "Examine the relationship between information theory, entropy, and the arrow of time in thermodynamics, and how this connects to consciousness studies.".to_string(),
    },
    PromptEntry {
        id: 105,
        difficulty: PromptDifficulty::Hard,
        text: "Evaluate the trade-offs between centralized and decentralized governance models in blockchain systems, considering scalability, security, and democratic participation.".to_string(),
    },
    PromptEntry {
        id: 106,
        difficulty: PromptDifficulty::Hard,
        text: "Analyze how machine learning interpretability requirements conflict with model complexity, and propose a framework for balancing these competing needs in high-stakes applications.".to_string(),
    },
    PromptEntry {
        id: 107,
        difficulty: PromptDifficulty::Hard,
        text: "Discuss the implications of Gödel's incompleteness theorems for artificial general intelligence, particularly regarding self-modification and recursive reasoning capabilities.".to_string(),
    },
    PromptEntry {
        id: 108,
        difficulty: PromptDifficulty::Hard,
        text: "Examine the role of topology in understanding neural network generalization, including how persistent homology might reveal hidden structure in learned representations.".to_string(),
    },
    PromptEntry {
        id: 109,
        difficulty: PromptDifficulty::Hard,
        text: "Compare the computational complexity of different consensus algorithms in distributed systems, analyzing their behavior under Byzantine failures and network partitions.".to_string(),
    },
    PromptEntry {
        id: 110,
        difficulty: PromptDifficulty::Hard,
        text: "Evaluate how emergent properties in complex systems challenge reductionist approaches to science, using examples from biology, economics, and information systems.".to_string(),
    },
]);

