use rand::{rngs::StdRng, Rng, SeedableRng};

/// Deterministically generate synthetic prompts for evaluation
pub fn generate_prompts(num: usize, seed: u64) -> Vec<String> {
    let mut rng = StdRng::seed_from_u64(seed);
    let emotions = [
        "joy", "sadness", "anger", "fear", "surprise", "trust", "disgust", "anticipation",
    ];
    let styles = [
        "poetic", "analytical", "casual", "formal", "concise", "elaborate", "empathetic", "stoic",
    ];

    (0..num)
        .map(|i| {
            let e1 = emotions[rng.gen_range(0..emotions.len())];
            let e2 = emotions[rng.gen_range(0..emotions.len())];
            let style = styles[rng.gen_range(0..styles.len())];
            format!(
                "[{}] Write a {} response that resolves a conflict between {} and {}.",
                i + 1,
                style,
                e1,
                e2
            )
        })
        .collect()
}

/// Deterministically derive a synthetic reference for a given prompt
pub fn reference_for(prompt: &str) -> String {
    // Simple heuristic: mirror the emotions and provide a clear resolution.
    // Ensures stable target for ROUGE-L while allowing diversity in candidates.
    let mut parts = Vec::new();
    for token in prompt.split_whitespace() {
        if [
            "joy", "sadness", "anger", "fear", "surprise", "trust", "disgust", "anticipation",
        ]
        .contains(&token.trim_matches(|c: char| !c.is_alphabetic()))
        {
            parts.push(token.trim_matches(|c: char| !c.is_alphabetic()).to_string());
        }
    }
    parts.sort();
    parts.dedup();
    let theme = if parts.is_empty() {
        "balanced emotions".to_string()
    } else {
        parts.join(" and ")
    };
    format!(
        "A clear, compassionate resolution aligning {} with practical steps and calm tone.",
        theme
    )
}



