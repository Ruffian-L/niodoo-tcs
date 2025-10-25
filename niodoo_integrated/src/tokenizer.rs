use anyhow::Result;
use rand::prelude::*;
use tokio::time::{sleep, Duration};
use crate::emotional_mapping::EmotionalState;
use crate::erag::ERAGResult;

#[derive(Debug)]
pub struct TokenizedResult {
    pub tokens: Vec<String>,
    pub promotions: Vec<String>,
    pub mirage_applied: bool,
}

#[derive(Debug)]
pub struct TokenizerEngine {
    mirage_sigma: f64,
    crdt_state: std::collections::HashMap<String, u64>, // Token -> version
}

impl TokenizerEngine {
    pub fn new(mirage_sigma: f64) -> Result<Self> {
        Ok(Self {
            mirage_sigma,
            crdt_state: std::collections::HashMap::new(),
        })
    }

    pub async fn process(&mut self, erag_result: &ERAGResult, emotional_state: &EmotionalState) -> Result<TokenizedResult> {
        // Simulate CRDT processing time
        sleep(Duration::from_millis(5)).await;

        // CRDT promotion: merge contexts with version control
        let mut tokens = Vec::new();
        let mut promotions = Vec::new();

        // Tokenize the collapsed context
        let words: Vec<String> = erag_result.collapsed_context.split_whitespace()
            .map(|s| s.to_string()).collect();

        for word in words {
            // Check if token needs promotion
            let current_version = self.crdt_state.get(&word).unwrap_or(&0);
            let new_version = current_version + 1;
            self.crdt_state.insert(word.clone(), new_version);

            if new_version > 1 {
                promotions.push(format!("Promoted token '{}' to v{}", word, new_version));
            }

            tokens.push(word);
        }

        // Rut mirage: apply noise if entropy is low
        let mirage_applied = if emotional_state.entropy < 1.0 {
            let mut rng = thread_rng();
            for token in tokens.iter_mut() {
                if rng.gen_bool(0.1) { // 10% chance
                    *token = format!("{}_mirage_{:.2}", token, rng.gen_range(-self.mirage_sigma..self.mirage_sigma));
                }
            }
            true
        } else {
            false
        };

        Ok(TokenizedResult {
            tokens,
            promotions,
            mirage_applied,
        })
    }
}
