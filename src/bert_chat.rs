//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

use anyhow::Result;
use std::io::{self, Write};
use tracing::{info, error};

fn main() -> Result<()> {
    info!("🤖 BERT Emotion Chat - Starting interactive session");
    info!("Type 'quit' or 'exit' to end the session");
    info!("========================================");

    // Create the BERT emotion analyzer (using fallback mode since we don't have the ONNX model)
    let bert = crate::bert_emotion::BertEmotionAnalyzer::new("models/bert-emotion-model.onnx")?;

    loop {
        tracing::info!("You: ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input == "quit" || input == "exit" {
            info!("👋 Session ended by user");
            break;
        }

        if input.is_empty() {
            continue;
        }

        // Analyze the emotion of the input
        match bert.classify_emotion(input) {
            Ok(emotions) => {
                info!("🤖 BERT Analysis:");
                info!("  Joy: {:.3}", emotions.joy);
                info!("  Sadness: {:.3}", emotions.sadness);
                info!("  Anger: {:.3}", emotions.anger);
                info!("  Fear: {:.3}", emotions.fear);
                info!("  Surprise: {:.3}", emotions.surprise);

                // Determine dominant emotion
                let emotions_vec = [emotions.joy, emotions.sadness, emotions.anger, emotions.fear, emotions.surprise];
                let emotion_names = ["Joy", "Sadness", "Anger", "Fear", "Surprise"];
                let max_idx = emotions_vec.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap().0;

                info!("🎭 Dominant Emotion: {}", emotion_names[max_idx]);
                info!("========================================");
            }
            Err(e) => {
                tracing::error!("❌ BERT Error: {}", e);
                tracing::error!("========================================");
            }
        }
    }

    Ok(())
}
