pub mod api_clients;
pub mod compass;
pub mod config;
pub mod curator;
pub mod curator_parser;
pub mod data;
pub mod embedding;
pub mod erag;
pub mod generation;
pub mod learning;
pub mod lora_trainer;
pub mod mcts;
pub mod metrics;
pub mod pipeline;
pub mod tcs_analysis;
pub mod tokenizer;
pub mod torus;
pub mod util;

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_run_retry_loop() {
        // Mock pipeline with config
        let mut pipeline = Pipeline::initialise(CliArgs::default()).await.unwrap();
        let tokenizer_output = TokenizerOutput { /* mock */ };
        let compass = CompassOutcome { /* mock */ };
        let collapse = CollapseResult { /* mock */ };

        let outcome = pipeline
            .run_retry_loop("test prompt", &tokenizer_output, &compass, &collapse)
            .await
            .unwrap();
        assert!(outcome.failure_tier == "none" || outcome.updated_counts.total_retries > 0);
    }
}
