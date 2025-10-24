
#[tokio::test]
async fn test_run_retry_loop() {
    let mut pipeline = Pipeline::initialise(/* mock args */).await.unwrap();
    
    // Mock prompt/tokenizer/compass/collapse
    let outcome = pipeline.run_retry_loop("test prompt", &mock_tokenizer(), &mock_compass(), &mock_collapse()).await.unwrap();
    
    assert_eq!(outcome.failure_tier, "none");
    assert!(outcome.updated_counts.total_retries > 0);
    // Check counters inc'd
}

// Add mock helpers as needed
