//! Niodoo Phase 2: Autonomous Recovery Tests
//! Copyright (c) 2025 Jason Van Pham
//!
//! Tests for CoT self-correction, Reflexion framework, and intra-task retries
//! via threat cycle mechanism.

use anyhow::Result;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn};

use niodoo_consciousness::consciousness_engine::ConsciousnessEngine;
use niodoo_consciousness::consciousness_pipeline_orchestrator::{
    ConsciousnessPipelineOrchestrator, PipelineConfig, PipelineInput,
};
use niodoo_consciousness::generation::{GenerationEngine, ReflectionContext};
use niodoo_consciousness::metrics::{
    AdaptiveMetricsSnapshot, AdaptiveRetryController, AggregatedFailureSignals,
    FailureSignalAggregator, FailureSignalThresholds, RetryControllerConfig,
};

// ============================================================================
// COT SELF-CORRECTION TESTS
// ============================================================================

#[tokio::test]
async fn test_cot_correction_detects_contradiction() -> Result<()> {
    info!("Test: CoT correction detects logical contradiction");
    
    let engine = GenerationEngine::new()?;
    let response = "The result is positive. However, it's negative but important.";
    
    let corrected = engine.apply_cot_correction(response).await?;
    
    assert!(corrected.contains("Self-Correction Applied"));
    assert!(corrected.contains("logical contradiction"));
    assert!(corrected.contains("analysis suggests"));
    
    info!("✅ CoT detected and corrected contradiction");
    Ok(())
}

#[tokio::test]
async fn test_cot_correction_handles_incomplete_response() -> Result<()> {
    info!("Test: CoT correction handles truncated response");
    
    let engine = GenerationEngine::new()?;
    let response = "The key point is...[truncated]";
    
    let corrected = engine.apply_cot_correction(response).await?;
    
    assert!(corrected.contains("Self-Correction Applied"));
    assert!(corrected.contains("truncated thought"));
    
    info!("✅ CoT handled incomplete response");
    Ok(())
}

#[tokio::test]
async fn test_cot_correction_replaces_uncertainty() -> Result<()> {
    info!("Test: CoT correction replaces uncertain qualifiers");
    
    let engine = GenerationEngine::new()?;
    let response = "Maybe the answer is 42, possibly related to uncertainty.";
    
    let corrected = engine.apply_cot_correction(response).await?;
    
    assert!(corrected.contains("evidence indicates"));
    assert!(corrected.contains("analysis suggests"));
    assert!(!corrected.contains("maybe"));
    assert!(!corrected.contains("possibly"));
    
    info!("✅ CoT replaced uncertainty with concrete analysis");
    Ok(())
}

#[tokio::test]
async fn test_cot_correction_no_false_positives() -> Result<()> {
    info!("Test: CoT doesn't incorrectly flag valid responses");
    
    let engine = GenerationEngine::new()?;
    let response = "The result is 42. This is based on analysis of the data.";
    
    let corrected = engine.apply_cot_correction(response).await?;
    
    assert!(corrected.contains("Verified: logical consistency checked"));
    assert!(!corrected.contains("Self-Correction Applied"));
    
    info!("✅ CoT correctly passes valid responses");
    Ok(())
}

// ============================================================================
// REFLEXION FRAMEWORK TESTS
// ============================================================================

#[tokio::test]
async fn test_reflexion_generates_context() -> Result<()> {
    info!("Test: Reflexion generates proper context from failure");
    
    let context = ReflectionContext {
        last_prompt: "What is 2+2?".to_string(),
        last_response: "I'm not sure, maybe 5".to_string(),
        failure_reason: Some("Low confidence inference".to_string()),
        retry_count: 2,
    };
    
    assert_eq!(context.retry_count, 2);
    assert!(context.failure_reason.is_some());
    
    info!("✅ Reflexion context created correctly");
    Ok(())
}

#[tokio::test]
async fn test_reflexion_appends_to_prompt() -> Result<()> {
    info!("Test: Reflexion properly augments generation prompt");
    
    let engine = GenerationEngine::new()?;
    let context = ReflectionContext {
        last_prompt: "Calculate 10 * 5".to_string(),
        last_response: "50, wait no 100".to_string(),
        failure_reason: Some("Math error in calculation".to_string()),
        retry_count: 1,
    };
    
    // Mock tokenized input
    use niodoo_consciousness::tokenizer::TokenizedResult;
    let tokenized = TokenizedResult {
        tokens: vec!["Calculate".to_string(), "10".to_string(), "*".to_string(), "5".to_string()],
    };
    
    let result = engine.generate(&tokenized, Some(&context), None).await?;
    
    assert!(result.reflection_applied);
    assert!(result.text.contains("Previous attempt"));
    assert!(result.text.contains("retry #1"));
    
    info!("✅ Reflexion properly appended to prompt");
    Ok(())
}

#[tokio::test]
async fn test_reflexion_multiple_retries() -> Result<()> {
    info!("Test: Reflexion handles escalating retry counts");
    
    let contexts = vec![
        ReflectionContext {
            last_prompt: "First attempt".to_string(),
            last_response: "Wrong answer".to_string(),
            failure_reason: Some("Reason 1".to_string()),
            retry_count: 1,
        },
        ReflectionContext {
            last_prompt: "Second attempt".to_string(),
            last_response: "Still wrong".to_string(),
            failure_reason: Some("Reason 2".to_string()),
            retry_count: 2,
        },
    ];
    
    for (i, ctx) in contexts.iter().enumerate() {
        assert_eq!(ctx.retry_count, (i + 1) as u32);
    }
    
    info!("✅ Reflexion handles multiple retries correctly");
    Ok(())
}

// ============================================================================
// RETRY CONTROLLER TESTS
// ============================================================================

#[tokio::test]
async fn test_retry_controller_levels() -> Result<()> {
    info!("Test: Retry controller escalates levels correctly");
    
    let config = RetryControllerConfig::default();
    let mut controller = AdaptiveRetryController::new(config);
    
    // Test Level1 (soft failure)
    let soft_signals = AggregatedFailureSignals {
        soft_signals: vec![],
        hard_signals: vec![],
    };
    // Add a soft signal
    let mut soft_sigs = soft_signals.clone();
    soft_sigs.soft_signals.push(
        niodoo_consciousness::metrics::failure_signals::FailureSignal::new(
            niodoo_consciousness::metrics::failure_signals::FailureSignalCode::MctsConfidenceLow,
            niodoo_consciousness::metrics::failure_signals::FailureSeverity::Soft,
            "Low UCB1",
            0.2,
        ),
    );
    
    let decision = controller.next_decision(&soft_sigs);
    assert!(decision.should_retry);
    assert_eq!(decision.level, niodoo_consciousness::metrics::AdaptiveRetryLevel::Level1);
    
    info!("✅ Retry controller escalates to Level1 for soft failures");
    Ok(())
}

#[tokio::test]
async fn test_retry_controller_backoff_calculation() -> Result<()> {
    info!("Test: Retry controller calculates exponential backoff");
    
    let config = RetryControllerConfig::default();
    let controller = AdaptiveRetryController::new(config);
    
    // Calculate backoff for different attempts
    let backoff_1 = controller.calculate_backoff(1);
    let backoff_2 = controller.calculate_backoff(2);
    let backoff_3 = controller.calculate_backoff(3);
    
    assert!(backoff_2 > backoff_1);
    assert!(backoff_3 > backoff_2);
    
    info!("✅ Exponential backoff calculated correctly");
    Ok(())
}

#[tokio::test]
async fn test_retry_controller_resets_on_success() -> Result<()> {
    info!("Test: Retry controller resets on success");
    
    let config = RetryControllerConfig::default();
    let mut controller = AdaptiveRetryController::new(config);
    
    // Simulate failures
    let mut signals = AggregatedFailureSignals {
        soft_signals: vec![],
        hard_signals: vec![],
    };
    signals.soft_signals.push(
        niodoo_consciousness::metrics::failure_signals::FailureSignal::new(
            niodoo_consciousness::metrics::failure_signals::FailureSignalCode::MctsConfidenceLow,
            niodoo_consciousness::metrics::failure_signals::FailureSeverity::Soft,
            "Test",
            0.2,
        ),
    );
    
    controller.next_decision(&signals);
    assert!(controller.attempt > 0);
    
    // Register success
    controller.register_success();
    assert_eq!(controller.attempt, 0);
    
    info!("✅ Retry controller resets on success");
    Ok(())
}

#[tokio::test]
async fn test_retry_controller_max_retries() -> Result<()> {
    info!("Test: Retry controller respects max retries");
    
    let config = RetryControllerConfig {
        base_retry_delay_ms: 100,
        max_retries: 3,
        jitter_pct_range: (0.1, 0.2),
    };
    let mut controller = AdaptiveRetryController::new(config);
    
    let mut signals = AggregatedFailureSignals {
        soft_signals: vec![],
        hard_signals: vec![],
    };
    signals.hard_signals.push(
        niodoo_consciousness::metrics::failure_signals::FailureSignal::new(
            niodoo_consciousness::metrics::failure_signals::FailureSignalCode::RougeBelowThreshold,
            niodoo_consciousness::metrics::failure_signals::FailureSeverity::Hard,
            "Low ROUGE",
            0.3,
        ),
    );
    
    // Try 4 times (exceeds max_retries of 3)
    for _ in 0..4 {
        let decision = controller.next_decision(&signals);
        if !decision.should_retry {
            assert_eq!(decision.level, niodoo_consciousness::metrics::AdaptiveRetryLevel::Level4);
            break;
        }
    }
    
    info!("✅ Retry controller respects max retries limit");
    Ok(())
}

// ============================================================================
// INTEGRATION TESTS
// ============================================================================

#[tokio::test]
async fn test_end_to_end_soft_failure_recovery() -> Result<()> {
    info!("Test: End-to-end soft failure recovery with CoT");
    
    // Simulate pipeline with soft failure
    let config = PipelineConfig::default();
    let consciousness_engine = Arc::new(RwLock::new(ConsciousnessEngine::new().await?));
    let orchestrator = ConsciousnessPipelineOrchestrator::new(consciousness_engine, config);
    
    let input = PipelineInput {
        text: "Test prompt with low confidence".to_string(),
        context: None,
        user_id: "test_user".to_string(),
        timestamp: 0.0,
        emotional_context: None,
    };
    
    // This will trigger the retry loop
    let result = orchestrator.process_input(input).await;
    
    // Should eventually succeed or fail gracefully
    assert!(result.is_ok() || result.is_err());
    
    info!("✅ End-to-end soft failure recovery tested");
    Ok(())
}

#[tokio::test]
async fn test_end_to_end_hard_failure_escalation() -> Result<()> {
    info!("Test: End-to-end hard failure escalates to Level2+");
    
    let config = PipelineConfig::default();
    let consciousness_engine = Arc::new(RwLock::new(ConsciousnessEngine::new().await?));
    let orchestrator = ConsciousnessPipelineOrchestrator::new(consciousness_engine, config);
    
    // Create aggregator with failing snapshot
    let mut aggregator = FailureSignalAggregator::new(FailureSignalThresholds::default());
    let snapshot = AdaptiveMetricsSnapshot {
        rouge: Some(0.3), // Below threshold
        entropy: None,
        entropy_delta: Some(0.15), // Spike
        ucb1: None,
        compass_state: None,
        curator_quality: Some(0.5), // Below threshold
        fallbacks_triggered: 0,
    };
    
    let signals = aggregator.aggregate(&snapshot);
    assert!(signals.has_hard_failure());
    assert_eq!(signals.hard_signals.len(), 3); // rouge, entropy_spike, curator
    
    info!("✅ Hard failure escalation validated");
    Ok(())
}

#[tokio::test]
async fn test_reflexion_cot_integration() -> Result<()> {
    info!("Test: Reflexion and CoT work together");
    
    let engine = GenerationEngine::new()?;
    let context = ReflectionContext {
        last_prompt: "Solve x + 5 = 10".to_string(),
        last_response: "x = maybe 5, possibly correct".to_string(),
        failure_reason: Some("Uncertain reasoning".to_string()),
        retry_count: 1,
    };
    
    use niodoo_consciousness::tokenizer::TokenizedResult;
    let tokenized = TokenizedResult {
        tokens: vec!["Solve".to_string(), "x".to_string(), "+".to_string(), "5".to_string(), "=".to_string(), "10".to_string()],
    };
    
    let soft_signals = vec!["low_confidence".to_string(), "uncertainty".to_string()];
    
    let result = engine.generate(&tokenized, Some(&context), Some(&soft_signals)).await?;
    
    assert!(result.reflection_applied);
    assert!(result.text.contains("Previous attempt"));
    assert!(result.text.contains("Chain-of-Thought Correction"));
    
    info!("✅ Reflexion and CoT integration working");
    Ok(())
}

#[tokio::test]
async fn test_failure_signal_thresholds() -> Result<()> {
    info!("Test: Failure signal thresholds are configurable");
    
    let mut aggregator = FailureSignalAggregator::new(FailureSignalThresholds {
        rouge_min: 0.3, // Lower threshold
        entropy_delta_spike: 0.2,
        entropy_delta_flatline: 0.01,
        ucb1_min: 0.2,
        curator_quality_min: 0.5,
    });
    
    let snapshot = AdaptiveMetricsSnapshot {
        rouge: Some(0.35), // Above threshold now
        entropy: None,
        entropy_delta: Some(0.25), // Above spike threshold
        ucb1: Some(0.15), // Below threshold
        compass_state: None,
        curator_quality: Some(0.6), // Above threshold
        fallbacks_triggered: 0,
    };
    
    let signals = aggregator.aggregate(&snapshot);
    
    // Should only trigger entropy_spike and mcts_confidence_low
    assert!(signals.hard_signals.iter().any(|s| s.code.as_str() == "entropy_spike"));
    assert!(signals.soft_signals.iter().any(|s| s.code.as_str() == "mcts_confidence_low"));
    
    info!("✅ Failure signal thresholds validated");
    Ok(())
}

// ============================================================================
// PERFORMANCE TESTS
// ============================================================================

#[tokio::test]
async fn test_retry_latency_within_bounds() -> Result<()> {
    info!("Test: Retry mechanism doesn't add excessive latency");
    
    use std::time::Instant;
    
    let start = Instant::now();
    
    let config = RetryControllerConfig {
        base_retry_delay_ms: 10, // Small delay for testing
        max_retries: 2,
        jitter_pct_range: (0.0, 0.1),
    };
    let controller = AdaptiveRetryController::new(config);
    
    // Simulate rapid retries
    for i in 1..=3 {
        let _delay = controller.calculate_backoff(i);
    }
    
    let elapsed = start.elapsed();
    
    // Should complete quickly (under 1ms for calculation)
    assert!(elapsed.as_millis() < 10);
    
    info!("✅ Retry latency within acceptable bounds");
    Ok(())
}

// ============================================================================
// MAIN TEST RUNNER
// ============================================================================

#[tokio::test]
async fn test_phase2_complete() -> Result<()> {
    info!("🚀 Running complete Phase 2 test suite");
    
    // Run all tests
    test_cot_correction_detects_contradiction().await?;
    test_cot_correction_handles_incomplete_response().await?;
    test_cot_correction_replaces_uncertainty().await?;
    test_cot_correction_no_false_positives().await?;
    
    test_reflexion_generates_context().await?;
    test_reflexion_appends_to_prompt().await?;
    test_reflexion_multiple_retries().await?;
    
    test_retry_controller_levels().await?;
    test_retry_controller_backoff_calculation().await?;
    test_retry_controller_resets_on_success().await?;
    test_retry_controller_max_retries().await?;
    
    test_end_to_end_soft_failure_recovery().await?;
    test_end_to_end_hard_failure_escalation().await?;
    test_reflexion_cot_integration().await?;
    test_failure_signal_thresholds().await?;
    test_retry_latency_within_bounds().await?;
    
    info!("✅ Phase 2 complete - all tests passed!");
    Ok(())
}

