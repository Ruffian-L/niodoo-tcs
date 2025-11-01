//! Resource Budget Stress Test
//!
//! Simulates resource exhaustion scenarios to validate graceful degradation
//! and ensure the system survives without crashes.

use anyhow::Result;
use niodoo_real_integrated::degradation_tiers::DegradationManager;
use niodoo_real_integrated::resource_budget::GlobalResourceBudget;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;
use tracing::info;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    info!("🧪 Starting Resource Budget Stress Test");

    // Test 1: Gradual token exhaustion
    info!("Test 1: Gradual token exhaustion");
    test_gradual_exhaustion().await?;

    // Test 2: Sudden resource depletion
    info!("Test 2: Sudden resource depletion");
    test_sudden_depletion().await?;

    // Test 3: Recovery after exhaustion
    info!("Test 3: Recovery after exhaustion");
    test_recovery().await?;

    // Test 4: Degradation tier transitions
    info!("Test 4: Degradation tier transitions");
    test_degradation_tiers().await?;

    info!("✅ All resource budget tests passed!");
    Ok(())
}

async fn test_gradual_exhaustion() -> Result<()> {
    let budget = Arc::new(GlobalResourceBudget::new(1000, 100, 10000, 1000));
    let degradation = Arc::new(DegradationManager::new(
        niodoo_real_integrated::config::DegradationConfig::default(),
    ));

    // Consume tokens gradually
    for i in 0..10 {
        let success = budget.consume_tokens(100);
        assert!(success, "Should be able to consume tokens");

        let overall = budget.get_overall_availability();

        info!(
            iteration = i,
            overall_availability = overall,
            "Gradual consumption"
        );

        // Check degradation tier
        let tier = degradation.determine_tier(overall);
        info!(tier = ?tier, "Current degradation tier");

        sleep(Duration::from_millis(100)).await;
    }

    // Should have triggered degradation
    let final_availability = budget.get_overall_availability();
    assert!(final_availability < 0.5, "Should have degraded below 50%");

    Ok(())
}

async fn test_sudden_depletion() -> Result<()> {
    let budget = Arc::new(GlobalResourceBudget::new(1000, 100, 10000, 1000));

    // Consume most tokens at once
    let success = budget.consume_tokens(900);
    assert!(success, "Should be able to consume tokens");

    let availability = budget.get_overall_availability();
    assert!(availability < 0.2, "Should be in emergency tier");

    info!(availability = availability, "Sudden depletion test passed");

    Ok(())
}

async fn test_recovery() -> Result<()> {
    let budget = Arc::new(GlobalResourceBudget::new(1000, 100, 10000, 1000));

    // Exhaust resources
    budget.consume_tokens(1000);
    let availability_before = budget.get_overall_availability();
    assert!(availability_before < 0.1, "Should be exhausted");

    // Replenish by resetting
    budget.reset_tokens(Some(500));
    let availability_after = budget.get_overall_availability();
    assert!(availability_after > 0.4, "Should have recovered above 40%");

    info!(
        before = availability_before,
        after = availability_after,
        "Recovery test passed"
    );

    Ok(())
}

async fn test_degradation_tiers() -> Result<()> {
    let degradation = Arc::new(DegradationManager::new(
        niodoo_real_integrated::config::DegradationConfig::default(),
    ));

    // Test Tier 1 (70-100%)
    let tier1 = degradation.determine_tier(0.85);
    assert!(
        matches!(
            tier1,
            niodoo_real_integrated::degradation_tiers::DegradationTier::Tier1
        ),
        "Should be Tier 1 at 85%"
    );

    // Test Tier 2 (50-70%)
    let tier2 = degradation.determine_tier(0.60);
    assert!(
        matches!(
            tier2,
            niodoo_real_integrated::degradation_tiers::DegradationTier::Tier2
        ),
        "Should be Tier 2 at 60%"
    );

    // Test Tier 3 (30-50%)
    let tier3 = degradation.determine_tier(0.40);
    assert!(
        matches!(
            tier3,
            niodoo_real_integrated::degradation_tiers::DegradationTier::Tier3
        ),
        "Should be Tier 3 at 40%"
    );

    // Test Tier 4 (0-30%)
    let tier4 = degradation.determine_tier(0.15);
    assert!(
        matches!(
            tier4,
            niodoo_real_integrated::degradation_tiers::DegradationTier::Tier4
        ),
        "Should be Tier 4 at 15%"
    );

    info!("Degradation tier transitions validated");
    Ok(())
}
