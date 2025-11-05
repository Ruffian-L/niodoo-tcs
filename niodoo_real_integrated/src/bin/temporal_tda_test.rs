//! Temporal TDA Failure Chain Detection Test
//!
//! Tests temporal TDA detection of failure patterns using synthetic failure chains.

use anyhow::Result;
use chrono::Utc;
use niodoo_real_integrated::compass::CompassQuadrant;
use niodoo_real_integrated::tcs_analysis::TopologicalSignature;
use niodoo_real_integrated::temporal_tda::{
    FailurePatternType, TemporalTDADetector, TopologicalSnapshot,
};
use tracing::info;
use uuid::Uuid;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    info!("🧪 Starting Temporal TDA Failure Chain Detection Test");

    // Test 1: Detect rate limit pattern
    info!("Test 1: Rate limit barcode pattern");
    test_rate_limit_pattern().await?;

    // Test 2: Detect overload pattern
    info!("Test 2: Overload barcode pattern");
    test_overload_pattern().await?;

    // Test 3: Detect failure loop
    info!("Test 3: Failure loop detection");
    test_failure_loop().await?;

    // Test 4: Danger signature detection
    info!("Test 4: Danger signature detection");
    test_danger_signature().await?;

    info!("✅ All temporal TDA tests passed!");
    Ok(())
}

fn create_test_topology(beta_1: usize, beta_2: usize) -> TopologicalSignature {
    TopologicalSignature {
        id: Uuid::new_v4(),
        timestamp: Utc::now(),
        persistence_features: vec![],
        betti_numbers: [1, beta_1, beta_2],
        knot_complexity: 0.5,
        knot_polynomial: "".to_string(),
        tqft_dimension: 2,
        cobordism_type: None,
        persistence_entropy: 1.5,
        spectral_gap: 0.3,
        euler_characteristic: 1.0 - beta_1 as f64 + beta_2 as f64,
        total_persistence: 0.0,
        max_persistence: 0.0,
        mean_persistence: 0.0,
        laplacian_spectral_radius: 0.0,
        computation_time_ms: 10.0,
    }
}

fn create_test_snapshot(
    beta_1: usize,
    beta_2: usize,
    quadrant: CompassQuadrant,
    arousal: f32,
    entropy: f64,
) -> TopologicalSnapshot {
    let topology = create_test_topology(beta_1, beta_2);
    TopologicalSnapshot::new(
        topology, quadrant, 1000, // token_count
        arousal, entropy,
    )
}

async fn test_rate_limit_pattern() -> Result<()> {
    let mut detector = TemporalTDADetector::new(20, 0.5);

    // Create rate limit pattern: β₁=3-4, β₂≈0
    for i in 0..5 {
        let snapshot = create_test_snapshot(
            3 + (i % 2), // Alternate between 3 and 4
            0,
            CompassQuadrant::Panic,
            0.3 + (i as f32 * 0.1),
            2.0 + (i as f64 * 0.1),
        );
        detector.add_snapshot(snapshot);
    }

    // Check pattern detection
    let pattern = FailurePatternType::from_snapshot(&detector.history[0]);
    assert_eq!(
        pattern,
        FailurePatternType::RateLimitBarcode,
        "Should detect rate limit pattern"
    );

    info!("Rate limit pattern detection validated");
    Ok(())
}

async fn test_overload_pattern() -> Result<()> {
    let mut detector = TemporalTDADetector::new(20, 0.5);

    // Create overload pattern: β₁>5, β₂>2
    for i in 0..5 {
        let snapshot = create_test_snapshot(
            6 + (i % 2), // High β₁
            3 + (i % 2), // High β₂
            CompassQuadrant::Panic,
            0.5,
            2.5,
        );
        detector.add_snapshot(snapshot);
    }

    // Check pattern detection
    let pattern = FailurePatternType::from_snapshot(&detector.history[0]);
    assert_eq!(
        pattern,
        FailurePatternType::OverloadBarcode,
        "Should detect overload pattern"
    );

    info!("Overload pattern detection validated");
    Ok(())
}

async fn test_failure_loop() -> Result<()> {
    let mut detector = TemporalTDADetector::new(20, 0.5);

    // Create repeating pattern
    let pattern: Vec<TopologicalSnapshot> = (0..3)
        .map(|i| create_test_snapshot(4, 1, CompassQuadrant::Panic, 0.4, 2.2))
        .collect();

    // Add pattern twice to create a loop
    for snapshot in pattern.iter() {
        detector.add_snapshot(snapshot.clone());
    }
    for snapshot in pattern.iter() {
        detector.add_snapshot(snapshot.clone());
    }

    // Detect loop
    let failure_chain = detector.detect_failure_loop();
    assert!(failure_chain.is_some(), "Should detect failure loop");

    if let Some(chain) = failure_chain {
        assert!(chain.is_loop, "Should be marked as loop");
        assert!(chain.severity > 0.0, "Should have severity score");
        info!(
            pattern_type = ?chain.pattern_type,
            severity = chain.severity,
            "Failure loop detected"
        );
    }

    info!("Failure loop detection validated");
    Ok(())
}

async fn test_danger_signature() -> Result<()> {
    let mut detector = TemporalTDADetector::new(20, 0.5);

    // Create dangerous pattern: rising β₁, high arousal, entropy divergence
    for i in 0..5 {
        let snapshot = create_test_snapshot(
            3 + i, // Rising β₁
            0,
            CompassQuadrant::Panic,
            0.3 + (i as f32 * 0.1), // Increasing arousal
            1.5 + (i as f64 * 0.3), // Diverging entropy
        );
        detector.add_snapshot(snapshot);
    }

    // Analyze for danger signature
    let danger_signature = detector.analyze_sequence();
    assert!(danger_signature.is_some(), "Should detect danger signature");

    if let Some(signature) = danger_signature {
        assert!(signature.is_dangerous(), "Should be marked as dangerous");
        info!(
            beta_1_trend = ?signature.beta_1_trend,
            arousal = signature.arousal,
            token_velocity = signature.token_velocity,
            entropy_divergence = signature.entropy_divergence,
            "Danger signature detected"
        );
    }

    info!("Danger signature detection validated");
    Ok(())
}
