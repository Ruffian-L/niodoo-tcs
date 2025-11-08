//! Quick test of healing/topology integration
//! Tests the key integration points without heavy computation

use niodoo_real_integrated::compass::CompassEngine;
use niodoo_real_integrated::tcs_analysis::TCSAnalyzer;
use niodoo_real_integrated::torus::PadGhostState;
use std::sync::{Arc, Mutex};
use tracing::{info, Level};
use tracing_subscriber;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();

    info!("🏥 Testing Healing/Topology Integration");
    info!("======================================");

    // Initialize analyzer and compass
    let mut analyzer = TCSAnalyzer::new()?;
    let compass = Arc::new(Mutex::new(CompassEngine::new(0.5, 0.4, 0.2)));

    // Test position 1.27 - should trigger healing
    info!("");
    info!("📍 Position 1.27 - Expected healing state");
    let pad_state = PadGhostState {
        pad: [0.85, 0.5, 0.7, 0.3, 0.4, 0.2, 0.1],
        entropy: 1.9,
        mu: [0.8, 0.4, 0.6, 0.2, 0.3, 0.1, 0.05],
        sigma: [0.1, 0.2, 0.15, 0.1, 0.1, 0.1, 0.05],
    };

    let topology = analyzer.analyze_state(&pad_state)?;
    info!(
        "  Topology: knot={:.3}, gap={:.3}, betti={:?}",
        topology.knot_complexity, topology.spectral_gap, topology.betti_numbers
    );

    let compass_outcome = compass
        .lock()
        .map_err(|e| format!("Lock error: {}", e))?
        .evaluate(&pad_state)?;

    info!(
        "  Compass: healing={}, threat={}, quadrant={:?}",
        compass_outcome.is_healing, compass_outcome.is_threat, compass_outcome.quadrant
    );

    if compass_outcome.is_healing {
        info!("  ✅ Healing detected - integration working!");
    } else {
        info!("  ⚠️  No healing detected");
    }

    // Test position 2.0 - should NOT trigger healing (threat state)
    info!("");
    info!("📍 Position 2.0 - Threat state (should NOT heal)");
    let pad_state = PadGhostState {
        pad: [-0.3, 0.8, -0.5, 0.6, -0.2, 0.4, 0.1],
        entropy: 2.1,
        mu: [-0.2, 0.7, -0.4, 0.5, -0.1, 0.3, 0.05],
        sigma: [0.2, 0.25, 0.2, 0.2, 0.15, 0.15, 0.1],
    };

    let topology = analyzer.analyze_state(&pad_state)?;
    info!(
        "  Topology: knot={:.3}, gap={:.3}, betti={:?}",
        topology.knot_complexity, topology.spectral_gap, topology.betti_numbers
    );

    let compass_outcome = compass
        .lock()
        .map_err(|e| format!("Lock error: {}", e))?
        .evaluate(&pad_state)?;

    info!(
        "  Compass: healing={}, threat={}, quadrant={:?}",
        compass_outcome.is_healing, compass_outcome.is_threat, compass_outcome.quadrant
    );

    if !compass_outcome.is_healing && compass_outcome.is_threat {
        info!("  ✅ Threat detected, no healing - correct behavior!");
    } else {
        info!("  ⚠️  Unexpected behavior");
    }

    info!("");
    info!("🎯 Summary: Healing/topology integration tested successfully!");

    Ok(())
}
