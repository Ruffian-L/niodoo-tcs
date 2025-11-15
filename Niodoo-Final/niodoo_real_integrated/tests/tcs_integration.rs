#[cfg(feature = "tcs_integration_full")]
use anyhow::Result;
#[cfg(feature = "tcs_integration_full")]
use niodoo_real_integrated::config::TcsRuntimeConfig;
use niodoo_real_integrated::tcs_analysis::baseline_topological_signature;
#[cfg(feature = "tcs_integration_full")]
use niodoo_real_integrated::tcs_analysis::TCSAnalyzer;
use niodoo_real_integrated::torus::PadGhostState;
#[cfg(feature = "tcs_integration_full")]
use tempfile::tempdir;

fn sample_pad_state() -> PadGhostState {
    PadGhostState {
        pad: [0.35, -0.12, 0.68, -0.41, 0.27, 0.53, -0.22],
        entropy: 0.84,
        mu: [0.08, 0.05, -0.11, 0.19, -0.07, 0.31, 0.02],
        sigma: [0.42, 0.47, 0.33, 0.26, 0.29, 0.44, 0.21],
    }
}

#[cfg(feature = "tcs_integration_full")]
fn analyzer_with_config(mut config: TcsRuntimeConfig) -> Result<(TCSAnalyzer, tempfile::TempDir)> {
    let cache_dir = tempdir()?;
    config.enable_gpu = false;
    config.cache_dir = Some(cache_dir.path().to_string_lossy().into_owned());
    let analyzer = TCSAnalyzer::new_with_runtime(&config)?;
    Ok((analyzer, cache_dir))
}

#[cfg(feature = "tcs_integration_full")]
#[test]
fn tcs_analyzer_produces_persistent_features() -> Result<()> {
    let mut config = TcsRuntimeConfig::from_env();
    config.persistence_threshold = 0.0;
    let (mut analyzer, _cache_guard) = analyzer_with_config(config)?;

    let signature = analyzer.analyze_state(&sample_pad_state())?;

    assert!(
        !signature.persistence_features.is_empty(),
        "expected persistence features to be populated"
    );
    assert!(
        signature.euler_characteristic.is_finite(),
        "euler characteristic should be finite"
    );
    Ok(())
}

#[cfg(feature = "tcs_integration_full")]
#[test]
fn robust_mode_errors_when_persistence_filtered() -> Result<()> {
    let mut config = TcsRuntimeConfig::from_env();
    config.persistence_threshold = 10.0;
    config.robust_mode = true;
    let (mut analyzer, _cache_guard) = analyzer_with_config(config)?;

    let result = analyzer.analyze_state(&sample_pad_state());
    assert!(
        result.is_err(),
        "robust mode should surface errors when persistence is below threshold"
    );
    Ok(())
}

#[cfg(feature = "tcs_integration_full")]
#[test]
fn non_robust_mode_allows_persistence_fallback() -> Result<()> {
    let mut config = TcsRuntimeConfig::from_env();
    config.persistence_threshold = 10.0;
    config.robust_mode = false;
    let (mut analyzer, _cache_guard) = analyzer_with_config(config)?;

    let signature = analyzer.analyze_state(&sample_pad_state())?;
    assert!(
        !signature.persistence_features.is_empty(),
        "non-robust mode should retain original persistence features"
    );
    Ok(())
}

#[test]
fn baseline_topology_signature_provides_deterministic_metrics() {
    let pad_state = sample_pad_state();
    let embedding: Vec<f32> = (0..64).map(|value| value as f32 * 0.01).collect();

    let signature = baseline_topological_signature(&pad_state, &embedding);

    assert_eq!(signature.betti_numbers.len(), 3);
    assert!(
        signature.total_persistence.is_finite(),
        "baseline signature should compute finite total persistence"
    );
}
