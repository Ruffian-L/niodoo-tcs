use crate::sheaf_nn::{SheafDiffusionLayer, SheafMode};
use crate::consciousness_metrics::approximate_phi_from_betti;

/// Telemetry state for analysis
#[derive(Debug, Clone)]
pub struct TelemetryState {
    pub features: Vec<f32>,
    pub betti_numbers: [usize; 3],
}

/// Analysis result from TCS processing
#[derive(Debug, Clone)]
pub struct AnalysisResult {
    pub sheaf_output: Vec<f32>,
    pub phi_approximation: f64,
    pub processed: bool,
}

impl TelemetryState {
    pub fn new(features: Vec<f32>, betti_numbers: [usize; 3]) -> Self {
        Self { features, betti_numbers }
    }
}

pub fn analyze_state(state: &TelemetryState) -> AnalysisResult {
    let sheaf_layer = SheafDiffusionLayer::identity(state.features.len());
    let transformed = sheaf_layer.forward(&state.features, SheafMode::Listen);
    
    let phi = approximate_phi_from_betti(&state.betti_numbers)
        .unwrap_or(0.0);
    
    tracing::info!("Sheaf processed, Phi = {:.6}", phi);
    
    AnalysisResult {
        sheaf_output: transformed,
        phi_approximation: phi,
        processed: true,
    }
}

