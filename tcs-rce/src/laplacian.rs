use anyhow::Result;
use serde::{Deserialize, Serialize};
use tcs_tda::{FilteredComplex, LaplacianSpectrum, MotifMetrics, PersistentLaplacian};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaplacianSummary {
    pub spectra: Vec<LaplacianSpectrum>,
    pub spectral_flux: [f64; 3],
    pub harmonic_counts: [usize; 3],
}

pub struct LaplacianAnalyzer {
    engine: PersistentLaplacian,
    max_threshold: f32,
    resolution: usize,
}

impl LaplacianAnalyzer {
    pub fn new(max_dimension: usize, zero_tolerance: f64, max_threshold: f32, resolution: usize) -> Self {
        Self {
            engine: PersistentLaplacian::new(max_dimension, zero_tolerance),
            max_threshold,
            resolution,
        }
    }

    pub fn build_filtration(&self, distances: &[Vec<f32>]) -> Vec<FilteredComplex> {
        self.engine
            .build_filtration(distances, self.max_threshold, self.resolution)
    }

    pub fn analyze(&self, filtration: &[FilteredComplex]) -> LaplacianSummary {
        let spectra = self.engine.analyze(filtration);
        let spectral_flux = self.engine.spectral_flux(&spectra);
        let harmonic_counts = self.engine.harmonic_counts(&spectra);
        LaplacianSummary {
            spectra,
            spectral_flux,
            harmonic_counts,
        }
    }

    pub fn motifs(&self, distances: &[Vec<f32>], threshold: f32) -> MotifMetrics {
        MotifMetrics::compute(distances, threshold)
    }
}


