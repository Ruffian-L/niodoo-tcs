//! TDA Feature Extractor for EBM
//! 
//! Extracts feature vectors from Betti numbers and persistence diagrams
//! for input to the TopologicalEnergyNetwork.

use anyhow::Result;
use ndarray::{Array1, Array2};
use tracing::debug;

use crate::tcs_analysis::TopologicalSignature;
use tcs_core::PersistentFeature;

/// TDA Feature Extractor
/// 
/// Converts topological data (Betti numbers + persistence diagram) into
/// a fixed-dimensional feature vector for EBM training.
pub struct TDAFeatureExtractor {
    max_betti_dim: usize,
    persistence_resolution: usize,
}

impl TDAFeatureExtractor {
    /// Create a new TDAFeatureExtractor
    /// 
    /// Args:
    ///   - max_betti_dim: Maximum Betti dimension to include (default: 3)
    ///   - persistence_resolution: Number of histogram bins for persistence (default: 10)
    pub fn new(max_betti_dim: usize, persistence_resolution: usize) -> Self {
        Self {
            max_betti_dim,
            persistence_resolution,
        }
    }
    
    /// Default constructor with standard parameters
    pub fn default() -> Self {
        Self::new(3, 10)
    }
    
    /// Extract features from Betti numbers and persistence diagram
    /// 
    /// Features include:
    ///   1. Betti numbers (β₀, β₁, β₂, ...)
    ///   2. Persistence statistics (mean, max, total, count)
    ///   3. Persistence histogram
    /// 
    /// Args:
    ///   - betti_numbers: Array of Betti numbers
    ///   - persistence_diagram: 2D array of (birth, death) pairs
    /// 
    /// Returns:
    ///   - Feature vector as Array1<f64>
    pub fn extract_features(
        &self,
        betti_numbers: &[usize],
        persistence_diagram: &Array2<f64>,
    ) -> Result<Array1<f64>> {
        let mut features = Vec::new();
        
        // 1. Betti numbers (normalized)
        for &betti in betti_numbers.iter().take(self.max_betti_dim) {
            features.push(betti as f64);
        }
        
        // Pad with zeros if fewer Betti numbers than max_betti_dim
        while features.len() < self.max_betti_dim {
            features.push(0.0);
        }
        
        // 2. Persistence statistics
        let lifetimes: Vec<f64> = persistence_diagram
            .outer_iter()
            .map(|row| {
                if row.len() >= 2 {
                    (row[1] - row[0]).max(0.0)  // lifetime = death - birth
                } else {
                    0.0
                }
            })
            .collect();
        
        if !lifetimes.is_empty() {
            let mean_lifetime = lifetimes.iter().sum::<f64>() / lifetimes.len() as f64;
            let max_lifetime = lifetimes
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            let total_persistence = lifetimes.iter().sum::<f64>();
            let feature_count = lifetimes.len() as f64;
            
            features.push(mean_lifetime);
            features.push(max_lifetime);
            features.push(total_persistence);
            features.push(feature_count);
        } else {
            features.extend_from_slice(&[0.0, 0.0, 0.0, 0.0]);
        }
        
        // 3. Persistence histogram
        let hist = self.compute_persistence_histogram(&lifetimes);
        features.extend_from_slice(&hist);
        
        debug!(
            "Extracted {} TDA features: betti={:?}, persistence_stats={}",
            features.len(),
            &features[..self.max_betti_dim],
            &features[self.max_betti_dim..self.max_betti_dim + 4]
        );
        
        Ok(Array1::from_vec(features))
    }
    
    /// Extract features from TopologicalSignature
    /// 
    /// Convenience method that extracts features directly from a TopologicalSignature.
    pub fn extract_from_signature(&self, signature: &TopologicalSignature) -> Result<Array1<f64>> {
        // Convert persistence_features to Array2
        let mut persistence_pairs = Vec::new();
        for feat in &signature.persistence_features {
            persistence_pairs.push(vec![feat.birth as f64, feat.death as f64]);
        }
        
        let persistence_diagram = if persistence_pairs.is_empty() {
            Array2::zeros((0, 2))
        } else {
            Array2::from_shape_vec(
                (persistence_pairs.len(), 2),
                persistence_pairs.into_iter().flatten().collect(),
            )?
        };
        
        self.extract_features(&signature.betti_numbers, &persistence_diagram)
    }
    
    /// Compute persistence histogram
    /// 
    /// Bins persistence lifetimes into histogram for fixed-size feature representation.
    fn compute_persistence_histogram(&self, lifetimes: &[f64]) -> Vec<f64> {
        if lifetimes.is_empty() {
            return vec![0.0; self.persistence_resolution];
        }
        
        let max_lifetime = lifetimes
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
            .max(1.0);  // Avoid division by zero
        
        let mut histogram = vec![0.0; self.persistence_resolution];
        
        for &lifetime in lifetimes {
            if lifetime > 0.0 {
                let bin_idx = ((lifetime / max_lifetime) * (self.persistence_resolution - 1) as f64)
                    .min((self.persistence_resolution - 1) as f64) as usize;
                histogram[bin_idx] += 1.0;
            }
        }
        
        // Normalize histogram
        let total: f64 = histogram.iter().sum();
        if total > 0.0 {
            for count in &mut histogram {
                *count /= total;
            }
        }
        
        histogram
    }
    
    /// Get feature dimension
    /// 
    /// Returns the dimension of the feature vector produced by this extractor.
    pub fn feature_dim(&self) -> usize {
        self.max_betti_dim                    // Betti numbers
        + 4                                   // Persistence statistics
        + self.persistence_resolution        // Persistence histogram
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_feature_extraction() -> Result<()> {
        let extractor = TDAFeatureExtractor::default();
        
        let betti_numbers = vec![1, 2, 0];
        let persistence_diagram = Array2::from_shape_vec(
            (3, 2),
            vec![0.0, 1.0, 0.5, 2.0, 1.0, 1.5],
        )?;
        
        let features = extractor.extract_features(&betti_numbers, &persistence_diagram)?;
        
        assert_eq!(features.len(), extractor.feature_dim());
        assert!(features[0] > 0.0);  // First Betti number
        
        Ok(())
    }
}

