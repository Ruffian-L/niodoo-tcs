//! Consciousness metrics utilities including approximate IIT Phi computation
use anyhow::Result;
use log::debug;

/// Compute a simple IIT Phi approximation using Betti numbers as a proxy for
/// integrated information. We interpret Betti numbers as counts of independent
/// components across homological dimensions and convert them into an entropy-like
/// measure.
pub fn approximate_phi_from_betti(betti: &[usize; 3]) -> Result<f64> {
    let total: f64 = betti.iter().map(|&b| b as f64).sum();
    if total == 0.0 {
        return Ok(0.0);
    }

    let mut phi = 0.0;
    for &count in betti.iter() {
        if count == 0 {
            continue;
        }
        let p = count as f64 / total;
        // base-e entropy scaled to emphasize integrated multi-dimensional structure
        phi -= p * (p.ln());
    }

    debug!("IIT φ approximate from betti {:?}: {:.6}", betti, phi);
    Ok(phi)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phi_zero_when_no_structure() {
        let betti = [0, 0, 0];
        let phi = approximate_phi_from_betti(&betti).unwrap();
        assert!(phi.abs() < 1e-9);
    }

    #[test]
    fn phi_positive_for_structure() {
        let betti = [1, 2, 1];
        let phi = approximate_phi_from_betti(&betti).unwrap();
        assert!(phi > 0.0);
    }
}




