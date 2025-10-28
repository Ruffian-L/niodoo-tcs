use anyhow::Result;

/// Compute ROUGE-L F1 using existing util function to ensure consistency
pub fn rouge_l_f1(candidate: &str, reference: &str) -> f64 {
    crate::util::rouge_l(candidate, reference)
}

/// Pearson correlation coefficient between xs and ys
pub fn pearson(xs: &[f64], ys: &[f64]) -> Result<f64> {
    anyhow::ensure!(xs.len() == ys.len() && !xs.is_empty(), "length mismatch");
    let n = xs.len() as f64;
    let mean_x = xs.iter().sum::<f64>() / n;
    let mean_y = ys.iter().sum::<f64>() / n;
    let mut num = 0.0;
    let mut den_x = 0.0;
    let mut den_y = 0.0;
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        let dx = x - mean_x;
        let dy = y - mean_y;
        num += dx * dy;
        den_x += dx * dx;
        den_y += dy * dy;
    }
    let denom = (den_x.sqrt() * den_y.sqrt()).max(1e-12);
    Ok((num / denom).clamp(-1.0, 1.0))
}

/// Spearman rank correlation coefficient (rho)
pub fn spearman(xs: &[f64], ys: &[f64]) -> Result<f64> {
    anyhow::ensure!(xs.len() == ys.len() && !xs.is_empty(), "length mismatch");
    let rx = rank(xs);
    let ry = rank(ys);
    pearson(&rx, &ry)
}

fn rank(values: &[f64]) -> Vec<f64> {
    // Average ranks for ties
    let mut indexed: Vec<(usize, f64)> = values.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut ranks = vec![0.0; values.len()];
    let mut i = 0;
    while i < indexed.len() {
        let j = (i + 1..indexed.len())
            .find(|&k| indexed[k].1 > indexed[i].1)
            .unwrap_or(indexed.len());
        let avg_rank = (i + j - 1) as f64 / 2.0 + 1.0; // 1-based average rank
        for k in i..j {
            ranks[indexed[k].0] = avg_rank;
        }
        i = j;
    }
    ranks
}

/// Wrapper struct to expose topology metrics uniformly
#[derive(Debug, Clone, Copy)]
pub struct TopologyMetrics {
    pub betti_1: f64,
    pub spectral_gap: f64,
    pub persistence_entropy: f64,
}

impl From<&crate::tcs_analysis::TopologicalSignature> for TopologyMetrics {
    fn from(sig: &crate::tcs_analysis::TopologicalSignature) -> Self {
        Self {
            betti_1: sig.betti_numbers[1] as f64,
            spectral_gap: sig.spectral_gap,
            persistence_entropy: sig.persistence_entropy,
        }
    }
}


