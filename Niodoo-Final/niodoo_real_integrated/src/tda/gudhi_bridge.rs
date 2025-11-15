use crate::tda::PDPoint;

/// Compute a q=2 Wasserstein distance between two persistence diagrams using
/// a Hungarian assignment on squared Euclidean costs as an approximation.
/// Note: This implementation does not include diagonal matching; it is adequate
/// for diagrams of comparable cardinality. When Gudhi bindings are available,
/// prefer those for full generality and stability.
pub fn wasserstein_w2(current: &[PDPoint], baseline: &[PDPoint]) -> f64 {
    if current.is_empty() && baseline.is_empty() {
        return 0.0;
    }
    let n = current.len().min(baseline.len());
    if n == 0 {
        // If only one is empty, treat full mass difference as distance
        let s: f64 = current
            .iter()
            .map(|p| (p.death - p.birth).abs())
            .sum::<f64>()
            + baseline
                .iter()
                .map(|p| (p.death - p.birth).abs())
                .sum::<f64>();
        return s.sqrt();
    }

    // Build cost matrix of squared L2 distances
    let mut costs = vec![vec![0f64; n]; n];
    for (i, a) in current.iter().take(n).enumerate() {
        for (j, b) in baseline.iter().take(n).enumerate() {
            let dx = a.birth - b.birth;
            let dy = a.death - b.death;
            costs[i][j] = dx * dx + dy * dy;
        }
    }

    let assignment = hungarian_min_cost(&costs);
    let mut total_cost = 0.0f64;
    for (i, maybe_j) in assignment.into_iter().enumerate() {
        if let Some(j_idx) = maybe_j {
            total_cost += costs[i][j_idx];
        }
    }

    // q=2 Wasserstein distance is sqrt of minimal sum of squared distances
    total_cost.sqrt()
}

fn hungarian_min_cost(costs: &[Vec<f64>]) -> Vec<Option<usize>> {
    let n = costs.len();
    let mut u = vec![0.0f64; n + 1];
    let mut v = vec![0.0f64; n + 1];
    let mut p = vec![0usize; n + 1];
    let mut way = vec![0usize; n + 1];

    for i in 1..=n {
        p[0] = i;
        let mut minv = vec![f64::INFINITY; n + 1];
        let mut used = vec![false; n + 1];
        let mut j0 = 0usize;

        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = f64::INFINITY;
            let mut j1 = 0usize;

            for j in 1..=n {
                if used[j] {
                    continue;
                }

                let cur = costs[i0 - 1][j - 1] - u[i0] - v[j];
                if cur < minv[j] {
                    minv[j] = cur;
                    way[j] = j0;
                }
                if minv[j] < delta {
                    delta = minv[j];
                    j1 = j;
                }
            }

            for j in 0..=n {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }

            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }

        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }

    let mut assignment = vec![None; n];
    for j in 1..=n {
        if p[j] != 0 {
            assignment[p[j] - 1] = Some(j - 1);
        }
    }
    assignment
}
