use nalgebra::Vector3;

/// Compute a discrete approximation of Gauss's linking integral between two closed trajectories
/// in 3D (e.g., PAD-space). Returns a float that is typically close to an integer invariant.
pub fn compute_linking_number(traj_a: &[Vector3<f64>], traj_b: &[Vector3<f64>]) -> f64 {
    if traj_a.len() < 2 || traj_b.len() < 2 {
        return 0.0;
    }

    let mut linking_sum = 0.0f64;

    for i in 0..traj_a.len() {
        let p1 = traj_a[i];
        let p2 = traj_a[(i + 1) % traj_a.len()];
        let vec_a = p2 - p1;

        for j in 0..traj_b.len() {
            let q1 = traj_b[j];
            let q2 = traj_b[(j + 1) % traj_b.len()];
            let vec_b = q2 - q1;
            let vec_r = p1 - q1;

            let distance_sq = vec_r.norm_squared();
            if distance_sq < 1e-9 {
                continue;
            }
            let determinant = vec_a.cross(&vec_b).dot(&vec_r);
            linking_sum += determinant / (4.0 * std::f64::consts::PI * distance_sq.powf(1.5));
        }
    }

    // Round to nearest integer for invariance (approximation)
    linking_sum.round()
}
