use nalgebra::{DMatrix, DVector};

#[derive(Debug, Clone)]
pub struct GeometricLayer {
    weight: DMatrix<f32>,
    bias: DVector<f32>,
}

impl GeometricLayer {
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        assert!(input_dim > 0, "input_dim must be positive");
        assert!(output_dim > 0, "output_dim must be positive");

        let mut weight = DMatrix::<f32>::zeros(input_dim, output_dim);
        let diagonal = input_dim.min(output_dim);
        for idx in 0..diagonal {
            weight[(idx, idx)] = 1.0;
        }

        let bias = DVector::<f32>::zeros(output_dim);
        Self { weight, bias }
    }

    pub fn with_parameters(weight: DMatrix<f32>, bias: DVector<f32>) -> Self {
        assert_eq!(weight.ncols(), bias.len(), "bias dimension must match weight output dimension");
        assert!(weight.nrows() > 0, "weight must have a non-zero input dimension");
        Self { weight, bias }
    }

    pub fn forward(&self, positions: &DMatrix<f32>, features: &DMatrix<f32>) -> DMatrix<f32> {
        assert_eq!(positions.nrows(), features.nrows(), "positions and features must have the same batch size");
        assert_eq!(features.ncols(), self.weight.nrows(), "feature dimension must match layer input dimension");

        let invariant_kernel = Self::pairwise_squared_distances(positions);
        let invariant_features = invariant_kernel * features;
        let mut output = invariant_features * &self.weight;

        let bias_matrix = DMatrix::<f32>::from_fn(output.nrows(), self.bias.len(), |_, col| self.bias[col]);
        output += bias_matrix;
        output
    }

    pub fn parameters(&self) -> (&DMatrix<f32>, &DVector<f32>) {
        (&self.weight, &self.bias)
    }

    fn pairwise_squared_distances(positions: &DMatrix<f32>) -> DMatrix<f32> {
        let n_points = positions.nrows();
        let gram = positions * positions.transpose();
        let mut distances = DMatrix::<f32>::zeros(n_points, n_points);

        for i in 0..n_points {
            let norm_i = gram[(i, i)];
            for j in 0..n_points {
                let norm_j = gram[(j, j)];
                let value = norm_i + norm_j - 2.0 * gram[(i, j)];
                distances[(i, j)] = value.max(0.0);
            }
        }

        distances
    }
}

#[cfg(test)]
mod tests {
    use super::GeometricLayer;
    use nalgebra::DMatrix;

    fn assert_close(lhs: &DMatrix<f32>, rhs: &DMatrix<f32>, tol: f32) {
        assert_eq!(lhs.nrows(), rhs.nrows());
        assert_eq!(lhs.ncols(), rhs.ncols());

        for (a, b) in lhs.iter().zip(rhs.iter()) {
            assert!((a - b).abs() <= tol, "mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn rotation_invariance() {
        let layer = GeometricLayer::new(2, 2);

        let positions = DMatrix::from_row_slice(
            3,
            3,
            &[
                1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                -1.0, 0.5, 0.0,
            ],
        );
        let features = DMatrix::from_row_slice(
            3,
            2,
            &[
                0.2, -0.1,
                0.0, 1.0,
                -0.4, 0.3,
            ],
        );

        let angle = 0.5_f32;
        let cosine = angle.cos();
        let sine = angle.sin();
        let rotation = DMatrix::from_row_slice(
            3,
            3,
            &[cosine, -sine, 0.0, sine, cosine, 0.0, 0.0, 0.0, 1.0],
        );
        let rotated_positions = positions.clone() * rotation;

        let expected = layer.forward(&positions, &features);
        let rotated = layer.forward(&rotated_positions, &features);

        assert_close(&expected, &rotated, 1.0e-4);
    }

    #[test]
    fn translation_invariance() {
        let layer = GeometricLayer::new(2, 2);

        let positions = DMatrix::from_row_slice(
            4,
            3,
            &[
                0.1, 0.2, 0.3,
                -0.2, 0.3, 0.4,
                0.5, -0.4, 0.6,
                -0.7, -0.8, 0.9,
            ],
        );
        let translation = DMatrix::from_row_slice(
            4,
            3,
            &[1.0, -2.0, 0.5, 1.0, -2.0, 0.5, 1.0, -2.0, 0.5, 1.0, -2.0, 0.5],
        );
        let translated_positions = &positions + &translation;

        let features = DMatrix::from_row_slice(
            4,
            2,
            &[
                0.3, -0.2,
                0.6, 0.1,
                -0.4, 0.8,
                1.0, -0.5,
            ],
        );

        let expected = layer.forward(&positions, &features);
        let translated = layer.forward(&translated_positions, &features);

        assert_close(&expected, &translated, 1.0e-4);
    }
}
