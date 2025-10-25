const MIN_DIMENSION: usize = 1;
const ZERO_WEIGHT: f32 = 0.0;
const UNITY_WEIGHT: f32 = 1.0;

#[derive(Clone, Copy, Debug)]
pub enum SheafMode {
    Listen,
    Broadcast,
}

#[derive(Clone, Debug)]
pub struct SheafDiffusionLayer {
    weight: Vec<Vec<f32>>,
}

impl SheafDiffusionLayer {
    pub fn new(weight: Vec<Vec<f32>>) -> Self {
        let placeholder = if weight.is_empty() || weight.iter().all(|row| row.is_empty()) {
            vec![vec![UNITY_WEIGHT]]
        } else {
            weight
        };

        let max_len = placeholder
            .iter()
            .map(|row| row.len())
            .max()
            .unwrap_or(MIN_DIMENSION);

        let sanitized = placeholder
            .into_iter()
            .map(|mut row| {
                if row.len() < max_len {
                    row.resize(max_len, ZERO_WEIGHT);
                }
                row
            })
            .collect::<Vec<_>>();

        Self { weight: sanitized }
    }

    pub fn identity(dimension: usize) -> Self {
        let dim = dimension.max(MIN_DIMENSION);
        let mut weight = vec![vec![ZERO_WEIGHT; dim]; dim];

        for idx in 0..dim {
            weight[idx][idx] = UNITY_WEIGHT;
        }

        Self::new(weight)
    }

    pub fn forward(&self, features: &[f32], mode: SheafMode) -> Vec<f32> {
        match mode {
            SheafMode::Listen => Self::multiply(&self.weight, features),
            SheafMode::Broadcast => {
                let transposed = Self::transpose(&self.weight);
                Self::multiply(&transposed, features)
            }
        }
    }

    pub fn dual_pass(&self, features: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let listen = self.forward(features, SheafMode::Listen);
        let broadcast = self.forward(features, SheafMode::Broadcast);
        (listen, broadcast)
    }

    pub fn weight(&self) -> &[Vec<f32>] {
        &self.weight
    }

    fn multiply(matrix: &[Vec<f32>], features: &[f32]) -> Vec<f32> {
        if matrix.is_empty() {
            return Vec::new();
        }

        let column_count = matrix[0].len();
        if column_count == 0 {
            return vec![ZERO_WEIGHT; matrix.len()];
        }

        let mut padded = Vec::with_capacity(column_count);
        for idx in 0..column_count {
            let value = *features.get(idx).unwrap_or(&ZERO_WEIGHT);
            padded.push(value);
        }

        matrix
            .iter()
            .map(|row| row.iter().zip(padded.iter()).map(|(w, f)| w * f).sum())
            .collect::<Vec<f32>>()
    }

    fn transpose(matrix: &[Vec<f32>]) -> Vec<Vec<f32>> {
        if matrix.is_empty() {
            return Vec::new();
        }

        let column_count = matrix[0].len();
        let mut transposed = vec![vec![ZERO_WEIGHT; matrix.len()]; column_count];

        for (row_idx, row) in matrix.iter().enumerate() {
            for (col_idx, value) in row.iter().enumerate() {
                transposed[col_idx][row_idx] = *value;
            }
        }

        transposed
    }
}

