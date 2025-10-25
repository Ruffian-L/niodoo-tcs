use tch::{nn, Tensor, Kind, Device};

pub struct TcsLoRaPredictor {
    // fields
}

impl TcsLoRaPredictor {
    pub fn new(r: i64) -> Self {
        // init
    }
    pub fn forward(&self, input: &Tensor) -> (Tensor, Tensor) {
        // impl
    }
    pub fn train_on_tcs(&mut self, features: Vec<Vec<f64>>, labels: Vec<(f64, usize)>) {
        // impl
    }
    pub fn predict_action(&self, input: Vec<f64>) -> DqnAction {
        // impl
    }
}
