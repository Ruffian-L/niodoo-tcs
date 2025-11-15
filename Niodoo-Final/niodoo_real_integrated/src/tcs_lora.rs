use anyhow::Result;
use tracing::debug;

use crate::tcs_analysis::TopologicalSignature;
use crate::tcs_predictor::TcsPredictor;

/// Lightweight trainer that adapts the topology predictor using ridge regression.
pub struct TcsLoRaTrainer {
    predictor: TcsPredictor,
}

impl TcsLoRaTrainer {
    pub fn new(capacity: usize) -> Self {
        let mut predictor = TcsPredictor::new();
        predictor.set_capacity(capacity);
        Self { predictor }
    }

    pub fn from_predictor(predictor: TcsPredictor) -> Self {
        Self { predictor }
    }

    pub fn ingest_sample(
        &mut self,
        signature: TopologicalSignature,
        reward_delta: f64,
        performance: f64,
    ) {
        self.predictor.update(&signature, reward_delta, performance);
    }

    pub fn train_epoch(&mut self, samples: &[(TopologicalSignature, f64, f64)]) -> Result<()> {
        for (signature, reward, performance) in samples {
            self.ingest_sample(signature.clone(), *reward, *performance);
        }
        debug!("Trained TCS LoRA trainer with {} samples", samples.len());
        Ok(())
    }

    pub fn predictor(&self) -> &TcsPredictor {
        &self.predictor
    }

    pub fn predictor_mut(&mut self) -> &mut TcsPredictor {
        &mut self.predictor
    }

    pub fn predict_reward(&self, signature: &TopologicalSignature) -> f64 {
        self.predictor.predict_reward_delta(signature)
    }
}

impl Default for TcsLoRaTrainer {
    fn default() -> Self {
        Self::new(128)
    }
}
