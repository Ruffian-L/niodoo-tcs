#[derive(Debug, Clone, Default)]
pub struct StageTimings {
    pub embedding_ms: f64,
    pub torus_ms: f64,
    pub tcs_ms: f64,
    pub compass_ms: f64,
    pub erag_ms: f64,
    pub tokenizer_ms: f64,
    pub generation_ms: f64,
    pub learning_ms: f64,
    pub threat_cycle_ms: f64,
}

impl StageTimings {
    pub fn total_latency_ms(&self) -> f64 {
        self.embedding_ms
            + self.torus_ms
            + self.tcs_ms
            + self.compass_ms
            + self.erag_ms
            + self.tokenizer_ms
            + self.generation_ms
            + self.learning_ms
            + self.threat_cycle_ms
    }
}

