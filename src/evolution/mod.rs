pub mod dna_transcription;

#[derive(Debug, Clone)]
pub struct AIDNA {
    pub genes: Vec<String>,
    pub fitness: f32,
}

impl AIDNA {
    pub fn mock() -> Self {
        AIDNA {
            genes: vec!["default_gene".to_string()],
            fitness: 0.5,
        }
    }

    pub fn express_candidates(&self, count: usize) -> Vec<ConsciousnessBlueprint> {
        (0..count)
            .map(|_| ConsciousnessBlueprint {
                id: "mock".to_string(),
                fitness: 0.0,
            })
            .collect()
    }

    pub fn crossover(&mut self, fittest: &ConsciousnessBlueprint) {
        self.fitness = fittest.fitness;
    }

    pub fn mutate(&mut self, rate: f32) {
        self.fitness += rate * 0.1;
    }
}

#[derive(Debug, Clone, Default)]
pub struct ConsciousnessBlueprint {
    pub id: String,
    pub fitness: f32,
}
