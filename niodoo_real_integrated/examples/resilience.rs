// future
use crate::compass::CompassEngine;
use crate::config::RuntimeConfig;
use crate::learning::LearningLoop;
use std::collections::HashMap;

// Cleaned ResilienceDispatcher
pub trait ResilienceDispatcher {
    fn dispatch_recovery(
        &self,
        state: &mut dyn std::any::Any,
        config: &RuntimeConfig,
    ) -> HashMap<String, f64>;
}

impl ResilienceDispatcher for LearningLoop {
    fn dispatch_recovery(
        &self,
        state: &mut dyn std::any::Any,
        config: &RuntimeConfig,
    ) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        if low_innovation {
            // Derived check
            params.insert(
                "mutation_rate".to_string(),
                1.0 + config.emergency_multiplier,
            );
        }
        params
    }
}

impl ResilienceDispatcher for CompassEngine {
    fn dispatch_recovery(
        &self,
        state: &mut dyn std::any::Any,
        config: &RuntimeConfig,
    ) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        if low_variance {
            // Derived check
            params.insert("exploration_amp".to_string(), 1.5);
        }
        params
    }
}


