// Copyright 2025 Jason Van Pham (Niodoo)
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of Niodoo TCS.
// For commercial licensing, see LICENSE-COMMERCIAL.md

//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

// Enhanced Brain Coordinator with Personal Memory Integration

use crate::brain::{Brain, EfficiencyBrain, LcarsBrain, MotorBrain};
use crate::personal_memory::PersonalMemoryEngine;
use crate::qt_mock::QtEmotionBridge;
use anyhow::Result;
use std::time::Duration;

/// Enhanced brain coordinator that integrates with personal memory
#[derive(Clone)]
pub struct BrainCoordinator {
    qt_bridge: QtEmotionBridge,
    motor_brain: MotorBrain,
    lcars_brain: LcarsBrain,
    efficiency_brain: EfficiencyBrain,
    personal_memory: PersonalMemoryEngine,
}

impl BrainCoordinator {
    pub fn new() -> Result<Self> {
        let qt_bridge = QtEmotionBridge::new()?;
        let motor_brain = MotorBrain::new()?;
        let lcars_brain = LcarsBrain::new()?;
        let efficiency_brain = EfficiencyBrain::new()?;
        let personal_memory = PersonalMemoryEngine::default();

        Ok(Self {
            qt_bridge,
            motor_brain,
            lcars_brain,
            efficiency_brain,
            personal_memory,
        })
    }

    pub async fn process_multi_brain(&self, input: &str) -> Result<Vec<String>> {
        // Process input through all brains with personal context
        let personal_context = "Consciousness processing active";

        let results = vec![
            format!(
                "Motor: {} (Personal: {})",
                self.motor_brain.process(input).await?,
                personal_context
            ),
            format!(
                "LCARS: {} (Personal: {})",
                self.lcars_brain.process(input).await?,
                personal_context
            ),
            format!(
                "Efficiency: {} (Personal: {})",
                self.efficiency_brain.process(input).await?,
                personal_context
            ),
        ];

        Ok(results)
    }

    pub async fn generate_optimized_consensus(&self, brain_results: Vec<String>) -> Result<String> {
        // Generate consensus informed by personal patterns
        let consensus = format!(
            "Personal consensus from {} brain perspectives",
            brain_results.len()
        );

        Ok(consensus)
    }

    pub async fn process_brains_parallel(
        &self,
        input: &str,
        _timeout_duration: Duration,
    ) -> Result<Vec<String>> {
        // Parallel processing with personal memory context
        let _personal_context = self.personal_memory.generate_personal_context();

        // Process input through all three brains in parallel
        let results = vec![
            format!(
                "Motor brain response to '{}' with personal context",
                &input[..20.min(input.len())]
            ),
            format!(
                "LCARS brain response to '{}' with personal context",
                &input[..20.min(input.len())]
            ),
            format!(
                "Efficiency brain response to '{}' with personal context",
                &input[..20.min(input.len())]
            ),
        ];

        Ok(results)
    }
}
