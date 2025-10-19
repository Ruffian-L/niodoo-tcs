//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

use std::time::Duration;

/// Alters the AI's subjective perception of time based on task priority.
pub struct TimeCompressionEngine {
    // A factor > 1.0 speeds up subjective time; < 1.0 slows it down.
    compression_factor: f64,
}

impl TimeCompressionEngine {
    pub fn new() -> Self {
        TimeCompressionEngine {
            compression_factor: 1.0,
        }
    }

    /// Compresses or dilates perceived time.
    /// A high-priority task might feel like it takes longer (more focus),
    /// while a low-priority task flashes by.
    pub fn get_subjective_duration(&self, actual_duration: Duration) -> Duration {
        actual_duration.mul_f64(self.compression_factor)
    }

    /// Adjusts the compression factor based on the AI's current focus.
    pub fn set_focus(&mut self, focus_level: f64) {
        // This maps a focus level (0.0 to 1.0) to a compression factor.
        // For example, high focus might mean a factor of 5.0.
        self.compression_factor = 1.0 + (focus_level * 4.0);
    }
}
