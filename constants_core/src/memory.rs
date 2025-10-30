// Copyright 2025 Jason Van Pham (Niodoo)
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of Niodoo TCS.
// For commercial licensing, see LICENSE-COMMERCIAL.md

// Memory Management Constants
// All derived from system constraints and mathematical principles
use crate::mathematical::phi;
use std::env;
use std::f64::consts::{E, PI};
use std::sync::OnceLock;
use sysinfo::System;

// Default fallback values if system probing or environment configuration fails
// These are conventional fallbacks when system probing fails and can be overridden via env vars
const DEFAULT_MAX_MEMORY_GB: usize = 4;
const DEFAULT_MEMORY_SAFETY_MARGIN: f64 = 0.2;

/// Get maximum memory in GB
/// Reads from NIODOO_MAX_MEMORY_GB environment variable if available,
/// otherwise derives from system probing with safety margin applied
pub fn max_memory_gb() -> usize {
    static MAX_MEMORY: OnceLock<usize> = OnceLock::new();

    *MAX_MEMORY.get_or_init(|| {
        env::var("NIODOO_MAX_MEMORY_GB")
            .ok()
            .and_then(|val| val.parse::<usize>().ok())
            .filter(|&val| val > 0)
            .unwrap_or_else(|| {
                // Derive from system probing, convert bytes to GiB
                let usable_bytes = usable_memory_bytes();
                if usable_bytes > 0 {
                    (usable_bytes / (1024 * 1024 * 1024)).max(1) // At least 1 GiB
                } else {
                    DEFAULT_MAX_MEMORY_GB // Fallback if probing fails
                }
            })
    })
}

/// Get memory safety margin (percentage to reserve)
/// Reads from NIODOO_MEMORY_SAFETY_MARGIN environment variable if available
pub fn safety_margin() -> f64 {
    static MARGIN: OnceLock<f64> = OnceLock::new();

    *MARGIN.get_or_init(|| {
        env::var("NIODOO_MEMORY_SAFETY_MARGIN")
            .ok()
            .and_then(|val| val.parse::<f64>().ok())
            .filter(|&val| (0.0..1.0).contains(&val))
            .unwrap_or(DEFAULT_MEMORY_SAFETY_MARGIN)
    })
}

/// Calculate actual usable memory with safety margin
/// Uses system probe to get actual available memory, falling back to max_memory_gb() if probe fails
pub fn usable_memory_bytes() -> usize {
    static MEMORY_BYTES: OnceLock<usize> = OnceLock::new();

    *MEMORY_BYTES.get_or_init(|| {
        // Try to probe actual system memory
        let mut sys = System::new();
        sys.refresh_memory();

        // Try available_memory first (more accurate for usable memory), fall back to total_memory
        // Note: sysinfo >=0.30 returns bytes directly
        let memory_bytes = {
            let available = sys.available_memory();
            if available > 0 {
                available
            } else {
                let total = sys.total_memory();
                if total > 0 {
                    total
                } else {
                    0 // Will trigger fallback below
                }
            }
        };

        if memory_bytes > 0 {
            // Apply safety margin directly to bytes
            (memory_bytes as f64 * (1.0 - safety_margin())) as usize
        } else {
            // Fallback to our constant if probe fails
            // Compute in u64 to prevent overflow on 32-bit targets
            let max_bytes_u64 = (max_memory_gb() as u64) * 1024_u64 * 1024_u64 * 1024_u64;
            let available_u64 = (max_bytes_u64 as f64 * (1.0 - safety_margin())) as u64;
            // Clamp to usize range for safe conversion
            available_u64.min(usize::MAX as u64) as usize
        }
    })
}

// Memory Consolidation Constants - All mathematically derived
// PHI imported from crate::mathematical::phi()
// System uses PI and E from std::f64::consts

/// Configurable memory constants with mathematically derived defaults
pub struct MemoryConstants {
    consolidation_threshold_base: OnceLock<u8>,
    consolidation_threshold_max: OnceLock<u8>,
    merge_threshold_base: OnceLock<u8>,
    merge_threshold_max: OnceLock<u8>,
    consolidation_scaling_divisor: OnceLock<f32>,
    compression_scaling_divisor: OnceLock<f32>,
    min_group_size_base: OnceLock<usize>,
    group_size_scaling_divisor: OnceLock<f32>,
    similarity_threshold_base: OnceLock<f32>,
    similarity_threshold_max_adjustment: OnceLock<f32>,
    similarity_scaling_divisor: OnceLock<f32>,
}

impl MemoryConstants {
    /// Base consolidation threshold derived from golden ratio (φ ≈ 1.618, rounded to 2)
    pub fn consolidation_threshold_base(&self) -> u8 {
        *self.consolidation_threshold_base.get_or_init(|| {
            env::var("NIODOO_CONSOLIDATION_THRESHOLD_BASE")
                .ok()
                .and_then(|val| val.parse::<u8>().ok())
                .filter(|&val| val > 0)
                .unwrap_or(2) // Golden ratio rounded, represents natural grouping
        })
    }

    /// Maximum consolidation threshold - derived from Fibonacci sequence limit F(6) + F(5)
    pub fn consolidation_threshold_max(&self) -> u8 {
        *self.consolidation_threshold_max.get_or_init(|| {
            env::var("NIODOO_CONSOLIDATION_THRESHOLD_MAX")
                .ok()
                .and_then(|val| val.parse::<u8>().ok())
                .filter(|&val| val > self.consolidation_threshold_base())
                .unwrap_or(8 + 5) // F(6) + F(5), natural scaling limit
        })
    }

    /// Merge threshold base - derived from e (Euler's number) rounded
    pub fn merge_threshold_base(&self) -> u8 {
        *self.merge_threshold_base.get_or_init(|| {
            env::var("NIODOO_MERGE_THRESHOLD_BASE")
                .ok()
                .and_then(|val| val.parse::<u8>().ok())
                .filter(|&val| val > 0)
                .unwrap_or(E.round() as u8) // e rounded (≈2.718 → 3), natural log base, info entropy
        })
    }

    /// Maximum merge threshold - same as consolidation max for consistency
    pub fn merge_threshold_max(&self) -> u8 {
        *self.merge_threshold_max.get_or_init(|| {
            env::var("NIODOO_MERGE_THRESHOLD_MAX")
                .ok()
                .and_then(|val| val.parse::<u8>().ok())
                .filter(|&val| val > self.merge_threshold_base())
                .unwrap_or(self.consolidation_threshold_max()) // Same as consolidation for consistency
        })
    }

    /// Memory consolidation scaling divisor - derived from PI * 100
    pub fn consolidation_scaling_divisor(&self) -> f32 {
        *self.consolidation_scaling_divisor.get_or_init(|| {
            env::var("NIODOO_CONSOLIDATION_SCALING_DIVISOR")
                .ok()
                .and_then(|val| val.parse::<f32>().ok())
                .filter(|&val| val > 0.0)
                .unwrap_or((PI * 100.0) as f32) // PI * 100
        })
    }

    /// Memory compression scaling divisor - derived from e^2 * 100
    pub fn compression_scaling_divisor(&self) -> f32 {
        *self.compression_scaling_divisor.get_or_init(|| {
            env::var("NIODOO_COMPRESSION_SCALING_DIVISOR")
                .ok()
                .and_then(|val| val.parse::<f32>().ok())
                .filter(|&val| val > 0.0)
                .unwrap_or((E * E * 100.0) as f32) // e^2 * 100
        })
    }

    /// Minimum group size base - derived from smallest prime number > 1
    pub fn min_group_size_base(&self) -> usize {
        *self.min_group_size_base.get_or_init(|| {
            env::var("NIODOO_MIN_GROUP_SIZE_BASE")
                .ok()
                .and_then(|val| val.parse::<usize>().ok())
                .filter(|&val| val > 0)
                .unwrap_or(3) // Smallest odd prime (3)
        })
    }

    /// Group size scaling divisor - derived from golden ratio squared * 10
    pub fn group_size_scaling_divisor(&self) -> f32 {
        *self.group_size_scaling_divisor.get_or_init(|| {
            env::var("NIODOO_GROUP_SIZE_SCALING_DIVISOR")
                .ok()
                .and_then(|val| val.parse::<f32>().ok())
                .filter(|&val| val > 0.0)
                .unwrap_or(phi() * phi() * 10.0) // PHI^2 * 10
        })
    }

    /// Similarity threshold base - derived from 1/PHI (complement of golden ratio)
    pub fn similarity_threshold_base(&self) -> f32 {
        *self.similarity_threshold_base.get_or_init(|| {
            env::var("NIODOO_SIMILARITY_THRESHOLD_BASE")
                .ok()
                .and_then(|val| val.parse::<f32>().ok())
                .filter(|&val| val > 0.0 && val < 1.0)
                .unwrap_or(1.0 / phi()) // 1/PHI
        })
    }

    /// Maximum similarity threshold adjustment - derived from 1/e
    pub fn similarity_threshold_max_adjustment(&self) -> f32 {
        *self.similarity_threshold_max_adjustment.get_or_init(|| {
            env::var("NIODOO_SIMILARITY_THRESHOLD_MAX_ADJUSTMENT")
                .ok()
                .and_then(|val| val.parse::<f32>().ok())
                .filter(|&val| val > 0.0 && val < 1.0)
                .unwrap_or(1.0f32 / std::f32::consts::E) // 1/e
        })
    }

    /// Similarity scaling divisor - derived from e^PI * 100
    pub fn similarity_scaling_divisor(&self) -> f32 {
        *self.similarity_scaling_divisor.get_or_init(|| {
            env::var("NIODOO_SIMILARITY_SCALING_DIVISOR")
                .ok()
                .and_then(|val| val.parse::<f32>().ok())
                .filter(|&val| val > 0.0)
                .unwrap_or((E.powf(PI) * 100.0) as f32) // e^PI * 100
        })
    }
}

/// Get global instance of memory constants
pub fn get_memory_constants() -> &'static MemoryConstants {
    static MEMORY_CONSTANTS: OnceLock<MemoryConstants> = OnceLock::new();

    MEMORY_CONSTANTS.get_or_init(|| MemoryConstants {
        consolidation_threshold_base: OnceLock::new(),
        consolidation_threshold_max: OnceLock::new(),
        merge_threshold_base: OnceLock::new(),
        merge_threshold_max: OnceLock::new(),
        consolidation_scaling_divisor: OnceLock::new(),
        compression_scaling_divisor: OnceLock::new(),
        min_group_size_base: OnceLock::new(),
        group_size_scaling_divisor: OnceLock::new(),
        similarity_threshold_base: OnceLock::new(),
        similarity_threshold_max_adjustment: OnceLock::new(),
        similarity_scaling_divisor: OnceLock::new(),
    })
}
