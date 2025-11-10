//! Graceful Degradation Tiers
//!
//! Implements tiered resource management with soft zones instead of hard cutoffs:
//! - Tier 1 (70-100%): Mild optimization
//! - Tier 2 (50-70%): Aggressive compression
//! - Tier 3 (30-50%): Emergency mode
//! - Tier 4 (0-30%): Controlled panic
//!
//! Each tier adjusts resource penalty multipliers and curator behavior.

use crate::config::DegradationConfig;
use crate::resource_budget::ResourceAvailability;

/// Degradation tier based on resource availability
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DegradationTier {
    /// Tier 1: 70-100% resources - Mild optimization
    Tier1,
    /// Tier 2: 50-70% resources - Aggressive compression
    Tier2,
    /// Tier 3: 30-50% resources - Emergency mode
    Tier3,
    /// Tier 4: 0-30% resources - Controlled panic
    Tier4,
}

impl DegradationTier {
    /// Get tier name for logging/debugging
    pub fn name(&self) -> &'static str {
        match self {
            DegradationTier::Tier1 => "Tier1 (Mild)",
            DegradationTier::Tier2 => "Tier2 (Aggressive)",
            DegradationTier::Tier3 => "Tier3 (Emergency)",
            DegradationTier::Tier4 => "Tier4 (Panic)",
        }
    }
}

/// Curator mode based on degradation tier
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CuratorMode {
    /// Efficient mode: Standard processing
    Efficient,
    /// Brief mode: Aggressive compression
    Brief,
    /// Emergency mode: Force summarization
    Emergency,
}

impl CuratorMode {
    pub fn name(&self) -> &'static str {
        match self {
            CuratorMode::Efficient => "efficient",
            CuratorMode::Brief => "brief",
            CuratorMode::Emergency => "emergency",
        }
    }
}

/// Degradation manager that determines current tier and actions
pub struct DegradationManager {
    config: DegradationConfig,
}

impl DegradationManager {
    /// Create a new degradation manager
    pub fn new(config: DegradationConfig) -> Self {
        Self { config }
    }

    /// Determine current degradation tier based on resource availability
    pub fn determine_tier(&self, resource_availability: f32) -> DegradationTier {
        if resource_availability >= self.config.tier1_threshold {
            DegradationTier::Tier1
        } else if resource_availability >= self.config.tier2_threshold {
            DegradationTier::Tier2
        } else if resource_availability >= self.config.tier3_threshold {
            DegradationTier::Tier3
        } else {
            DegradationTier::Tier4
        }
    }

    /// Get resource penalty multiplier for a given tier
    pub fn get_resource_penalty_multiplier(&self, tier: DegradationTier) -> f32 {
        match tier {
            DegradationTier::Tier1 => self.config.tier1_multiplier,
            DegradationTier::Tier2 => self.config.tier2_multiplier,
            DegradationTier::Tier3 => self.config.tier3_multiplier,
            DegradationTier::Tier4 => self.config.tier4_multiplier,
        }
    }

    /// Get curator mode for a given tier
    pub fn get_curator_mode(&self, tier: DegradationTier) -> CuratorMode {
        match tier {
            DegradationTier::Tier1 => CuratorMode::Efficient,
            DegradationTier::Tier2 => CuratorMode::Brief,
            DegradationTier::Tier3 | DegradationTier::Tier4 => CuratorMode::Emergency,
        }
    }

    /// Calculate adjusted resource penalty weight based on current tier
    ///
    /// Returns the base w₆ weight multiplied by the tier-specific multiplier
    pub fn calculate_adjusted_resource_weight(
        &self,
        base_resource_weight: f32,
        resource_availability: &ResourceAvailability,
    ) -> f32 {
        let overall_avail = resource_availability.minimum();
        let tier = self.determine_tier(overall_avail);
        let multiplier = self.get_resource_penalty_multiplier(tier);
        base_resource_weight * multiplier
    }

    /// Check if dynamic tokenization compression should be activated
    pub fn should_activate_tokenization_compression(&self, tier: DegradationTier) -> bool {
        matches!(
            tier,
            DegradationTier::Tier2 | DegradationTier::Tier3 | DegradationTier::Tier4
        )
    }

    /// Check if memory summarization should be forced
    pub fn should_force_summarization(&self, tier: DegradationTier) -> bool {
        matches!(tier, DegradationTier::Tier3 | DegradationTier::Tier4)
    }

    /// Check if cooldown state should be entered
    pub fn should_enter_cooldown(&self, tier: DegradationTier) -> bool {
        tier == DegradationTier::Tier4
    }

    /// Check if meta-communication about resource constraints should be sent
    pub fn should_meta_communicate(&self, tier: DegradationTier) -> bool {
        matches!(tier, DegradationTier::Tier3 | DegradationTier::Tier4)
    }

    /// Get meta-communication message for current tier
    pub fn get_meta_message(&self, tier: DegradationTier) -> Option<String> {
        match tier {
            DegradationTier::Tier3 => Some(
                "Resources constrained, activating emergency mode. Response may be compressed."
                    .to_string(),
            ),
            DegradationTier::Tier4 => Some(
                "Critical resource constraints detected. Entering minimal viable output mode."
                    .to_string(),
            ),
            _ => None,
        }
    }

    /// Analyze resource availability and return degradation actions
    pub fn analyze(&self, resource_availability: &ResourceAvailability) -> DegradationAnalysis {
        let overall_avail = resource_availability.minimum();
        let tier = self.determine_tier(overall_avail);
        let curator_mode = self.get_curator_mode(tier);
        let penalty_multiplier = self.get_resource_penalty_multiplier(tier);

        DegradationAnalysis {
            tier,
            curator_mode,
            resource_penalty_multiplier: penalty_multiplier,
            activate_tokenization_compression: self.should_activate_tokenization_compression(tier),
            force_summarization: self.should_force_summarization(tier),
            enter_cooldown: self.should_enter_cooldown(tier),
            meta_message: self.get_meta_message(tier),
            overall_availability: overall_avail,
        }
    }
}

/// Degradation analysis result
#[derive(Debug, Clone)]
pub struct DegradationAnalysis {
    /// Current degradation tier
    pub tier: DegradationTier,
    /// Curator mode to use
    pub curator_mode: CuratorMode,
    /// Resource penalty multiplier for fitness calculation
    pub resource_penalty_multiplier: f32,
    /// Whether to activate tokenization compression
    pub activate_tokenization_compression: bool,
    /// Whether to force memory summarization
    pub force_summarization: bool,
    /// Whether to enter cooldown state
    pub enter_cooldown: bool,
    /// Optional meta-communication message for user
    pub meta_message: Option<String>,
    /// Overall resource availability (0.0-1.0)
    pub overall_availability: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resource_budget::ResourceAvailability;

    #[test]
    fn test_tier_determination() {
        let config = DegradationConfig::default();
        let manager = DegradationManager::new(config);

        // Tier 1
        assert_eq!(manager.determine_tier(0.85), DegradationTier::Tier1);
        assert_eq!(manager.determine_tier(0.70), DegradationTier::Tier1);

        // Tier 2
        assert_eq!(manager.determine_tier(0.60), DegradationTier::Tier2);
        assert_eq!(manager.determine_tier(0.50), DegradationTier::Tier2);

        // Tier 3
        assert_eq!(manager.determine_tier(0.40), DegradationTier::Tier3);
        assert_eq!(manager.determine_tier(0.30), DegradationTier::Tier3);

        // Tier 4
        assert_eq!(manager.determine_tier(0.20), DegradationTier::Tier4);
        assert_eq!(manager.determine_tier(0.05), DegradationTier::Tier4);
    }

    #[test]
    fn test_curator_mode() {
        let config = DegradationConfig::default();
        let manager = DegradationManager::new(config);

        assert_eq!(
            manager.get_curator_mode(DegradationTier::Tier1),
            CuratorMode::Efficient
        );
        assert_eq!(
            manager.get_curator_mode(DegradationTier::Tier2),
            CuratorMode::Brief
        );
        assert_eq!(
            manager.get_curator_mode(DegradationTier::Tier3),
            CuratorMode::Emergency
        );
        assert_eq!(
            manager.get_curator_mode(DegradationTier::Tier4),
            CuratorMode::Emergency
        );
    }

    #[test]
    fn test_penalty_multipliers() {
        let config = DegradationConfig::default();
        let manager = DegradationManager::new(config);

        assert_eq!(
            manager.get_resource_penalty_multiplier(DegradationTier::Tier1),
            1.2
        );
        assert_eq!(
            manager.get_resource_penalty_multiplier(DegradationTier::Tier2),
            2.0
        );
        assert_eq!(
            manager.get_resource_penalty_multiplier(DegradationTier::Tier3),
            5.0
        );
        assert_eq!(
            manager.get_resource_penalty_multiplier(DegradationTier::Tier4),
            10.0
        );
    }

    #[test]
    fn test_analysis() {
        let config = DegradationConfig::default();
        let manager = DegradationManager::new(config);

        // High availability
        let avail_high = ResourceAvailability {
            tokens: 0.9,
            api_rate_limit: 0.9,
            compute_cycles: 0.9,
            memory_bandwidth: 0.9,
        };
        let analysis = manager.analyze(&avail_high);
        assert_eq!(analysis.tier, DegradationTier::Tier1);
        assert_eq!(analysis.curator_mode, CuratorMode::Efficient);
        assert!(!analysis.activate_tokenization_compression);
        assert!(!analysis.force_summarization);

        // Low availability
        let avail_low = ResourceAvailability {
            tokens: 0.2,
            api_rate_limit: 0.2,
            compute_cycles: 0.2,
            memory_bandwidth: 0.2,
        };
        let analysis = manager.analyze(&avail_low);
        assert_eq!(analysis.tier, DegradationTier::Tier4);
        assert_eq!(analysis.curator_mode, CuratorMode::Emergency);
        assert!(analysis.activate_tokenization_compression);
        assert!(analysis.force_summarization);
        assert!(analysis.enter_cooldown);
        assert!(analysis.meta_message.is_some());
    }
}
