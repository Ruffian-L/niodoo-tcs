/// Triple-threat detection system tests
///
/// Validates the three trigger scenarios:
/// 1. Mismatch Crisis (H > 2.0, mean < 0.7)
/// 2. Uniform Stagnation (H > 2.0, var < 0.01)
/// 3. Variance Spike (var > 0.05)
pub mod test_triple_threat_triggers;
pub mod triple_threat_learning_routine;
