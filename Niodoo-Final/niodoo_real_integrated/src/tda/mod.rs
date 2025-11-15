pub mod gudhi_bridge;

/// Lightweight plain (birth, death) representation used by the bridge.
#[derive(Debug, Clone)]
pub struct PDPoint {
    pub birth: f64,
    pub death: f64,
}
