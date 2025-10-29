#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TopologyState {
    #[prost(float, tag = "1")]
    pub entropy: f32,
    #[prost(float, tag = "2")]
    pub iit_phi: f32,
    #[prost(float, repeated, tag = "3")]
    pub knots: ::prost::alloc::vec::Vec<f32>,
    #[prost(int32, repeated, tag = "4")]
    pub betti_numbers: ::prost::alloc::vec::Vec<i32>,
    #[prost(float, tag = "5")]
    pub spectral_gap: f32,
    #[prost(float, tag = "6")]
    pub persistent_entropy: f32,
}

#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PadGhostState {
    #[prost(float, repeated, tag = "1")]
    pub pad: ::prost::alloc::vec::Vec<f32>,
    #[prost(float, repeated, tag = "2")]
    pub mu: ::prost::alloc::vec::Vec<f32>,
    #[prost(float, repeated, tag = "3")]
    pub sigma: ::prost::alloc::vec::Vec<f32>,
    #[prost(float, repeated, tag = "4")]
    pub raw_stds: ::prost::alloc::vec::Vec<f32>,
}

#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ConsciousnessState {
    #[prost(message, optional, tag = "1")]
    pub topology: ::core::option::Option<TopologyState>,
    #[prost(message, optional, tag = "2")]
    pub pad_ghost: ::core::option::Option<PadGhostState>,
    #[prost(string, tag = "3")]
    pub quadrant: ::prost::alloc::string::String,
    #[prost(bool, tag = "4")]
    pub threat: bool,
    #[prost(bool, tag = "5")]
    pub healing: bool,
    #[prost(float, tag = "6")]
    pub rouge_score: f32,
    #[prost(int64, tag = "7")]
    pub timestamp: i64,
}

#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PipelineResponse {
    #[prost(string, tag = "1")]
    pub prompt: ::prost::alloc::string::String,
    #[prost(string, tag = "2")]
    pub baseline_response: ::prost::alloc::string::String,
    #[prost(string, tag = "3")]
    pub hybrid_response: ::prost::alloc::string::String,
    #[prost(message, optional, tag = "4")]
    pub state: ::core::option::Option<ConsciousnessState>,
    #[prost(float, tag = "5")]
    pub latency_ms: f32,
}
