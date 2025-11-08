//! Validation module - statistical analysis and comparison utilities

pub mod stats;
pub mod locomo;
pub mod aqa_bench;
pub mod docpuzzle;
pub mod counterbench;
pub mod criticbench;
pub mod ablation_studies;
pub mod topology_validation;
pub mod benchmarks;
pub mod learning_validation;
pub mod scale_testing;
pub mod roi_analysis;
pub mod terminology_validation;
pub mod report_generator;

pub use stats::*;
pub use locomo::*;
pub use aqa_bench::*;
pub use docpuzzle::*;
pub use counterbench::*;
pub use criticbench::*;
pub use ablation_studies::*;
pub use report_generator::*;

