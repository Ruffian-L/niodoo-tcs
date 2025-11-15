//! Training Service Module
//!
//! Provides a production-grade training service that runs independently from the test loop,
//! enabling non-blocking training requests and parallel training workers.

pub mod adapter_storage;
pub mod client;
pub mod job_queue;
pub mod server;
pub mod worker;

pub use adapter_storage::{AdapterStorage, AdapterVersion};
pub use client::TrainingServiceClient;
pub use job_queue::{JobQueue, JobStatus, TrainingJob};
pub use server::TrainingServiceServer;
pub use worker::TrainingWorker;
