//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::mpsc;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use dashmap::DashMap;
use std::any::TypeId;

/// High-performance event bus for cognitive events
pub struct CognitiveEventBus {
    subscribers: Arc<DashMap<TypeId, Vec<EventHandler>>>,
    event_log: Arc<DashMap<SystemTime, CognitiveEvent>>,
    metrics: Arc<EventMetrics>,
}

impl CognitiveEventBus {
    pub fn new() -> Self {
        Self {
            subscribers: Arc::new(DashMap::new()),
            event_log: Arc::new(DashMap::new()),
            metrics: Arc::new(EventMetrics::new()),
        }
    }

    pub async fn publish(&self, event: CognitiveEvent) {
        // Log event
        let timestamp = match &event {
            CognitiveEvent::H0Split { timestamp, .. } => *timestamp,
            CognitiveEvent::H0Merge { timestamp, .. } => *timestamp,
            CognitiveEvent::H1Birth { timestamp, .. } => *timestamp,
            CognitiveEvent::H1Death { timestamp, .. } => *timestamp,
            CognitiveEvent::H2Birth { timestamp, .. } => *timestamp,
            CognitiveEvent::H2Death { timestamp, .. } => *timestamp,
        };
        self.event_log.insert(timestamp, event.clone());

        // Update metrics
        self.metrics.record_event(&event);

        // Notify subscribers in parallel
        if let Some(handlers) = self.subscribers.get(&TypeId::of::<CognitiveEvent>()) {
            let handlers = handlers.clone();
            tokio::spawn(async move {
                for handler in handlers {
                    handler.handle(event.clone()).await;
                }
            });
        }
    }

    pub fn subscribe(&self, handler: EventHandler) {
        self.subscribers
            .entry(TypeId::of::<CognitiveEvent>())
            .or_insert_with(Vec::new)
            .push(handler);
    }
}

/// Event handler trait
#[async_trait]
pub trait EventHandlerTrait: Send + Sync {
    async fn handle(&self, event: CognitiveEvent);
}

pub type EventHandler = Arc<dyn EventHandlerTrait>;

/// Event metrics for monitoring
pub struct EventMetrics {
    h0_events: std::sync::atomic::AtomicUsize,
    h1_events: std::sync::atomic::AtomicUsize,
    h2_events: std::sync::atomic::AtomicUsize,
}

impl EventMetrics {
    pub fn new() -> Self {
        Self {
            h0_events: std::sync::atomic::AtomicUsize::new(0),
            h1_events: std::sync::atomic::AtomicUsize::new(0),
            h2_events: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    pub fn record_event(&self, event: &CognitiveEvent) {
        match event {
            CognitiveEvent::H0Split { .. } | CognitiveEvent::H0Merge { .. } => {
                self.h0_events.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            CognitiveEvent::H1Birth { .. } | CognitiveEvent::H1Death { .. } => {
                self.h1_events.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            CognitiveEvent::H2Birth { .. } | CognitiveEvent::H2Death { .. } => {
                self.h2_events.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
    }
}

// Re-export from core module
pub use super::core::*;