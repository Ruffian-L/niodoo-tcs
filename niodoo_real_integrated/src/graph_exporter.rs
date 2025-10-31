//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham
//!
//! Graph Exporter - Phase 2 Integration Module
//!
//! This module exports GuessingMemorySystem to JSON/GraphML format for visualization.
//! It serializes spheres (nodes), links (edges), positions, and emotions.

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use niodoo_core::memory::{EmotionalVector, GuessingMemorySystem};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Write;
use std::path::Path;
use tracing::{debug, info};

/// Node representation for graph export
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    pub id: String,
    pub label: String,
    pub position: [f32; 3],
    pub emotion: EmotionalVector,
    pub concept: String,
    pub memory_fragment: String,
}

/// Edge representation for graph export
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    pub source: String,
    pub target: String,
    pub probability: f32,
    pub emotional_weight: EmotionalVector,
    pub weight: f32, // Combined weight for visualization
}

/// Graph metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphMetadata {
    pub export_timestamp: DateTime<Utc>,
    pub sphere_count: usize,
    pub link_count: usize,
    pub version: String,
}

/// Complete graph export structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphExport {
    pub metadata: GraphMetadata,
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
}

/// Graph exporter for GuessingMemorySystem
pub struct GraphExporter;

impl GraphExporter {
    /// Export graph to JSON format
    pub fn export_to_json(
        graph: &GuessingMemorySystem,
        output_path: impl AsRef<Path>,
    ) -> Result<GraphExport> {
        info!("Exporting graph to JSON: {:?}", output_path.as_ref());
        
        let export = Self::build_export(graph)?;
        
        let json = serde_json::to_string_pretty(&export)
            .context("Failed to serialize graph to JSON")?;
        
        let mut file = File::create(&output_path)
            .context("Failed to create output file")?;
        
        file.write_all(json.as_bytes())
            .context("Failed to write JSON to file")?;
        
        info!(
            "Exported {} nodes and {} edges to {:?}",
            export.nodes.len(),
            export.edges.len(),
            output_path.as_ref()
        );
        
        Ok(export)
    }

    /// Export graph to GraphML format
    pub fn export_to_graphml(
        graph: &GuessingMemorySystem,
        output_path: impl AsRef<Path>,
    ) -> Result<()> {
        info!("Exporting graph to GraphML: {:?}", output_path.as_ref());
        
        let export = Self::build_export(graph)?;
        
        let mut file = File::create(&output_path)
            .context("Failed to create output file")?;
        
        // Write GraphML header
        writeln!(file, r#"<?xml version="1.0" encoding="UTF-8"?>"#)?;
        writeln!(
            file,
            r#"<graphml xmlns="http://graphml.graphdrawing.org/xmlns" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">"#
        )?;
        
        // Define attributes
        writeln!(file, r#"  <key id="emotion" for="node" attr.name="emotion" attr.type="string"/>"#)?;
        writeln!(file, r#"  <key id="concept" for="node" attr.name="concept" attr.type="string"/>"#)?;
        writeln!(file, r#"  <key id="x" for="node" attr.name="x" attr.type="float"/>"#)?;
        writeln!(file, r#"  <key id="y" for="node" attr.name="y" attr.type="float"/>"#)?;
        writeln!(file, r#"  <key id="z" for="node" attr.name="z" attr.type="float"/>"#)?;
        writeln!(file, r#"  <key id="probability" for="edge" attr.name="probability" attr.type="float"/>"#)?;
        writeln!(file, r#"  <key id="weight" for="edge" attr.name="weight" attr.type="float"/>"#)?;
        
        // Write graph
        writeln!(file, r#"  <graph id="emotional_graph" edgedefault="directed">"#)?;
        
        // Write nodes
        for node in &export.nodes {
            writeln!(
                file,
                r#"    <node id="{}">"#,
                xml_escape(&node.id)
            )?;
            writeln!(file, r#"      <data key="label">{}</data>"#, xml_escape(&node.label))?;
            writeln!(
                file,
                r#"      <data key="concept">{}</data>"#,
                xml_escape(&node.concept)
            )?;
            writeln!(
                file,
                r#"      <data key="emotion">joy={:.3},sadness={:.3},anger={:.3},fear={:.3},surprise={:.3}</data>"#,
                node.emotion.joy,
                node.emotion.sadness,
                node.emotion.anger,
                node.emotion.fear,
                node.emotion.surprise
            )?;
            writeln!(file, r#"      <data key="x">{}</data>"#, node.position[0])?;
            writeln!(file, r#"      <data key="y">{}</data>"#, node.position[1])?;
            writeln!(file, r#"      <data key="z">{}</data>"#, node.position[2])?;
            writeln!(file, r#"    </node>"#)?;
        }
        
        // Write edges
        for edge in &export.edges {
            writeln!(
                file,
                r#"    <edge source="{}" target="{}">"#,
                xml_escape(&edge.source),
                xml_escape(&edge.target)
            )?;
            writeln!(file, r#"      <data key="probability">{}</data>"#, edge.probability)?;
            writeln!(file, r#"      <data key="weight">{}</data>"#, edge.weight)?;
            writeln!(file, r#"    </edge>"#)?;
        }
        
        writeln!(file, r#"  </graph>"#)?;
        writeln!(file, r#"</graphml>"#)?;
        
        info!(
            "Exported {} nodes and {} edges to GraphML {:?}",
            export.nodes.len(),
            export.edges.len(),
            output_path.as_ref()
        );
        
        Ok(())
    }

    /// Build export structure from graph
    fn build_export(graph: &GuessingMemorySystem) -> Result<GraphExport> {
        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        
        // Collect all spheres as nodes
        for sphere in graph.spheres() {
            let node = GraphNode {
                id: sphere.id.0.clone(),
                label: sphere.core_concept.clone(),
                position: sphere.position,
                emotion: sphere.emotional_profile.clone(),
                concept: sphere.core_concept.clone(),
                memory_fragment: sphere.memory_fragment.clone(),
            };
            nodes.push(node);
        }
        
        // Collect all links as edges
        for sphere in graph.spheres() {
            for (target_id, link) in &sphere.links {
                let edge = GraphEdge {
                    source: sphere.id.0.clone(),
                    target: target_id.0.clone(),
                    probability: link.probability,
                    emotional_weight: link.emotional_weight.clone(),
                    weight: link.probability * link.emotional_weight.magnitude(),
                };
                edges.push(edge);
            }
        }
        
        let metadata = GraphMetadata {
            export_timestamp: Utc::now(),
            sphere_count: nodes.len(),
            link_count: edges.len(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        };
        
        debug!(
            "Built export: {} nodes, {} edges",
            metadata.sphere_count, metadata.link_count
        );
        
        Ok(GraphExport {
            metadata,
            nodes,
            edges,
        })
    }

    /// Export partial graph filtered by emotion similarity
    pub fn export_filtered_by_emotion(
        graph: &GuessingMemorySystem,
        query_emotion: &EmotionalVector,
        threshold: f32,
        output_path: impl AsRef<Path>,
    ) -> Result<GraphExport> {
        info!(
            "Exporting graph filtered by emotion similarity (threshold={})",
            threshold
        );
        
        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        let mut included_sphere_ids = std::collections::HashSet::new();
        
        // Filter nodes by emotion similarity
        for sphere in graph.spheres() {
            let similarity = sphere.emotional_similarity(query_emotion);
            if similarity >= threshold {
                included_sphere_ids.insert(sphere.id.0.clone());
                
                let node = GraphNode {
                    id: sphere.id.0.clone(),
                    label: sphere.core_concept.clone(),
                    position: sphere.position,
                    emotion: sphere.emotional_profile.clone(),
                    concept: sphere.core_concept.clone(),
                    memory_fragment: sphere.memory_fragment.clone(),
                };
                nodes.push(node);
            }
        }
        
        // Include edges only between included nodes
        for sphere in graph.spheres() {
            if !included_sphere_ids.contains(&sphere.id.0) {
                continue;
            }
            
            for (target_id, link) in &sphere.links {
                if included_sphere_ids.contains(&target_id.0) {
                    let edge = GraphEdge {
                        source: sphere.id.0.clone(),
                        target: target_id.0.clone(),
                        probability: link.probability,
                        emotional_weight: link.emotional_weight.clone(),
                        weight: link.probability * link.emotional_weight.magnitude(),
                    };
                    edges.push(edge);
                }
            }
        }
        
        let metadata = GraphMetadata {
            export_timestamp: Utc::now(),
            sphere_count: nodes.len(),
            link_count: edges.len(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        };
        
        let export = GraphExport {
            metadata,
            nodes,
            edges,
        };
        
        let json = serde_json::to_string_pretty(&export)
            .context("Failed to serialize filtered graph to JSON")?;
        
        let mut file = File::create(&output_path)
            .context("Failed to create output file")?;
        
        file.write_all(json.as_bytes())
            .context("Failed to write JSON to file")?;
        
        info!(
            "Exported {} filtered nodes and {} edges to {:?}",
            export.nodes.len(),
            export.edges.len(),
            output_path.as_ref()
        );
        
        Ok(export)
    }
}

/// Escape XML special characters
fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

#[cfg(test)]
mod tests {
    use super::*;
    use niodoo_core::memory::SphereId;

    #[test]
    fn test_build_export() {
        let mut graph = GuessingMemorySystem::new();
        
        let id1 = SphereId("sphere1".to_string());
        let id2 = SphereId("sphere2".to_string());
        
        graph.store_memory(
            id1.clone(),
            "Test concept 1".to_string(),
            [0.0, 0.0, 0.0],
            EmotionalVector::new(0.8, 0.1, 0.0, 0.0, 0.1),
            "Fragment 1".to_string(),
        );
        
        graph.store_memory(
            id2.clone(),
            "Test concept 2".to_string(),
            [1.0, 1.0, 1.0],
            EmotionalVector::new(0.7, 0.2, 0.0, 0.0, 0.1),
            "Fragment 2".to_string(),
        );
        
        // Add a link
        if let Some(sphere) = graph.spheres_mut().find(|s| s.id == id1) {
            sphere.add_link(
                id2.clone(),
                0.5,
                EmotionalVector::new(0.75, 0.15, 0.0, 0.0, 0.1),
            );
        }
        
        let export = GraphExporter::build_export(&graph).unwrap();
        assert_eq!(export.nodes.len(), 2);
        assert_eq!(export.edges.len(), 1);
    }

    #[test]
    fn test_xml_escape() {
        assert_eq!(xml_escape("test&test"), "test&amp;test");
        assert_eq!(xml_escape("test<test"), "test&lt;test");
        assert_eq!(xml_escape("test>test"), "test&gt;test");
    }
}

