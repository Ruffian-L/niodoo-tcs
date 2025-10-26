use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

use anyhow::{anyhow, Result};

use super::{BettiCurve, DistanceMetric, PersistenceDiagram, PersistenceFeature, PersistenceResult, Point, TopologyEngine, TopologyParams};

/// Reference implementation of a sparsified Vietoris–Rips persistence engine.
#[derive(Debug, Default)]
pub struct RustVREngine;

impl RustVREngine {
    pub fn new() -> Self {
        Self
    }
}

impl TopologyEngine for RustVREngine {
    fn compute_persistence(
        &self,
        points: &[Point],
        max_dim: u8,
        params: &TopologyParams,
    ) -> Result<PersistenceResult> {
        if points.is_empty() {
            return Ok(PersistenceResult::empty());
        }
        if points.iter().any(|p| p.dimensions() != points[0].dimensions()) {
            return Err(anyhow!("all points must share the same dimensionality"));
        }

        let distances = compute_distance_matrix(points, params.metric);
        let edges = build_knn_edges(&distances, params);
        let triangles = if max_dim >= 1 { build_triangles(points.len(), &edges, &distances, params) } else { Vec::new() };

        let simplices = assemble_simplices(points.len(), &edges, &triangles);
        let result = persistent_reduction(&simplices, max_dim.min(1))?;
        crate::metrics::record_topology_metrics(&result);
        Ok(result)
    }
}

#[derive(Debug, Clone)]
struct Simplex {
    dimension: u8,
    vertices: Vec<usize>,
    filtration: f32,
}

impl Simplex {
    fn vertex(index: usize) -> Self {
        Self {
            dimension: 0,
            vertices: vec![index],
            filtration: 0.0,
        }
    }

    fn edge(u: usize, v: usize, filtration: f32) -> Self {
        let mut vertices = vec![u, v];
        vertices.sort_unstable();
        Self {
            dimension: 1,
            vertices,
            filtration,
        }
    }

    fn triangle(mut vertices: [usize; 3], filtration: f32) -> Self {
        vertices.sort_unstable();
        Self {
            dimension: 2,
            vertices: vertices.to_vec(),
            filtration,
        }
    }
}

#[derive(Debug, Clone)]
struct Edge {
    u: usize,
    v: usize,
    filtration: f32,
}

#[derive(Debug, Clone)]
struct Triangle {
    vertices: [usize; 3],
    filtration: f32,
}

fn compute_distance_matrix(points: &[Point], metric: DistanceMetric) -> Vec<Vec<f32>> {
    let n = points.len();
    let mut matrix = vec![vec![0.0_f32; n]; n];

    for i in 0..n {
        for j in (i + 1)..n {
            let d = match metric {
                DistanceMetric::Euclidean => euclidean_distance(&points[i], &points[j]),
            };
            matrix[i][j] = d;
            matrix[j][i] = d;
        }
    }

    matrix
}

fn euclidean_distance(a: &Point, b: &Point) -> f32 {
    a.coords
        .iter()
        .zip(&b.coords)
        .map(|(x, y)| {
            let diff = x - y;
            diff * diff
        })
        .sum::<f32>()
        .sqrt()
}

fn build_knn_edges(distances: &[Vec<f32>], params: &TopologyParams) -> Vec<Edge> {
    let n = distances.len();
    let mut adjacency: Vec<HashSet<usize>> = vec![HashSet::new(); n];

    for i in 0..n {
        let mut neighbors: Vec<(usize, f32)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (j, distances[i][j]))
            .collect();
        neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        for &(neighbor, dist) in neighbors.iter().take(params.k) {
            if let Some(max_r) = params.max_filtration_value {
                if dist > max_r {
                    continue;
                }
            }
            adjacency[i].insert(neighbor);
            adjacency[neighbor].insert(i);
        }
    }

    let mut edges = Vec::new();
    let mut seen = HashSet::new();
    for i in 0..n {
        for &j in &adjacency[i] {
            if i < j && seen.insert((i, j)) {
                edges.push(Edge {
                    u: i,
                    v: j,
                    filtration: distances[i][j],
                });
            }
        }
    }
    edges.sort_by(|a, b| a
        .filtration
        .partial_cmp(&b.filtration)
        .unwrap_or(Ordering::Equal));
    edges
}

fn build_triangles(
    n: usize,
    edges: &[Edge],
    distances: &[Vec<f32>],
    params: &TopologyParams,
) -> Vec<Triangle> {
    let mut adjacency: Vec<HashSet<usize>> = vec![HashSet::new(); n];
    for edge in edges {
        adjacency[edge.u].insert(edge.v);
        adjacency[edge.v].insert(edge.u);
    }

    let mut triangles = Vec::new();
    let mut seen = HashSet::new();
    for i in 0..n {
        let neighbors: Vec<usize> = adjacency[i].iter().cloned().collect();
        for a_idx in 0..neighbors.len() {
            for b_idx in (a_idx + 1)..neighbors.len() {
                let mut vertices = [i, neighbors[a_idx], neighbors[b_idx]];
                vertices.sort_unstable();
                let (v0, v1, v2) = (vertices[0], vertices[1], vertices[2]);
                if !adjacency[v0].contains(&v1) || !adjacency[v0].contains(&v2) || !adjacency[v1].contains(&v2) {
                    continue;
                }
                if !seen.insert((v0, v1, v2)) {
                    continue;
                }
                let filtration = distances[v0][v1]
                    .max(distances[v0][v2])
                    .max(distances[v1][v2]);
                if let Some(max_r) = params.max_filtration_value {
                    if filtration > max_r {
                        continue;
                    }
                }
                triangles.push(Triangle { vertices, filtration });
            }
        }
    }
    triangles.sort_by(|a, b| a
        .filtration
        .partial_cmp(&b.filtration)
        .unwrap_or(Ordering::Equal));
    triangles
}

fn assemble_simplices(n: usize, edges: &[Edge], triangles: &[Triangle]) -> Vec<Simplex> {
    let mut simplices: Vec<Simplex> = (0..n).map(Simplex::vertex).collect();
    simplices.extend(edges.iter().map(|e| Simplex::edge(e.u, e.v, e.filtration)));
    simplices.extend(triangles.iter().map(|t| Simplex::triangle(t.vertices, t.filtration)));

    simplices.sort_by(|a, b| {
        a.filtration
            .partial_cmp(&b.filtration)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.dimension.cmp(&b.dimension))
            .then_with(|| a.vertices.cmp(&b.vertices))
    });
    simplices
}

fn persistent_reduction(simplices: &[Simplex], target_dim: u8) -> Result<PersistenceResult> {
    if simplices.is_empty() {
        return Ok(PersistenceResult::empty());
    }

    let mut vertex_index: HashMap<usize, usize> = HashMap::new();
    let mut edge_index: HashMap<(usize, usize), usize> = HashMap::new();
    for (idx, simplex) in simplices.iter().enumerate() {
        match simplex.dimension {
            0 => {
                vertex_index.insert(simplex.vertices[0], idx);
            }
            1 => {
                let key = edge_key(simplex.vertices[0], simplex.vertices[1]);
                edge_index.insert(key, idx);
            }
            _ => {}
        }
    }

    let target_dim = target_dim.min(1) as usize;
    let mut diagrams: Vec<PersistenceDiagram> = (0..=target_dim)
        .map(PersistenceDiagram::new)
        .collect();
    let mut columns: Vec<Vec<usize>> = vec![Vec::new(); simplices.len()];
    let mut low_map: HashMap<usize, usize> = HashMap::new();
    let mut creators: HashMap<usize, (usize, usize)> = HashMap::new();

    for (j, simplex) in simplices.iter().enumerate() {
        let mut column = boundary(simplex, &vertex_index, &edge_index)?;

        loop {
            let low = column.last().cloned();
            if let Some(low_simplex) = low {
                if let Some(&paired_column) = low_map.get(&low_simplex) {
                    let replacement = xor_columns(&column, &columns[paired_column]);
                    column = replacement;
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        if column.is_empty() {
            if simplex.dimension as usize <= target_dim {
                let dim = simplex.dimension as usize;
                let feature_idx = diagrams[dim].features.len();
                diagrams[dim].features.push(PersistenceFeature {
                    birth: simplex.filtration,
                    death: f32::INFINITY,
                    dimension: dim,
                });
                creators.insert(j, (dim, feature_idx));
            }
        } else {
            let low_simplex = *column.last().unwrap();
            low_map.insert(low_simplex, j);
            if let Some((dim, feature_idx)) = creators.remove(&low_simplex) {
                if dim <= target_dim {
                    diagrams[dim].features[feature_idx].death = simplex.filtration;
                }
            }
        }
        columns[j] = column;
    }

    // Remove triangles if we never intended to consume H2 results.
    let diagrams: Vec<PersistenceDiagram> = diagrams
        .into_iter()
        .map(|mut d| {
            d.features.sort_by(|a, b| a
                .birth
                .partial_cmp(&b.birth)
                .unwrap_or(Ordering::Equal));
            d
        })
        .collect();
    let betti_curves: Vec<BettiCurve> = diagrams.iter().map(|d| d.betti_curve()).collect();
    let entropy = diagrams
        .iter()
        .map(|d| (d.dimension, d.persistent_entropy()))
        .collect();
    Ok(PersistenceResult {
        diagrams,
        betti_curves,
        entropy,
    })
}

fn boundary(
    simplex: &Simplex,
    vertex_index: &HashMap<usize, usize>,
    edge_index: &HashMap<(usize, usize), usize>,
) -> Result<Vec<usize>> {
    match simplex.dimension {
        0 => Ok(Vec::new()),
        1 => {
            let u = simplex.vertices[0];
            let v = simplex.vertices[1];
            let mut boundary = vec![vertex_index
                .get(&u)
                .copied()
                .ok_or_else(|| anyhow!("missing vertex {} in boundary", u))?, vertex_index
                .get(&v)
                .copied()
                .ok_or_else(|| anyhow!("missing vertex {} in boundary", v))?];
            boundary.sort_unstable();
            Ok(boundary)
        }
        2 => {
            let v = &simplex.vertices;
            let mut boundary = vec![
                edge_index
                    .get(&edge_key(v[0], v[1]))
                    .copied()
                    .ok_or_else(|| anyhow!("missing edge ({}, {}) in boundary", v[0], v[1]))?,
                edge_index
                    .get(&edge_key(v[0], v[2]))
                    .copied()
                    .ok_or_else(|| anyhow!("missing edge ({}, {}) in boundary", v[0], v[2]))?,
                edge_index
                    .get(&edge_key(v[1], v[2]))
                    .copied()
                    .ok_or_else(|| anyhow!("missing edge ({}, {}) in boundary", v[1], v[2]))?,
            ];
            boundary.sort_unstable();
            Ok(boundary)
        }
        _ => Ok(Vec::new()),
    }
}

fn xor_columns(a: &[usize], b: &[usize]) -> Vec<usize> {
    let mut result = Vec::new();
    let mut i = 0;
    let mut j = 0;
    while i < a.len() || j < b.len() {
        match (a.get(i), b.get(j)) {
            (Some(&va), Some(&vb)) => {
                if va == vb {
                    i += 1;
                    j += 1;
                } else if va < vb {
                    result.push(va);
                    i += 1;
                } else {
                    result.push(vb);
                    j += 1;
                }
            }
            (Some(&va), None) => {
                result.push(va);
                i += 1;
            }
            (None, Some(&vb)) => {
                result.push(vb);
                j += 1;
            }
            (None, None) => break,
        }
    }
    result
}

#[inline]
fn edge_key(u: usize, v: usize) -> (usize, usize) {
    if u < v { (u, v) } else { (v, u) }
}
