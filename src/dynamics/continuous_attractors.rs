//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

/*
 * 🔄 CONTINUOUS ATTRACTOR NETWORKS (CANs) 🔄
 *
 * Implements Continuous Attractor Networks for generative modeling
 * of geometric structures from neural interactions.
 *
 * Based on "The Geometry of Thought" framework:
 * - Mexican-hat connectivity for pattern formation
 * - Ring attractors for head direction
 * - Toroidal attractors for grid cells
 * - Emergent manifold generation
 */

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// Continuous attractor network parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CANParameters {
    pub num_neurons: usize,
    pub connectivity_radius: f64,
    pub excitation_strength: f64,
    pub inhibition_strength: f64,
    pub time_constant: f64,
    pub noise_level: f64,
    pub learning_rate: f64,
}

impl Default for CANParameters {
    fn default() -> Self {
        Self {
            num_neurons: 100,
            connectivity_radius: 0.3,
            excitation_strength: 1.0,
            inhibition_strength: 0.5,
            time_constant: 10.0,
            noise_level: 0.01,
            learning_rate: 0.01,
        }
    }
}

/// Mexican-hat connectivity kernel
pub struct MexicanHatKernel {
    pub excitation_radius: f64,
    pub inhibition_radius: f64,
    pub excitation_strength: f64,
    pub inhibition_strength: f64,
}

impl MexicanHatKernel {
    pub fn new(
        excitation_radius: f64,
        inhibition_radius: f64,
        excitation_strength: f64,
        inhibition_strength: f64,
    ) -> Self {
        Self {
            excitation_radius,
            inhibition_radius,
            excitation_strength,
            inhibition_strength,
        }
    }

    /// Calculate connection strength between two neurons
    pub fn connection_strength(&self, distance: f64) -> f64 {
        let excitation = self.excitation_strength
            * (-distance.powi(2) / (2.0 * self.excitation_radius.powi(2))).exp();
        let inhibition = self.inhibition_strength
            * (-distance.powi(2) / (2.0 * self.inhibition_radius.powi(2))).exp();
        excitation - inhibition
    }

    /// Generate connectivity matrix for 1D ring
    pub fn generate_ring_connectivity(&self, num_neurons: usize) -> Array2<f64> {
        let mut connectivity = Array2::zeros((num_neurons, num_neurons));

        for i in 0..num_neurons {
            for j in 0..num_neurons {
                let distance = self.ring_distance(i, j, num_neurons);
                connectivity[[i, j]] = self.connection_strength(distance);
            }
        }

        connectivity
    }

    /// Generate connectivity matrix for 2D torus
    pub fn generate_torus_connectivity(&self, width: usize, height: usize) -> Array2<f64> {
        let num_neurons = width * height;
        let mut connectivity = Array2::zeros((num_neurons, num_neurons));

        for i in 0..num_neurons {
            for j in 0..num_neurons {
                let distance = self.torus_distance(i, j, width, height);
                connectivity[[i, j]] = self.connection_strength(distance);
            }
        }

        connectivity
    }

    /// Calculate distance on 1D ring
    fn ring_distance(&self, i: usize, j: usize, num_neurons: usize) -> f64 {
        let diff = (i as i32 - j as i32).abs() as f64;
        let circular_diff = diff.min(num_neurons as f64 - diff);
        circular_diff / num_neurons as f64
    }

    /// Calculate distance on 2D torus
    fn torus_distance(&self, i: usize, j: usize, width: usize, height: usize) -> f64 {
        let i_x = i % width;
        let i_y = i / width;
        let j_x = j % width;
        let j_y = j / width;

        let dx = ((i_x as i32 - j_x as i32).abs() as f64)
            .min(width as f64 - (i_x as i32 - j_x as i32).abs() as f64);
        let dy = ((i_y as i32 - j_y as i32).abs() as f64)
            .min(height as f64 - (i_y as i32 - j_y as i32).abs() as f64);

        (dx.powi(2) + dy.powi(2)).sqrt() / ((width * height) as f64).sqrt()
    }
}

/// Continuous attractor network
pub struct ContinuousAttractorNetwork {
    pub parameters: CANParameters,
    pub connectivity: Array2<f64>,
    pub activity: Array1<f64>,
    pub input: Array1<f64>,
    pub kernel: MexicanHatKernel,
}

impl ContinuousAttractorNetwork {
    pub fn new(parameters: CANParameters) -> Self {
        let kernel = MexicanHatKernel::new(
            parameters.connectivity_radius,
            parameters.connectivity_radius * 2.0,
            parameters.excitation_strength,
            parameters.inhibition_strength,
        );

        let connectivity = kernel.generate_ring_connectivity(parameters.num_neurons);
        let activity = Array1::zeros(parameters.num_neurons);
        let input = Array1::zeros(parameters.num_neurons);

        Self {
            parameters,
            connectivity,
            activity,
            input,
            kernel,
        }
    }

    /// Create ring attractor for head direction
    pub fn new_ring_attractor(num_neurons: usize) -> Self {
        let parameters = CANParameters {
            num_neurons,
            connectivity_radius: 0.2,
            excitation_strength: 1.5,
            inhibition_strength: 0.8,
            time_constant: 5.0,
            noise_level: 0.005,
            learning_rate: 0.01,
        };

        Self::new(parameters)
    }

    /// Create toroidal attractor for grid cells
    pub fn new_toroidal_attractor(width: usize, height: usize) -> Self {
        let num_neurons = width * height;
        let parameters = CANParameters {
            num_neurons,
            connectivity_radius: 0.3,
            excitation_strength: 1.2,
            inhibition_strength: 0.6,
            time_constant: 8.0,
            noise_level: 0.01,
            learning_rate: 0.005,
        };

        let mut can = Self::new(parameters);
        can.connectivity = can.kernel.generate_torus_connectivity(width, height);
        can
    }

    /// Update network activity using differential equation
    pub fn update(&mut self, dt: f64) {
        let mut new_activity = Array1::zeros(self.parameters.num_neurons);

        for i in 0..self.parameters.num_neurons {
            let mut total_input = 0.0;

            // Sum weighted inputs from all neurons
            for j in 0..self.parameters.num_neurons {
                total_input += self.connectivity[[i, j]] * self.activity[j];
            }

            // Add external input
            total_input += self.input[i];

            // Add noise
            total_input += self.parameters.noise_level * (2.0 * rand::random::<f64>() - 1.0);

            // Apply activation function (sigmoid)
            let activated = 1.0 / (1.0 + (-total_input).exp());

            // Update activity with time constant
            new_activity[i] = self.activity[i]
                + (dt / self.parameters.time_constant) * (activated - self.activity[i]);
        }

        self.activity = new_activity;
    }

    /// Set external input
    pub fn set_input(&mut self, input: Array1<f64>) {
        self.input = input;
    }

    /// Get current activity pattern
    pub fn get_activity(&self) -> &Array1<f64> {
        &self.activity
    }

    /// Find activity bump center
    pub fn find_bump_center(&self) -> f64 {
        let mut weighted_sum = 0.0;
        let mut total_activity = 0.0;

        for (i, &activity) in self.activity.iter().enumerate() {
            weighted_sum += i as f64 * activity;
            total_activity += activity;
        }

        if total_activity > 0.0 {
            weighted_sum / total_activity
        } else {
            0.0
        }
    }

    /// Calculate bump width
    pub fn calculate_bump_width(&self) -> f64 {
        let center = self.find_bump_center();
        let mut variance = 0.0;
        let mut total_activity = 0.0;

        for (i, &activity) in self.activity.iter().enumerate() {
            let distance = (i as f64 - center).abs();
            variance += distance.powi(2) * activity;
            total_activity += activity;
        }

        if total_activity > 0.0 {
            (variance / total_activity).sqrt()
        } else {
            0.0
        }
    }

    /// Simulate network dynamics
    pub fn simulate(&mut self, duration: f64, dt: f64) -> Vec<Array1<f64>> {
        let num_steps = (duration / dt) as usize;
        let mut trajectory = Vec::new();

        for _ in 0..num_steps {
            self.update(dt);
            trajectory.push(self.activity.clone());
        }

        trajectory
    }
}

/// Attractor landscape for cognitive states
pub struct AttractorLandscape {
    pub attractors: Vec<Attractor>,
    pub potential_function: Array2<f64>,
    pub state_space: Array2<f64>,
}

#[derive(Debug, Clone)]
pub struct Attractor {
    pub position: Array1<f64>,
    pub strength: f64,
    pub basin_radius: f64,
    pub attractor_type: AttractorType,
}

#[derive(Debug, Clone)]
pub enum AttractorType {
    FixedPoint,
    LimitCycle,
    StrangeAttractor,
}

impl AttractorLandscape {
    pub fn new(_state_dimensions: usize, resolution: usize) -> Self {
        Self {
            attractors: Vec::new(),
            potential_function: Array2::zeros((resolution, resolution)),
            state_space: Array2::zeros((resolution, resolution)),
        }
    }

    /// Add an attractor to the landscape
    pub fn add_attractor(&mut self, attractor: Attractor) {
        self.attractors.push(attractor);
        self.update_potential_function();
    }

    /// Update potential function based on attractors
    fn update_potential_function(&mut self) {
        let (height, width) = self.potential_function.dim();

        for i in 0..height {
            for j in 0..width {
                let x = (i as f64) / (height as f64) * 2.0 - 1.0;
                let y = (j as f64) / (width as f64) * 2.0 - 1.0;

                let mut potential = 0.0;

                for attractor in &self.attractors {
                    let distance = ((x - attractor.position[0]).powi(2)
                        + (y - attractor.position[1]).powi(2))
                    .sqrt();
                    potential += attractor.strength
                        * (-distance.powi(2) / (2.0 * attractor.basin_radius.powi(2))).exp();
                }

                self.potential_function[[i, j]] = potential;
            }
        }
    }

    /// Calculate gradient at a point
    pub fn calculate_gradient(&self, position: &Array1<f64>) -> Array1<f64> {
        let mut gradient = Array1::zeros(position.len());

        for attractor in &self.attractors {
            let distance = ((position[0] - attractor.position[0]).powi(2)
                + (position[1] - attractor.position[1]).powi(2))
            .sqrt();

            if distance > 0.0 {
                let force = attractor.strength
                    * (-distance.powi(2) / (2.0 * attractor.basin_radius.powi(2))).exp()
                    * distance
                    / attractor.basin_radius.powi(2);

                gradient[0] += force * (attractor.position[0] - position[0]) / distance;
                gradient[1] += force * (attractor.position[1] - position[1]) / distance;
            }
        }

        gradient
    }

    /// Simulate dynamics on the landscape
    pub fn simulate_dynamics(
        &self,
        initial_position: Array1<f64>,
        duration: f64,
        dt: f64,
    ) -> Vec<Array1<f64>> {
        let mut position = initial_position;
        let mut trajectory = Vec::new();
        let num_steps = (duration / dt) as usize;

        for _ in 0..num_steps {
            let gradient = self.calculate_gradient(&position);
            position = &position + &(&gradient * dt);
            trajectory.push(position.clone());
        }

        trajectory
    }
}

/// Bifurcation analysis for attractor transitions
pub struct BifurcationAnalyzer {
    pub parameter_range: (f64, f64),
    pub num_points: usize,
}

impl BifurcationAnalyzer {
    pub fn new(parameter_range: (f64, f64), num_points: usize) -> Self {
        Self {
            parameter_range,
            num_points,
        }
    }

    /// Analyze bifurcation points
    pub fn analyze_bifurcation(
        &self,
        can: &mut ContinuousAttractorNetwork,
        parameter_name: &str,
    ) -> Vec<(f64, f64)> {
        let mut bifurcation_points = Vec::new();
        let (start, end) = self.parameter_range;
        let step = (end - start) / (self.num_points as f64);

        for i in 0..self.num_points {
            let parameter_value = start + i as f64 * step;

            // Update parameter
            match parameter_name {
                "excitation_strength" => can.parameters.excitation_strength = parameter_value,
                "inhibition_strength" => can.parameters.inhibition_strength = parameter_value,
                "noise_level" => can.parameters.noise_level = parameter_value,
                _ => {}
            }

            // Simulate and measure stability
            let trajectory = can.simulate(10.0, 0.01);
            let stability = self.measure_stability(&trajectory);

            bifurcation_points.push((parameter_value, stability));
        }

        bifurcation_points
    }

    /// Measure stability of trajectory
    fn measure_stability(&self, trajectory: &[Array1<f64>]) -> f64 {
        if trajectory.len() < 2 {
            return 0.0;
        }

        let mut variance = 0.0;
        let last_activity = &trajectory[trajectory.len() - 1];

        for activity in trajectory.iter().rev().take(10) {
            let diff = activity - last_activity;
            variance += diff.iter().map(|&x| x.powi(2)).sum::<f64>();
        }

        variance / trajectory.len() as f64
    }
}

/// Emergent manifold generator
pub struct EmergentManifoldGenerator {
    pub can: ContinuousAttractorNetwork,
    pub manifold_dimension: usize,
}

impl EmergentManifoldGenerator {
    pub fn new(can: ContinuousAttractorNetwork, manifold_dimension: usize) -> Self {
        Self {
            can,
            manifold_dimension,
        }
    }

    /// Generate manifold from CAN dynamics
    pub fn generate_manifold(&mut self, num_samples: usize) -> Array2<f64> {
        let mut manifold_points = Array2::zeros((num_samples, self.manifold_dimension));

        for i in 0..num_samples {
            // Random initial condition
            let mut initial_input = Array1::zeros(self.can.parameters.num_neurons);
            for j in 0..self.can.parameters.num_neurons {
                initial_input[j] = rand::random::<f64>() * 0.1;
            }

            self.can.set_input(initial_input);

            // Simulate until convergence
            let trajectory = self.can.simulate(5.0, 0.01);
            let final_activity = &trajectory[trajectory.len() - 1];

            // Extract manifold coordinates
            for j in 0..self.manifold_dimension {
                manifold_points[[i, j]] = final_activity[j % final_activity.len()];
            }
        }

        manifold_points
    }

    /// Detect manifold topology
    pub fn detect_topology(&self, manifold_points: &Array2<f64>) -> String {
        // Simplified topology detection
        let _num_points = manifold_points.nrows();
        let mut _connected_components = 0;
        let mut loops = 0;

        // Count connected components (simplified)
        _connected_components = 1; // Assume single connected component

        // Count loops (simplified heuristic)
        if self.manifold_dimension >= 2 {
            loops = 1; // Assume single loop for 2D manifold
        }

        format!("Topology: β₀={}, β₁={}", _connected_components, loops)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mexican_hat_kernel() {
        let kernel = MexicanHatKernel::new(0.2, 0.4, 1.0, 0.5);
        let strength = kernel.connection_strength(0.1);
        assert!(strength > 0.0);
    }

    #[test]
    fn test_can_creation() {
        let parameters = CANParameters::default();
        let can = ContinuousAttractorNetwork::new(parameters);
        assert_eq!(can.parameters.num_neurons, 100);
    }

    #[test]
    fn test_ring_attractor() {
        let mut ring_can = ContinuousAttractorNetwork::new_ring_attractor(50);
        let trajectory = ring_can.simulate(1.0, 0.01);
        assert!(!trajectory.is_empty());
    }

    #[test]
    fn test_attractor_landscape() {
        let mut landscape = AttractorLandscape::new(2, 100);
        let attractor = Attractor {
            position: Array1::from_vec(vec![0.0, 0.0]),
            strength: 1.0,
            basin_radius: 0.5,
            attractor_type: AttractorType::FixedPoint,
        };
        landscape.add_attractor(attractor);
        assert_eq!(landscape.attractors.len(), 1);
    }

    #[test]
    fn test_bifurcation_analysis() {
        let mut can = ContinuousAttractorNetwork::new_ring_attractor(20);
        let analyzer = BifurcationAnalyzer::new((0.5, 2.0), 10);
        let bifurcation_points = analyzer.analyze_bifurcation(&mut can, "excitation_strength");
        assert_eq!(bifurcation_points.len(), 10);
    }
}
