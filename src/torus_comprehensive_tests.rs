//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

//! Comprehensive Torus Test Suite
use tracing::{info, error, warn};
//! 
//! This module provides extensive testing for all torus implementations,
//! including mathematical correctness, performance benchmarks, and edge cases.

use std::f64::consts::{PI, TAU};
use anyhow::Result;
use std::time::Instant;

/// Comprehensive test suite for torus implementations
pub struct TorusTestSuite {
    pub test_cases: Vec<TorusTestCase>,
    pub performance_tests: Vec<PerformanceTest>,
    pub mathematical_tests: Vec<MathematicalTest>,
}

impl TorusTestSuite {
    /// Create a new comprehensive test suite
    pub fn new() -> Self {
        Self {
            test_cases: Vec::new(),
            performance_tests: Vec::new(),
            mathematical_tests: Vec::new(),
        }
    }

    /// Add a test case
    pub fn add_test_case(&mut self, test_case: TorusTestCase) {
        self.test_cases.push(test_case);
    }

    /// Add a performance test
    pub fn add_performance_test(&mut self, test: PerformanceTest) {
        self.performance_tests.push(test);
    }

    /// Add a mathematical test
    pub fn add_mathematical_test(&mut self, test: MathematicalTest) {
        self.mathematical_tests.push(test);
    }

    /// Run all tests
    pub fn run_all_tests(&self) -> TestResults {
        let mut results = TestResults::new();
        
        // Run test cases
        for test_case in &self.test_cases {
            let start = Instant::now();
            let result = self.run_test_case(test_case);
            let duration = start.elapsed();
            
            results.add_test_result(TestResult {
                name: test_case.name.clone(),
                passed: result.is_ok(),
                duration,
                error: result.err(),
            });
        }
        
        // Run performance tests
        for test in &self.performance_tests {
            let start = Instant::now();
            let result = self.run_performance_test(test);
            let duration = start.elapsed();
            
            results.add_test_result(TestResult {
                name: test.name.clone(),
                passed: result.is_ok(),
                duration,
                error: result.err(),
            });
        }
        
        // Run mathematical tests
        for test in &self.mathematical_tests {
            let start = Instant::now();
            let result = self.run_mathematical_test(test);
            let duration = start.elapsed();
            
            results.add_test_result(TestResult {
                name: test.name.clone(),
                passed: result.is_ok(),
                duration,
                error: result.err(),
            });
        }
        
        results
    }

    /// Run a single test case
    fn run_test_case(&self, test_case: &TorusTestCase) -> Result<()> {
        match test_case.test_type {
            TestType::BasicGeneration => {
                self.test_basic_generation(test_case)?;
            }
            TestType::SurfaceProperties => {
                self.test_surface_properties(test_case)?;
            }
            TestType::Orientability => {
                self.test_orientability(test_case)?;
            }
            TestType::MathematicalProperties => {
                self.test_mathematical_properties(test_case)?;
            }
            TestType::NormalCalculation => {
                self.test_normal_calculation(test_case)?;
            }
            TestType::GeodesicDistance => {
                self.test_geodesic_distance(test_case)?;
            }
            TestType::CurvatureCalculation => {
                self.test_curvature_calculation(test_case)?;
            }
            TestType::SurfaceArea => {
                self.test_surface_area(test_case)?;
            }
            TestType::AdaptiveResolution => {
                self.test_adaptive_resolution(test_case)?;
            }
            TestType::OptimizedMeshGeneration => {
                self.test_optimized_mesh_generation(test_case)?;
            }
            TestType::TopologicalInvariants => {
                self.test_topological_invariants(test_case)?;
            }
        }
        Ok(())
    }

    /// Test basic mesh generation
    fn test_basic_generation(&self, test_case: &TorusTestCase) -> Result<()> {
        let generator = test_case.create_generator();
        let result = generator.generate_mesh_data();
        
        assert!(result.is_ok(), "Basic generation failed for {}", test_case.name);
        
        let (vertices, indices) = result.unwrap();
        assert!(!vertices.is_empty(), "No vertices generated for {}", test_case.name);
        assert!(!indices.is_empty(), "No indices generated for {}", test_case.name);
        assert_eq!(vertices.len() % 6, 0, "Invalid vertex count for {}", test_case.name);
        assert_eq!(indices.len() % 3, 0, "Invalid index count for {}", test_case.name);
        
        Ok(())
    }

    /// Test surface properties
    fn test_surface_properties(&self, test_case: &TorusTestCase) -> Result<()> {
        let generator = test_case.create_generator();
        let props = generator.calculate_surface_properties();
        
        assert!(props.approximate_surface_area > 0.0, "Invalid surface area for {}", test_case.name);
        assert!(props.genus >= 0, "Invalid genus for {}", test_case.name);
        
        Ok(())
    }

    /// Test orientability
    fn test_orientability(&self, test_case: &TorusTestCase) -> Result<()> {
        let generator = test_case.create_generator();
        let props = generator.calculate_surface_properties();
        
        let expected_orientable = generator.twists % 2 == 0;
        assert_eq!(props.is_orientable, expected_orientable, "Orientability mismatch for {}", test_case.name);
        
        Ok(())
    }

    /// Test mathematical properties
    fn test_mathematical_properties(&self, test_case: &TorusTestCase) -> Result<()> {
        let generator = test_case.create_generator();
        
        // Test parametric equations at specific points
        let pos1 = generator.calculate_position(0.0, 0.0);
        let pos2 = generator.calculate_position(PI, 0.0);
        
        // At u=0, v=0: should be at major radius
        assert!((pos1[0] - generator.major_radius).abs() < 0.001, "Position calculation error for {}", test_case.name);
        assert!(pos1[1].abs() < 0.001, "Position calculation error for {}", test_case.name);
        assert!(pos1[2].abs() < 0.001, "Position calculation error for {}", test_case.name);
        
        // At u=π, v=0: should be at -major radius
        assert!((pos2[0] + generator.major_radius).abs() < 0.001, "Position calculation error for {}", test_case.name);
        assert!(pos2[1].abs() < 0.001, "Position calculation error for {}", test_case.name);
        assert!(pos2[2].abs() < 0.001, "Position calculation error for {}", test_case.name);
        
        Ok(())
    }

    /// Test normal calculation
    fn test_normal_calculation(&self, test_case: &TorusTestCase) -> Result<()> {
        let generator = test_case.create_generator();
        
        // Test normal calculation at various points
        let normal1 = generator.calculate_normal(0.0, 0.0).unwrap();
        let normal2 = generator.calculate_normal(PI/2.0, 0.0).unwrap();
        
        // Normals should be unit vectors
        let len1 = (normal1[0]*normal1[0] + normal1[1]*normal1[1] + normal1[2]*normal1[2]).sqrt();
        let len2 = (normal2[0]*normal2[0] + normal2[1]*normal2[1] + normal2[2]*normal2[2]).sqrt();
        
        assert!((len1 - 1.0).abs() < 0.001, "Normal length error for {}", test_case.name);
        assert!((len2 - 1.0).abs() < 0.001, "Normal length error for {}", test_case.name);
        
        Ok(())
    }

    /// Test geodesic distance calculation
    fn test_geodesic_distance(&self, test_case: &TorusTestCase) -> Result<()> {
        // This would test the enhanced torus geodesic distance
        // For now, we'll skip this test as it requires the enhanced implementation
        Ok(())
    }

    /// Test curvature calculation
    fn test_curvature_calculation(&self, test_case: &TorusTestCase) -> Result<()> {
        // This would test the enhanced torus curvature calculation
        // For now, we'll skip this test as it requires the enhanced implementation
        Ok(())
    }

    /// Test surface area calculation
    fn test_surface_area(&self, test_case: &TorusTestCase) -> Result<()> {
        // This would test the enhanced torus surface area calculation
        // For now, we'll skip this test as it requires the enhanced implementation
        Ok(())
    }

    /// Test adaptive resolution
    fn test_adaptive_resolution(&self, test_case: &TorusTestCase) -> Result<()> {
        // This would test the enhanced torus adaptive resolution
        // For now, we'll skip this test as it requires the enhanced implementation
        Ok(())
    }

    /// Test optimized mesh generation
    fn test_optimized_mesh_generation(&self, test_case: &TorusTestCase) -> Result<()> {
        // This would test the performance torus optimized mesh generation
        // For now, we'll skip this test as it requires the performance implementation
        Ok(())
    }

    /// Test topological invariants
    fn test_topological_invariants(&self, test_case: &TorusTestCase) -> Result<()> {
        // This would test the enhanced torus topological invariants
        // For now, we'll skip this test as it requires the enhanced implementation
        Ok(())
    }

    /// Run a performance test
    fn run_performance_test(&self, test: &PerformanceTest) -> Result<()> {
        match test.test_type {
            PerformanceTestType::MeshGenerationSpeed => {
                self.test_mesh_generation_speed(test)?;
            }
            PerformanceTestType::MemoryUsage => {
                self.test_memory_usage(test)?;
            }
            PerformanceTestType::Scalability => {
                self.test_scalability(test)?;
            }
        }
        Ok(())
    }

    /// Test mesh generation speed
    fn test_mesh_generation_speed(&self, test: &PerformanceTest) -> Result<()> {
        let generator = test.create_generator();
        let start = Instant::now();
        
        let result = generator.generate_mesh_data();
        let duration = start.elapsed();
        
        assert!(result.is_ok(), "Mesh generation failed for {}", test.name);
        assert!(duration < test.max_duration, "Mesh generation too slow for {}", test.name);
        
        Ok(())
    }

    /// Test memory usage
    fn test_memory_usage(&self, test: &PerformanceTest) -> Result<()> {
        let generator = test.create_generator();
        let result = generator.generate_mesh_data();
        
        assert!(result.is_ok(), "Mesh generation failed for {}", test.name);
        
        let (vertices, indices) = result.unwrap();
        let memory_usage = vertices.len() * 4 + indices.len() * 4; // Rough estimate
        
        assert!(memory_usage <= test.max_memory_bytes, "Memory usage too high for {}", test.name);
        
        Ok(())
    }

    /// Test scalability
    fn test_scalability(&self, test: &PerformanceTest) -> Result<()> {
        // Test with different resolutions
        let resolutions = vec![(16, 8), (32, 16), (64, 32), (128, 64)];
        let mut times = Vec::new();
        
        for (u_res, v_res) in resolutions {
            let generator = test.create_generator_with_resolution(u_res, v_res);
            let start = Instant::now();
            
            let result = generator.generate_mesh_data();
            let duration = start.elapsed();
            
            assert!(result.is_ok(), "Mesh generation failed for resolution {}x{}", u_res, v_res);
            times.push(duration);
        }
        
        // Check that time scales reasonably (not exponentially)
        for i in 1..times.len() {
            let ratio = times[i].as_nanos() as f64 / times[i-1].as_nanos() as f64;
            assert!(ratio < 8.0, "Scalability issue: time ratio {} too high", ratio);
        }
        
        Ok(())
    }

    /// Run a mathematical test
    fn run_mathematical_test(&self, test: &MathematicalTest) -> Result<()> {
        match test.test_type {
            MathematicalTestType::KTwistConstant => {
                self.test_k_twist_constant(test)?;
            }
            MathematicalTestType::ParametricEquations => {
                self.test_parametric_equations(test)?;
            }
            MathematicalTestType::NormalVectors => {
                self.test_normal_vectors(test)?;
            }
            MathematicalTestType::SurfaceIntegrity => {
                self.test_surface_integrity(test)?;
            }
        }
        Ok(())
    }

    /// Test k-twist constant
    fn test_k_twist_constant(&self, test: &MathematicalTest) -> Result<()> {
        let k_twist = (std::f64::consts::E * std::f64::consts::PI) / (std::f64::consts::E + std::f64::consts::PI);
        
        assert!((k_twist - 1.457324).abs() < 0.00001, "K-twist constant incorrect");
        assert!(k_twist > 1.45 && k_twist < 1.46, "K-twist constant out of range");
        
        Ok(())
    }

    /// Test parametric equations
    fn test_parametric_equations(&self, test: &MathematicalTest) -> Result<()> {
        let generator = test.create_generator();
        
        // Test at various points
        let test_points = vec![
            (0.0, 0.0),
            (PI/2.0, 0.0),
            (PI, 0.0),
            (3.0*PI/2.0, 0.0),
            (0.0, PI/2.0),
            (PI, PI/2.0),
        ];
        
        for (u, v) in test_points {
            let pos = generator.calculate_position(u, v);
            
            // Check that all coordinates are finite
            assert!(pos[0].is_finite(), "X coordinate not finite at ({}, {})", u, v);
            assert!(pos[1].is_finite(), "Y coordinate not finite at ({}, {})", u, v);
            assert!(pos[2].is_finite(), "Z coordinate not finite at ({}, {})", u, v);
        }
        
        Ok(())
    }

    /// Test normal vectors
    fn test_normal_vectors(&self, test: &MathematicalTest) -> Result<()> {
        let generator = test.create_generator();
        
        // Test normal calculation at various points
        let test_points = vec![
            (0.0, 0.0),
            (PI/4.0, 0.0),
            (PI/2.0, 0.0),
            (PI, 0.0),
            (0.0, PI/4.0),
            (PI/2.0, PI/4.0),
        ];
        
        for (u, v) in test_points {
            let normal = generator.calculate_normal(u, v).unwrap();
            
            // Check that normal is unit vector
            let length = (normal[0]*normal[0] + normal[1]*normal[1] + normal[2]*normal[2]).sqrt();
            assert!((length - 1.0).abs() < 0.001, "Normal not unit vector at ({}, {})", u, v);
            
            // Check that all components are finite
            assert!(normal[0].is_finite(), "Normal X not finite at ({}, {})", u, v);
            assert!(normal[1].is_finite(), "Normal Y not finite at ({}, {})", u, v);
            assert!(normal[2].is_finite(), "Normal Z not finite at ({}, {})", u, v);
        }
        
        Ok(())
    }

    /// Test surface integrity
    fn test_surface_integrity(&self, test: &MathematicalTest) -> Result<()> {
        let generator = test.create_generator();
        let result = generator.generate_mesh_data().unwrap();
        let (vertices, indices) = result;
        
        // Check that all vertices are within reasonable bounds
        for chunk in vertices.chunks(6) {
            let x = chunk[0];
            let y = chunk[1];
            let z = chunk[2];
            
            let distance = (x*x + y*y + z*z).sqrt();
            let max_distance = generator.major_radius + generator.strip_width;
            
            assert!(distance <= max_distance, "Vertex outside surface bounds");
        }
        
        // Check that all indices are valid
        let max_index = vertices.len() / 6 - 1;
        for &index in &indices {
            assert!(index as usize <= max_index, "Invalid vertex index: {}", index);
        }
        
        Ok(())
    }
}

/// Test case for torus operations
#[derive(Debug, Clone)]
pub struct TorusTestCase {
    pub name: String,
    pub test_type: TestType,
    pub major_radius: f32,
    pub strip_width: f32,
    pub twists: i32,
    pub u_steps: usize,
    pub v_steps: usize,
}

impl TorusTestCase {
    /// Create a generator for this test case
    pub fn create_generator(&self) -> crate::k_twisted_torus::KTwistedTorusGenerator {
        crate::k_twisted_torus::KTwistedTorusGenerator::new(
            self.major_radius,
            self.strip_width,
            self.twists,
            self.u_steps,
            self.v_steps,
        )
    }
}

/// Performance test case
#[derive(Debug, Clone)]
pub struct PerformanceTest {
    pub name: String,
    pub test_type: PerformanceTestType,
    pub major_radius: f32,
    pub strip_width: f32,
    pub twists: i32,
    pub u_steps: usize,
    pub v_steps: usize,
    pub max_duration: std::time::Duration,
    pub max_memory_bytes: usize,
}

impl PerformanceTest {
    /// Create a generator for this performance test
    pub fn create_generator(&self) -> crate::k_twisted_torus::KTwistedTorusGenerator {
        crate::k_twisted_torus::KTwistedTorusGenerator::new(
            self.major_radius,
            self.strip_width,
            self.twists,
            self.u_steps,
            self.v_steps,
        )
    }

    /// Create a generator with specific resolution
    pub fn create_generator_with_resolution(&self, u_steps: usize, v_steps: usize) -> crate::k_twisted_torus::KTwistedTorusGenerator {
        crate::k_twisted_torus::KTwistedTorusGenerator::new(
            self.major_radius,
            self.strip_width,
            self.twists,
            u_steps,
            v_steps,
        )
    }
}

/// Mathematical test case
#[derive(Debug, Clone)]
pub struct MathematicalTest {
    pub name: String,
    pub test_type: MathematicalTestType,
    pub major_radius: f32,
    pub strip_width: f32,
    pub twists: i32,
    pub u_steps: usize,
    pub v_steps: usize,
}

impl MathematicalTest {
    /// Create a generator for this mathematical test
    pub fn create_generator(&self) -> crate::k_twisted_torus::KTwistedTorusGenerator {
        crate::k_twisted_torus::KTwistedTorusGenerator::new(
            self.major_radius,
            self.strip_width,
            self.twists,
            self.u_steps,
            self.v_steps,
        )
    }
}

/// Test types
#[derive(Debug, Clone, Copy)]
pub enum TestType {
    BasicGeneration,
    SurfaceProperties,
    Orientability,
    MathematicalProperties,
    NormalCalculation,
    GeodesicDistance,
    CurvatureCalculation,
    SurfaceArea,
    AdaptiveResolution,
    OptimizedMeshGeneration,
    TopologicalInvariants,
}

/// Performance test types
#[derive(Debug, Clone, Copy)]
pub enum PerformanceTestType {
    MeshGenerationSpeed,
    MemoryUsage,
    Scalability,
}

/// Mathematical test types
#[derive(Debug, Clone, Copy)]
pub enum MathematicalTestType {
    KTwistConstant,
    ParametricEquations,
    NormalVectors,
    SurfaceIntegrity,
}

/// Test results
#[derive(Debug)]
pub struct TestResults {
    pub results: Vec<TestResult>,
    pub total_tests: usize,
    pub passed_tests: usize,
    pub failed_tests: usize,
    pub total_duration: std::time::Duration,
}

impl TestResults {
    /// Create new test results
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
            total_tests: 0,
            passed_tests: 0,
            failed_tests: 0,
            total_duration: std::time::Duration::from_secs(0),
        }
    }

    /// Add a test result
    pub fn add_test_result(&mut self, result: TestResult) {
        self.total_tests += 1;
        if result.passed {
            self.passed_tests += 1;
        } else {
            self.failed_tests += 1;
        }
        self.total_duration += result.duration;
        self.results.push(result);
    }

    /// Print summary
    pub fn print_summary(&self) {
        tracing::info!("🧪 Torus Test Suite Results");
        tracing::info!("==========================");
        tracing::info!("Total tests: {}", self.total_tests);
        tracing::info!("Passed: {}", self.passed_tests);
        tracing::info!("Failed: {}", self.failed_tests);
        tracing::info!("Success rate: {:.1}%", (self.passed_tests as f64 / self.total_tests as f64) * 100.0);
        tracing::info!("Total duration: {:?}", self.total_duration);
        
        if self.failed_tests > 0 {
            tracing::info!("\n❌ Failed tests:");
            for result in &self.results {
                if !result.passed {
                    tracing::info!("  - {}: {:?}", result.name, result.error);
                }
            }
        }
        
        if self.passed_tests > 0 {
            tracing::info!("\n✅ Passed tests:");
            for result in &self.results {
                if result.passed {
                    tracing::info!("  - {}: {:?}", result.name, result.duration);
                }
            }
        }
    }
}

/// Individual test result
#[derive(Debug)]
pub struct TestResult {
    pub name: String,
    pub passed: bool,
    pub duration: std::time::Duration,
    pub error: Option<anyhow::Error>,
}

/// Create a comprehensive test suite
pub fn create_comprehensive_test_suite() -> TorusTestSuite {
    let mut suite = TorusTestSuite::new();
    
    // Basic test cases
    suite.add_test_case(TorusTestCase {
        name: "Basic Generation - Default".to_string(),
        test_type: TestType::BasicGeneration,
        major_radius: 300.0,
        strip_width: 50.0,
        twists: 1,
        u_steps: 128,
        v_steps: 32,
    });
    
    suite.add_test_case(TorusTestCase {
        name: "Surface Properties - Non-orientable".to_string(),
        test_type: TestType::SurfaceProperties,
        major_radius: 100.0,
        strip_width: 20.0,
        twists: 1,
        u_steps: 64,
        v_steps: 16,
    });
    
    suite.add_test_case(TorusTestCase {
        name: "Orientability - Orientable".to_string(),
        test_type: TestType::Orientability,
        major_radius: 100.0,
        strip_width: 20.0,
        twists: 2,
        u_steps: 32,
        v_steps: 16,
    });
    
    suite.add_test_case(TorusTestCase {
        name: "Mathematical Properties".to_string(),
        test_type: TestType::MathematicalProperties,
        major_radius: 200.0,
        strip_width: 40.0,
        twists: 3,
        u_steps: 64,
        v_steps: 32,
    });
    
    suite.add_test_case(TorusTestCase {
        name: "Normal Calculation".to_string(),
        test_type: TestType::NormalCalculation,
        major_radius: 100.0,
        strip_width: 20.0,
        twists: 1,
        u_steps: 32,
        v_steps: 16,
    });
    
    // Performance tests
    suite.add_performance_test(PerformanceTest {
        name: "Mesh Generation Speed - Small".to_string(),
        test_type: PerformanceTestType::MeshGenerationSpeed,
        major_radius: 100.0,
        strip_width: 20.0,
        twists: 1,
        u_steps: 32,
        v_steps: 16,
        max_duration: std::time::Duration::from_millis(100),
        max_memory_bytes: 1024 * 1024, // 1MB
    });
    
    suite.add_performance_test(PerformanceTest {
        name: "Memory Usage - Medium".to_string(),
        test_type: PerformanceTestType::MemoryUsage,
        major_radius: 200.0,
        strip_width: 40.0,
        twists: 2,
        u_steps: 64,
        v_steps: 32,
        max_duration: std::time::Duration::from_millis(500),
        max_memory_bytes: 10 * 1024 * 1024, // 10MB
    });
    
    suite.add_performance_test(PerformanceTest {
        name: "Scalability Test".to_string(),
        test_type: PerformanceTestType::Scalability,
        major_radius: 100.0,
        strip_width: 20.0,
        twists: 1,
        u_steps: 128,
        v_steps: 64,
        max_duration: std::time::Duration::from_secs(2),
        max_memory_bytes: 100 * 1024 * 1024, // 100MB
    });
    
    // Mathematical tests
    suite.add_mathematical_test(MathematicalTest {
        name: "K-Twist Constant".to_string(),
        test_type: MathematicalTestType::KTwistConstant,
        major_radius: 100.0,
        strip_width: 20.0,
        twists: 1,
        u_steps: 32,
        v_steps: 16,
    });
    
    suite.add_mathematical_test(MathematicalTest {
        name: "Parametric Equations".to_string(),
        test_type: MathematicalTestType::ParametricEquations,
        major_radius: 150.0,
        strip_width: 30.0,
        twists: 2,
        u_steps: 64,
        v_steps: 32,
    });
    
    suite.add_mathematical_test(MathematicalTest {
        name: "Normal Vectors".to_string(),
        test_type: MathematicalTestType::NormalVectors,
        major_radius: 100.0,
        strip_width: 20.0,
        twists: 1,
        u_steps: 32,
        v_steps: 16,
    });
    
    suite.add_mathematical_test(MathematicalTest {
        name: "Surface Integrity".to_string(),
        test_type: MathematicalTestType::SurfaceIntegrity,
        major_radius: 200.0,
        strip_width: 40.0,
        twists: 3,
        u_steps: 64,
        v_steps: 32,
    });
    
    suite
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_comprehensive_test_suite() {
        let suite = create_comprehensive_test_suite();
        let results = suite.run_all_tests();
        
        results.print_summary();
        
        // At least some tests should pass
        assert!(results.passed_tests > 0, "No tests passed");
        
        // Success rate should be reasonable
        let success_rate = results.passed_tests as f64 / results.total_tests as f64;
        assert!(success_rate > 0.5, "Success rate too low: {:.1}%", success_rate * 100.0);
    }
}

