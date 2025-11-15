//! Euler Problems Level 50 - Mathematical Intelligence Test Suite  
//! Integrated with niodoo_real_integrated pipeline for autonomous gating validation

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use tracing::{info, warn};

/// Level 50 Euler Problems for testing mathematical reasoning with full TCS pipeline
pub fn euler_level50_problems() -> Vec<String> {
    vec![
        format!("Solve {}. While solving, compute and cite Betti numbers β0/β1 for your solution graph and highlight any gaps in the proof chain.",
        "Implement an optimized segmented sieve to find the sum of all twin primes below 1 million. \
        Include mathematical proof, complexity analysis, and handle edge cases. \
        Twin primes are pairs differing by 2 (like 11,13)."),
        format!("Solve {}. While solving, compute and cite Betti numbers β0/β1 for your solution graph and highlight any gaps in the proof chain.",
        "Calculate ways to tile a 20×3 rectangle using 2×1 dominoes. \
        Derive recurrence relation, implement with dynamic programming, \
        prove via mathematical induction, optimize space to O(1)."),
        format!("Solve {}. While solving, compute and cite Betti numbers β0/β1 for your solution graph and highlight any gaps in the proof chain.",
        "Find the 10,000th convergent of π's continued fraction. \
        Use arbitrary precision arithmetic, prove convergence properties, \
        analyze error bounds, optimize for speed and memory."),
        format!("Solve {}. While solving, compute and cite Betti numbers β0/β1 for your solution graph and highlight any gaps in the proof chain.",
        "Find all Pythagorean quadruples (a,b,c,d) where a²+b²+c²=d² and a+b+c+d=1000. \
        Use parametric representation, optimize with number theory, \
        prove search completeness, analyze emerging patterns."),
        format!("Solve {}. While solving, compute and cite Betti numbers β0/β1 for your solution graph and highlight any gaps in the proof chain.",
        "Implement RSA-2048 with Miller-Rabin primality testing and Montgomery modular reduction. \
        Include mathematical security proofs, performance analysis, \
        and comparison with standard implementations."),
        format!("Solve {}. While solving, compute and cite Betti numbers β0/β1 for your solution graph and highlight any gaps in the proof chain.",
        "Calculate Hamiltonian paths in 15×15 grid with 20 strategically placed forbidden edges. \
        Use graph theory optimization, prove algorithm completeness, \
        apply intelligent pruning based on mathematical bounds."),
        format!("Solve {}. While solving, compute and cite Betti numbers β0/β1 for your solution graph and highlight any gaps in the proof chain.",
        "Find smallest positive integer divisible by all numbers 1 to 50. \
        Use prime factorization, analyze Landau function growth rate, \
        prove minimality, compare with theoretical bounds."),
        format!("Solve {}. While solving, compute and cite Betti numbers β0/β1 for your solution graph and highlight any gaps in the proof chain.",
        "Calculate 2^(2^16) mod (10^9+7) using advanced modular arithmetic. \
        Apply Euler's theorem, implement iterated exponentiation, \
        prove correctness, optimize using number theory."),
        format!("Solve {}. While solving, compute and cite Betti numbers β0/β1 for your solution graph and highlight any gaps in the proof chain.",
        "Calculate exact sum of digits of 10,000! using arbitrary precision. \
        Implement optimized factorial calculation, handle large numbers efficiently, \
        analyze mathematical properties of digit sums."),
        format!("Solve {}. While solving, compute and cite Betti numbers β0/β1 for your solution graph and highlight any gaps in the proof chain.",
        "Find millionth lexicographic permutation of 'ABCDEFGHIJ' without generation. \
        Use factorial number system, implement rank-to-permutation, \
        prove correctness, optimize time and space complexity."),
    ]
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EulerTestConfig {
    pub problems_to_run: usize,
    pub timeout_secs: u64,
    pub expected_high_quality_rate: f32, // Minimum rate for Memory Gate (≥8/10)
    pub max_learning_gate_rate: f32,     // Maximum acceptable failure rate (≤5/10)
}

impl Default for EulerTestConfig {
    fn default() -> Self {
        Self {
            problems_to_run: 10,
            timeout_secs: 300,               // 5 minutes per problem
            expected_high_quality_rate: 0.7, // Expect 70% high quality for mathematical problems
            max_learning_gate_rate: 0.3,     // Accept max 30% failures (system should learn)
        }
    }
}

#[derive(Debug, Serialize)]
pub struct EulerTestResult {
    pub problem_id: usize,
    pub problem: String,
    pub response: String,
    pub quality_score: f32,
    pub gating_path: String,
    pub mathematical_indicators: MathematicalIndicators,
    pub topology_signature: TopologySignature,
    pub pad_emotional_state: PADEmotionalState,
    pub processing_time_ms: u64,
    pub memory_retrieval_count: usize,
    pub breakthrough_detected: bool,
    pub novel_topology: bool,
    pub extreme_emotion: bool,
    pub topocot: Option<EulerTopoCotSummary>,
}

#[derive(Debug, Serialize)]
pub struct MathematicalIndicators {
    pub contains_code: bool,
    pub contains_proof: bool,
    pub contains_algorithm: bool,
    pub contains_optimization: bool,
    pub mathematical_depth: u8,    // 0-10 score
    pub code_quality: u8,          // 0-10 score
    pub problem_understanding: u8, // 0-10 score
}

#[derive(Debug, Serialize)]
pub struct TopologySignature {
    pub betti_numbers: Vec<usize>,
    pub knot_complexity: f32,
    pub spectral_gap: f32,
    pub persistence_entropy: f32,
}

#[derive(Debug, Serialize)]
pub struct PADEmotionalState {
    pub pleasure: f32,
    pub arousal: f32,
    pub dominance: f32,
    pub entropy: f32,
    pub surface_position: [f32; 3],
}

#[derive(Debug, Serialize)]
pub struct EulerTopoCotSummary {
    pub score_overall: f64,
    pub score_completeness: f64,
    pub score_consistency: f64,
    pub score_actionability: f64,
    pub issues: Vec<String>,
    pub raw_json: Option<String>,
    pub thinking_depth: f64,
    pub pivot_score: f64,
    pub reflection_summary: Option<String>,
    pub plan_summary: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct EulerTestSuiteResults {
    pub test_id: String,
    pub timestamp: String,
    pub config: EulerTestConfig,
    pub summary: TestSummary,
    pub results: Vec<EulerTestResult>,
    pub gating_analysis: GatingAnalysis,
    pub intelligence_assessment: IntelligenceAssessment,
}

#[derive(Debug, Serialize)]
pub struct TestSummary {
    pub total_problems: usize,
    pub completed_problems: usize,
    pub average_quality: f32,
    pub average_math_depth: f32,
    pub total_duration_secs: f64,
}

#[derive(Debug, Serialize)]
pub struct GatingAnalysis {
    pub learning_gate_count: usize, // Score ≤5 - routed to Gemini for correction
    pub indifferent_count: usize,   // Score 6-7 - discarded
    pub memory_gate_count: usize,   // Score ≥8 - candidates for Golden Memory
    pub novel_topology_count: usize, // Unique Betti signatures
    pub extreme_emotion_count: usize, // PAD spikes ≥0.4
    pub golden_memory_qualified: usize, // Novel OR Extreme + high quality
}

#[derive(Debug, Serialize)]
pub struct IntelligenceAssessment {
    pub mathematical_reasoning_grade: String, // A/B/C/D/F based on performance
    pub strengths: Vec<String>,
    pub weaknesses: Vec<String>,
    pub improvement_recommendations: Vec<String>,
    pub autonomous_learning_effectiveness: f32, // How well Learning Gate works
    pub memory_curation_effectiveness: f32,     // How well Memory Gate works
    pub system_intelligence_level: String,      // "Novice"/"Intermediate"/"Advanced"/"Expert"
}

impl EulerTestSuiteResults {
    pub fn analyze_intelligence(&mut self) {
        let high_quality_rate = self.summary.average_quality / 10.0;
        let math_depth_rate = self.summary.average_math_depth / 10.0;
        let memory_gate_rate =
            self.gating_analysis.memory_gate_count as f32 / self.summary.total_problems as f32;

        // Determine intelligence grade
        let grade = match (high_quality_rate * 10.0) as u8 {
            9..=10 => "A+ Expert Mathematical Reasoning",
            8..=8 => "A Advanced Mathematical Understanding",
            7..=7 => "B+ Solid Mathematical Capability",
            6..=6 => "B Adequate Mathematical Skills",
            5..=5 => "C+ Basic Mathematical Understanding",
            4..=4 => "C Limited Mathematical Reasoning",
            3..=3 => "D+ Weak Mathematical Capability",
            2..=2 => "D Poor Mathematical Understanding",
            _ => "F Mathematical Reasoning Failure",
        };

        let system_level = if memory_gate_rate >= 0.8 {
            "Expert - Consistently produces novel mathematical insights"
        } else if memory_gate_rate >= 0.6 {
            "Advanced - Often produces high-quality mathematical solutions"
        } else if memory_gate_rate >= 0.4 {
            "Intermediate - Sometimes produces good mathematical reasoning"
        } else {
            "Novice - Struggles with mathematical problem solving"
        };

        self.intelligence_assessment = IntelligenceAssessment {
            mathematical_reasoning_grade: grade.to_string(),
            strengths: self.identify_strengths(),
            weaknesses: self.identify_weaknesses(),
            improvement_recommendations: self.generate_recommendations(),
            autonomous_learning_effectiveness: (self.gating_analysis.learning_gate_count as f32
                / self.summary.total_problems as f32)
                .min(1.0),
            memory_curation_effectiveness: memory_gate_rate,
            system_intelligence_level: system_level.to_string(),
        };
    }

    fn identify_strengths(&self) -> Vec<String> {
        let mut strengths = Vec::new();

        let avg_math = self.summary.average_math_depth;
        if avg_math >= 7.0 {
            strengths.push("Strong mathematical reasoning".to_string());
        }
        if avg_math >= 5.0 {
            strengths.push("Adequate problem comprehension".to_string());
        }

        let novel_rate =
            self.gating_analysis.novel_topology_count as f32 / self.summary.total_problems as f32;
        if novel_rate >= 0.6 {
            strengths.push("Generates novel solution approaches".to_string());
        }

        if self.gating_analysis.memory_gate_count >= 7 {
            strengths.push("Consistently produces high-quality solutions".to_string());
        }

        if strengths.is_empty() {
            strengths.push("System demonstrates autonomous gating behavior".to_string());
        }

        strengths
    }

    fn identify_weaknesses(&self) -> Vec<String> {
        let mut weaknesses = Vec::new();

        if self.summary.average_quality < 5.0 {
            weaknesses.push("Low overall solution quality".to_string());
        }
        if self.summary.average_math_depth < 4.0 {
            weaknesses.push("Insufficient mathematical depth".to_string());
        }
        if self.gating_analysis.learning_gate_count > 6 {
            weaknesses.push("High failure rate on mathematical problems".to_string());
        }
        if self.gating_analysis.golden_memory_qualified < 3 {
            weaknesses.push("Few solutions qualify for memory retention".to_string());
        }

        weaknesses
    }

    fn generate_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        if self.summary.average_math_depth < 6.0 {
            recommendations
                .push("Enhance mathematical reasoning prompts with more context".to_string());
        }
        if self.gating_analysis.learning_gate_count > 5 {
            recommendations
                .push("Improve base model training on mathematical concepts".to_string());
        }
        if self.gating_analysis.memory_gate_count < 3 {
            recommendations
                .push("Adjust quality thresholds or improve solution generation".to_string());
        }

        recommendations
            .push("Continue autonomous learning through Gemini failure corrections".to_string());
        recommendations
            .push("Monitor Golden Memory accumulation for mathematical patterns".to_string());

        recommendations
    }

    pub fn print_intelligence_report(&self) {
        println!("\n🎓 INTELLIGENCE ASSESSMENT REPORT");
        println!("=================================");
        println!(
            "📊 Grade: {}",
            self.intelligence_assessment.mathematical_reasoning_grade
        );
        println!(
            "🎯 System Level: {}",
            self.intelligence_assessment.system_intelligence_level
        );
        println!("\n💪 Strengths:");
        for strength in &self.intelligence_assessment.strengths {
            println!("  ✅ {}", strength);
        }
        println!("\n⚠️  Areas for Improvement:");
        for weakness in &self.intelligence_assessment.weaknesses {
            println!("  🔄 {}", weakness);
        }
        println!("\n🎯 Recommendations:");
        for rec in &self.intelligence_assessment.improvement_recommendations {
            println!("  💡 {}", rec);
        }

        println!("\n🤖 Autonomous Learning Analysis:");
        println!(
            "  Learning Effectiveness: {:.1}% (failure corrections)",
            self.intelligence_assessment
                .autonomous_learning_effectiveness
                * 100.0
        );
        println!(
            "  Memory Curation Effectiveness: {:.1}% (golden memory rate)",
            self.intelligence_assessment.memory_curation_effectiveness * 100.0
        );
    }
}
