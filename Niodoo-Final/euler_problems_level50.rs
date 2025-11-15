// Euler Problems Level 50 - Mathematical Reasoning Test Suite
// Created 2025-11-11 for Full NIODOO System Intelligence Testing
// These problems require advanced mathematical reasoning, optimization, and multi-step thinking

use std::collections::HashMap;

/// Level 50 Euler Problems - Designed to test true mathematical intelligence
pub fn euler_level50_problems() -> Vec<String> {
    vec![
        // Problem 1: Advanced Prime Theory + Optimization
        "Implement an optimized segmented sieve algorithm to find the sum of all twin primes below 10 million. \
        Twin primes are pairs of primes that differ by 2 (like 11,13). Your solution should: \
        (1) Use memory-efficient segmentation for large ranges, \
        (2) Include mathematical proof of why your algorithm correctly identifies twin primes, \
        (3) Analyze time complexity and explain optimization choices, \
        (4) Handle edge cases near boundaries. \
        Provide both the implementation and mathematical reasoning.".to_string(),

        // Problem 2: Dynamic Programming + Number Theory
        "Calculate the number of ways to tile a 50×3 rectangle using 2×1 dominoes. \
        This requires: (1) Deriving the recurrence relation mathematically, \
        (2) Implementing dynamic programming with state compression, \
        (3) Handling large numbers (use modular arithmetic mod 10^9+7), \
        (4) Proving correctness via mathematical induction, \
        (5) Optimizing space complexity to O(1). \
        Explain the mathematical foundation and provide efficient implementation.".to_string(),

        // Problem 3: Continued Fractions + Arbitrary Precision  
        "Find the 100,000th convergent of the continued fraction representation of π. \
        Your solution must: (1) Implement arbitrary precision arithmetic for large numerators/denominators, \
        (2) Use the optimal continued fraction algorithm with convergent calculations, \
        (3) Prove convergence properties and error bounds, \
        (4) Optimize for both speed and memory usage, \
        (5) Handle numerical stability near machine precision limits. \
        Include mathematical derivation and implementation.".to_string(),

        // Problem 4: Advanced Combinatorics + Optimization
        "Find all Pythagorean quadruples (a,b,c,d) where a² + b² + c² = d² and a + b + c + d = 10,000. \
        Requirements: (1) Derive parametric representation of quadruples mathematically, \
        (2) Implement efficient search using number theory constraints, \
        (3) Prove completeness of your search method, \
        (4) Optimize using GCD properties and primitive solutions, \
        (5) Analyze why certain patterns emerge. \
        Provide mathematical foundation, proof, and optimized code.".to_string(),

        // Problem 5: Cryptographic Mathematics + Implementation
        "Implement a complete RSA-4096 system with: (1) Probabilistic primality testing using Miller-Rabin with optimal iteration count, \
        (2) Efficient modular exponentiation using binary method with Montgomery reduction, \
        (3) Carmichael function calculation for private key optimization, \
        (4) Padding scheme (OAEP) for security, \
        (5) Performance analysis comparing with known implementations. \
        Include mathematical proofs of correctness and security analysis.".to_string(),

        // Problem 6: Advanced Graph Theory + Algorithms  
        "Calculate the number of Hamiltonian paths in a 20×20 grid graph with exactly 50 forbidden edges placed strategically to maximize computational difficulty. \
        Your solution should: (1) Use backtracking with intelligent pruning based on graph theory, \
        (2) Implement dynamic programming on subsets (if applicable), \
        (3) Apply mathematical bounds to reduce search space, \
        (4) Prove your algorithm's completeness, \
        (5) Analyze complexity and compare with brute force. \
        Provide graph-theoretic analysis and optimized implementation.".to_string(),

        // Problem 7: Advanced Number Theory + Prime Analysis
        "Find the smallest positive integer that is divisible by all integers from 1 to 100, \
        then factorize it completely and analyze its mathematical properties. \
        Requirements: (1) Use prime factorization and LCM theory optimally, \
        (2) Implement efficient exponentiation for verification, \
        (3) Analyze the growth rate of such numbers (Landau function), \
        (4) Prove minimality using mathematical arguments, \
        (5) Compare with known theoretical bounds. \
        Include number-theoretic proofs and efficient implementation.".to_string(),

        // Problem 8: Modular Arithmetic + Advanced Algorithms
        "Calculate 2^(2^20) mod (10^9 + 7) using optimal modular exponentiation. \
        This requires: (1) Understanding of Euler's theorem and Carmichael function, \
        (2) Implementing iterated modular exponentiation correctly, \
        (3) Handling extremely large exponents via mathematical reduction, \
        (4) Proving correctness using modular arithmetic theory, \
        (5) Optimizing using Chinese Remainder Theorem if beneficial. \
        Provide mathematical derivation and implementation with complexity analysis.".to_string(),

        // Problem 9: Combinatorial Mathematics + Big Integer Arithmetic
        "Calculate the exact sum of digits of 50,000! (50 thousand factorial). \
        Your solution should: (1) Implement arbitrary precision multiplication algorithm (Karatsuba or FFT), \
        (2) Use efficient factorial calculation with optimization, \
        (3) Handle memory management for extremely large numbers, \
        (4) Prove correctness of your big integer implementation, \
        (5) Analyze the mathematical properties of factorial digit sums. \
        Include algorithmic complexity analysis and mathematical reasoning.".to_string(),

        // Problem 10: Advanced Permutation Theory + Algorithms
        "Find the 10-millionth lexicographic permutation of the string 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789' \
        without generating all previous permutations. Requirements: \
        (1) Use mathematical direct calculation via factorial number system, \
        (2) Implement efficient rank-to-permutation algorithm, \
        (3) Handle character ordering and uniqueness correctly, \
        (4) Prove your algorithm generates the correct lexicographic ordering, \
        (5) Optimize for both time and space complexity. \
        Include combinatorial mathematics explanation and efficient implementation.".to_string(),
    ]
}

/// Test runner for Euler Level 50 problems
pub async fn run_euler_test_suite(
    system_runner: impl Fn(&str) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<EulerTestResult, Box<dyn std::error::Error>>> + Send>>,
) -> EulerTestSuiteResults {
    let problems = euler_level50_problems();
    let mut results = Vec::new();
    let mut learning_gate_count = 0;
    let mut memory_gate_count = 0;
    let mut indifferent_count = 0;
    
    for (idx, problem) in problems.iter().enumerate() {
        println!("\n🧮 Euler Problem {} of {}", idx + 1, problems.len());
        println!("=====================================");
        println!("{}", &problem[..problem.len().min(200)]); // Show first 200 chars
        
        let start_time = std::time::Instant::now();
        
        match system_runner(problem).await {
            Ok(result) => {
                let duration = start_time.elapsed();
                
                // Analyze gating behavior
                match result.quality_score {
                    0..=5 => {
                        learning_gate_count += 1;
                        println!("❌ LEARNING GATE: Score {}/10 - System learning from failure", result.quality_score);
                    }
                    6..=7 => {
                        indifferent_count += 1;  
                        println!("😐 INDIFFERENT: Score {}/10 - Mediocre response discarded", result.quality_score);
                    }
                    8..=10 => {
                        memory_gate_count += 1;
                        println!("✅ MEMORY GATE: Score {}/10 - Golden memory candidate", result.quality_score);
                    }
                    _ => {}
                }
                
                // Validate mathematical correctness (basic heuristics)
                let math_quality = analyze_mathematical_content(&result.response);
                
                println!("⏱️  Duration: {:?}", duration);
                println!("🧮 Math Quality: {}/10", math_quality);
                println!("📝 Response: {}...", &result.response[..result.response.len().min(150)]);
                
                results.push(EulerProblemResult {
                    problem_id: idx + 1,
                    problem: problem.clone(),
                    response: result.response,
                    quality_score: result.quality_score,
                    mathematical_correctness: math_quality,
                    duration,
                    gating_path: match result.quality_score {
                        0..=5 => "Learning Gate".to_string(),
                        6..=7 => "Indifferent Path".to_string(), 
                        8..=10 => "Memory Gate".to_string(),
                        _ => "Unknown".to_string(),
                    },
                });
            }
            Err(e) => {
                println!("💥 ERROR: {}", e);
                results.push(EulerProblemResult {
                    problem_id: idx + 1,
                    problem: problem.clone(),
                    response: format!("ERROR: {}", e),
                    quality_score: 0,
                    mathematical_correctness: 0,
                    duration: start_time.elapsed(),
                    gating_path: "Error".to_string(),
                });
            }
        }
    }
    
    EulerTestSuiteResults {
        problems_attempted: problems.len(),
        learning_gate_count,
        memory_gate_count, 
        indifferent_count,
        results,
    }
}

/// Simple mathematical content analysis
fn analyze_mathematical_content(response: &str) -> u8 {
    let mut score = 0u8;
    
    // Check for mathematical keywords/concepts
    let math_indicators = [
        "algorithm", "complexity", "proof", "theorem", "lemma",
        "O(", "optimization", "efficiency", "mathematical", "derive",
        "equation", "formula", "calculate", "implement", "analysis"
    ];
    
    let code_indicators = [
        "fn ", "def ", "class ", "impl", "struct", "function",
        "for ", "while", "if ", "return", "let ", "const"
    ];
    
    let advanced_indicators = [
        "modular", "prime", "factorial", "permutation", "combinatorial",
        "recursive", "dynamic programming", "graph theory", "number theory"
    ];
    
    for indicator in math_indicators {
        if response.to_lowercase().contains(indicator) {
            score = score.saturating_add(1);
        }
    }
    
    for indicator in code_indicators {
        if response.contains(indicator) {
            score = score.saturating_add(1);
        }
    }
    
    for indicator in advanced_indicators {
        if response.to_lowercase().contains(indicator) {
            score = score.saturating_add(2); // Higher weight for advanced concepts
        }
    }
    
    // Length-based quality (longer responses often indicate more thorough reasoning)
    if response.len() > 1000 { score = score.saturating_add(2); }
    else if response.len() > 500 { score = score.saturating_add(1); }
    
    score.min(10)
}

#[derive(Debug, Clone)]
pub struct EulerTestResult {
    pub response: String,
    pub quality_score: u8,
}

#[derive(Debug, Clone)]
pub struct EulerProblemResult {
    pub problem_id: usize,
    pub problem: String,
    pub response: String,
    pub quality_score: u8,
    pub mathematical_correctness: u8,
    pub duration: std::time::Duration,
    pub gating_path: String,
}

#[derive(Debug)]
pub struct EulerTestSuiteResults {
    pub problems_attempted: usize,
    pub learning_gate_count: usize,
    pub memory_gate_count: usize,
    pub indifferent_count: usize,
    pub results: Vec<EulerProblemResult>,
}

impl EulerTestSuiteResults {
    pub fn print_summary(&self) {
        println!("\n🧮 EULER TEST SUITE RESULTS");
        println!("============================");
        println!("📊 Problems Attempted: {}", self.problems_attempted);
        println!("❌ Learning Gate (≤5): {}", self.learning_gate_count); 
        println!("😐 Indifferent (6-7):  {}", self.indifferent_count);
        println!("✅ Memory Gate (≥8):   {}", self.memory_gate_count);
        
        let avg_quality = self.results.iter()
            .map(|r| r.quality_score as f32)
            .sum::<f32>() / self.results.len() as f32;
        let avg_math = self.results.iter()
            .map(|r| r.mathematical_correctness as f32)  
            .sum::<f32>() / self.results.len() as f32;
            
        println!("📈 Average Quality: {:.1}/10", avg_quality);
        println!("🧮 Average Math Score: {:.1}/10", avg_math);
        
        if self.memory_gate_count >= 7 {
            println!("🎉 INTELLIGENCE SUCCESS: ≥7 problems solved at high quality!");
        } else if self.memory_gate_count >= 4 {
            println!("⚠️  PARTIAL SUCCESS: Some mathematical reasoning demonstrated");
        } else {
            println!("🔥 INTELLIGENCE FAILURE: System needs improvement in mathematical reasoning");
        }
    }
}
