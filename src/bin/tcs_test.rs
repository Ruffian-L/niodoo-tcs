use niodoo_consciousness::tcs::*;

/// Simple test program for the Topological Cognitive System
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 TCS (Topological Cognitive System) Test Starting...");

    // Test basic topology components
    println!("Testing Takens Embedding...");
    let time_series = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
        vec![1.0, 1.0, 0.0],
        vec![0.0, 1.0, 1.0],
        vec![1.0, 0.0, 1.0],
    ];

    let embedding = topology::takens_embedding::TakensEmbedding::new(2, 1, 3);
    let embedded = embedding.embed(&time_series);
    println!("Embedded {} points into {}D space", embedded.len(), embedded[0].len());

    // Test persistent homology
    println!("Testing Persistent Homology...");
    let mut homology = topology::persistent_homology::PersistentHomology::new();
    let persistence_result = homology.compute(&embedded)?;
    println!("Found {} persistent features", persistence_result.points.len());

    // Test Jones polynomial
    println!("Testing Jones Polynomial...");
    let knot = topology::jones_polynomial::KnotDiagram {
        crossings: vec![],
        gauss_code: vec![],
        pd_code: vec![],
    };
    let jones = topology::jones_polynomial::JonesPolynomial::compute(&knot);
    println!("Jones polynomial has {} coefficients", jones.coefficients.len());

    // Test cognitive state
    println!("Testing Cognitive State...");
    let mut state = pipeline::CognitiveState::new();
    state.update_betti_numbers([1, 1, 0]);
    println!("Cognitive state: Betti numbers {:?}", state.betti_numbers);

    // Test knot analysis
    println!("Testing Knot Analysis...");
    let cycle = geometry::HomologyCycle {
        persistence: 0.8,
        dimension: 1,
        representative: vec![
            vec![0.0, 0.0, 1.0],
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
        ],
    };

    let analyzer = pipeline::KnotAnalyzer::new();
    let knot = analyzer.analyze_cycle(&cycle).await?;
    println!("Analyzed knot with persistence {:.2} and complexity {:.2}",
             knot.persistence, knot.complexity_score);

    println!("✅ TCS Test Completed Successfully!");
    Ok(())
}