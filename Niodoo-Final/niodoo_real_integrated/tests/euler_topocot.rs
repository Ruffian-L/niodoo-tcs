use niodoo_real_integrated::pipeline::generation::topo_reasoning::{
    CausalBridge, EmotionalMapping, TopoCoT, TopologicalAnalysis,
};

#[test]
fn euler_topocot_chi_is_two_for_sphere_betti() {
    // Simulate the TopoCoT JSON that an LLM would produce for a 2-sphere
    let cot = TopoCoT {
        step_1_analysis: TopologicalAnalysis {
            betti_0_components: 1,
            betti_1_loops: 0,
            betti_2_voids: 1,
            summary: "β=(1,0,1) indicates a 2-sphere signature".to_string(),
        },
        step_2_emotional_mapping: EmotionalMapping {
            pad_arousal_shift: -0.1,
            pad_valence_shift: 0.3,
            justification: "Closed and simple structure suggests consonance".to_string(),
        },
        step_3_causal_bridge: CausalBridge {
            obstacle: "Link topology to V-E+F=2".to_string(),
            resolution_path: "Use Euler characteristic χ from Betti numbers".to_string(),
            reasoning_chain: "χ = β0 - β1 + β2; with (1,0,1) gives χ=2".to_string(),
        },
        step_4_final_output_grounding:
            "V-E+F=2 holds as a combinatorial computation of χ for the sphere".to_string(),
        computed_artifacts: None,
    };

    let chi = (cot.step_1_analysis.betti_0_components as i32)
        - (cot.step_1_analysis.betti_1_loops as i32)
        + (cot.step_1_analysis.betti_2_voids as i32);
    assert_eq!(chi, 2, "Euler characteristic must be 2 for (1,0,1)");

    // Ensure the schema emits and contains required fields
    let schema = TopoCoT::json_schema();
    assert!(
        schema.get("properties").is_some(),
        "Schema should have properties"
    );
}
