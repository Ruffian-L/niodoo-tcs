#!/usr/bin/env python3
"""
Complete pipeline: Code → AST → Graph → Matrix → TDA

Uses RunPod's venv with giotto-tda already installed.
"""

import json
import numpy as np
from gtda.homology import VietorisRipsPersistence

def parse_code_stub(code: str, language: str) -> dict:
    """Stub: Returns mock AST data for now"""
    # For testing pipeline without tree-sitter version issues
    return {
        "file": "test.rs",
        "language": language,
        "graph": {"node_count": 5, "edge_count": 4},
        "matrix": {
            "shape": [5, 5],
            "data": [
                0.0, 1.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 1.0, 1.0,
                0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0
            ]
        }
    }

def compute_tda(matrix_data: dict) -> dict:
    """Compute persistent homology"""
    shape = tuple(matrix_data['shape'])
    matrix = np.array(matrix_data['data'], dtype=np.float64).reshape(shape)

    vr = VietorisRipsPersistence(metric="precomputed", homology_dimensions=[0, 1, 2])
    matrix_3d = np.expand_dims(matrix, axis=0)
    diagrams = vr.fit_transform(matrix_3d)

    # Compute Betti numbers
    beta_0 = beta_1 = beta_2 = 0
    for pair in diagrams[0]:
        birth, death, dim = pair
        persistence = death - birth
        if persistence > 1e-10 and not np.isinf(death):
            if dim == 0: beta_0 += 1
            elif dim == 1: beta_1 += 1
            elif dim == 2: beta_2 += 1

    return {
        "betti_numbers": {"beta_0": int(beta_0), "beta_1": int(beta_1), "beta_2": int(beta_2)},
        "persistence_pairs": [
            {"birth": float(p[0]), "death": float(p[1]), "dimension": int(p[2]), "persistence": float(p[1]-p[0])}
            for p in diagrams[0] if not np.isinf(p[1])
        ]
    }

def run_pipeline(code: str, language: str) -> dict:
    """Full pipeline"""
    import time
    start = time.time()

    # Step 1: Parse (stub for now)
    ast_data = parse_code_stub(code, language)

    # Step 2: TDA
    topology = compute_tda(ast_data["matrix"])

    elapsed = time.time() - start

    return {
        "file": ast_data["file"],
        "language": language,
        "graph": ast_data["graph"],
        "topology": topology,
        "latency_ms": elapsed * 1000
    }

if __name__ == "__main__":
    test_code = "fn main() { let x = 42; }"
    result = run_pipeline(test_code, "rust")
    print(json.dumps(result, indent=2))
    print(f"\n✓ Pipeline latency: {result['latency_ms']:.2f}ms")
