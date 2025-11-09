#!/usr/bin/env python3
"""
Standalone TDA computation script.
Reads adjacency matrix JSON, computes persistent homology, outputs Betti numbers.
"""
import sys
import json
import numpy as np
from gtda.homology import VietorisRipsPersistence

def compute_betti_numbers(persistence_diagram, threshold=1e-10):
    """Extract Betti numbers from persistence diagram"""
    beta_0 = beta_1 = beta_2 = 0
    for pair in persistence_diagram[0]:  # First sample
        birth, death, dim = pair
        persistence = death - birth
        if persistence > threshold and not np.isinf(death):
            if dim == 0:
                beta_0 += 1
            elif dim == 1:
                beta_1 += 1
            elif dim == 2:
                beta_2 += 1
    return beta_0, beta_1, beta_2

def main():
    if len(sys.argv) != 2:
        print("Usage: python compute_tda.py <matrix.json>", file=sys.stderr)
        sys.exit(1)

    # Read matrix from JSON
    with open(sys.argv[1], 'r') as f:
        data = json.load(f)

    shape = tuple(data['matrix']['shape'])
    matrix_flat = data['matrix']['data']
    matrix = np.array(matrix_flat, dtype=np.float64).reshape(shape)

    print(f"Input matrix shape: {shape}")
    print(f"Non-zero entries: {np.count_nonzero(matrix)}")

    # Compute persistent homology
    vr = VietorisRipsPersistence(metric="precomputed", homology_dimensions=[0, 1, 2])
    matrix_3d = np.expand_dims(matrix, axis=0)
    diagrams = vr.fit_transform(matrix_3d)

    # Extract Betti numbers
    beta_0, beta_1, beta_2 = compute_betti_numbers(diagrams)

    # Output results as JSON
    result = {
        "file": data["file"],
        "language": data["language"],
        "graph": data["graph"],
        "topology": {
            "betti_numbers": {
                "beta_0": int(beta_0),
                "beta_1": int(beta_1),
                "beta_2": int(beta_2)
            },
            "persistence_pairs": [
                {
                    "birth": float(pair[0]),
                    "death": float(pair[1]),
                    "dimension": int(pair[2]),
                    "persistence": float(pair[1] - pair[0])
                }
                for pair in diagrams[0]
                if not np.isinf(pair[1])  # Skip infinite persistence
            ]
        }
    }

    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
