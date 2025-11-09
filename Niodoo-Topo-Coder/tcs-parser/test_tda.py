#!/usr/bin/env python3
"""Test giotto-tda persistent homology computation"""

import numpy as np
from gtda.homology import VietorisRipsPersistence

# Example adjacency matrix from tcs-parser (3x3 with 2 edges)
# This is the actual output from our Rust code
matrix = np.array([
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0]
], dtype=np.float64)

print("Input matrix:")
print(matrix)
print(f"Shape: {matrix.shape}")

# VietorisRipsPersistence with precomputed metric
vr = VietorisRipsPersistence(metric="precomputed", homology_dimensions=[0, 1, 2])

# giotto-tda expects 3D input: [n_samples, n_points, n_points]
matrix_3d = np.expand_dims(matrix, axis=0)
print(f"\n3D shape for giotto-tda: {matrix_3d.shape}")

# Compute persistence diagram
diagrams = vr.fit_transform(matrix_3d)
print(f"\nPersistence diagram shape: {diagrams.shape}")
print(f"Persistence diagram:\n{diagrams}")

# Extract Betti numbers (count persistent features)
def compute_betti_numbers(diagram, threshold=1e-10):
    beta_0 = beta_1 = beta_2 = 0
    for pair in diagram[0]:  # First sample
        birth, death, dim = pair
        persistence = death - birth
        if persistence > threshold:
            if dim == 0:
                beta_0 += 1
            elif dim == 1:
                beta_1 += 1
            elif dim == 2:
                beta_2 += 1
    return beta_0, beta_1, beta_2

beta_0, beta_1, beta_2 = compute_betti_numbers(diagrams)
print(f"\nBetti numbers: β₀={beta_0}, β₁={beta_1}, β₂={beta_2}")
print("\nSUCCESS: giotto-tda pipeline works!")
