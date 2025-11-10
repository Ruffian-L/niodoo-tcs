#!/usr/bin/env python3
"""
Giotto-TDA Wrapper for NIODOO Lab

This script provides a simple CLI interface to compute persistent homology
using giotto-tda. It's called from Rust via subprocess for the TCS analyzer.

Input: JSON with point cloud coordinates
Output: JSON with Betti numbers and persistence features
"""

import sys
import json
import numpy as np

def compute_persistence(points, max_filtration=2.0):
    """
    Compute persistent homology using giotto-tda.
    
    Args:
        points: List of points (each point is a list of floats)
        max_filtration: Maximum filtration value
        
    Returns:
        dict with:
            - betti_numbers: [β₀, β₁, β₂]
            - persistence_pairs: [(birth, death, dimension), ...]
            - persistence_entropy: float
    """
    try:
        from gtda.homology import VietorisRipsPersistence
    except ImportError:
        # Fallback to giotto-tda if gtda not available
        try:
            from giotto.homology import VietorisRipsPersistence
        except ImportError:
            return {
                "error": "giotto-tda not installed. Run: pip install giotto-tda",
                "betti_numbers": [0, 0, 0],
                "persistence_pairs": [],
                "persistence_entropy": 0.0
            }
    
    # Convert to numpy array
    points_array = np.array(points, dtype=np.float32)
    
    # Reshape for giotto (expects 3D: [n_samples, n_points, n_features])
    if len(points_array.shape) == 2:
        points_array = points_array.reshape(1, points_array.shape[0], points_array.shape[1])
    
    # Create Vietoris-Rips persistence object
    vr_persistence = VietorisRipsPersistence(
        metric='euclidean',
        homology_dimensions=(0, 1, 2),
        max_edge_length=max_filtration,
        collapse_edges=True,  # Approximate mode for speed
    )
    
    # Compute persistence diagram
    try:
        persistence_diagram = vr_persistence.fit_transform(points_array)
    except Exception as e:
        return {
            "error": f"TDA computation failed: {str(e)}",
            "betti_numbers": [0, 0, 0],
            "persistence_pairs": [],
            "persistence_entropy": 0.0
        }
    
    # Extract Betti numbers and persistence pairs
    betti_numbers = [0, 0, 0]
    persistence_pairs = []
    
    # persistence_diagram shape: [n_samples, n_features, 3]
    # where features are (birth, death, dimension)
    if len(persistence_diagram.shape) == 3:
        diagram = persistence_diagram[0]  # First sample
        
        for feature in diagram:
            birth, death, dimension = feature
            dim_int = int(dimension)
            
            # Skip infinite persistence (death = inf)
            if not np.isfinite(death):
                continue
                
            if dim_int < 3:
                betti_numbers[dim_int] += 1
                persistence_pairs.append({
                    "birth": float(birth),
                    "death": float(death),
                    "dimension": dim_int,
                    "persistence": float(death - birth)
                })
    
    # Compute persistence entropy
    persistence_entropy = compute_entropy(persistence_pairs)
    
    return {
        "betti_numbers": betti_numbers,
        "persistence_pairs": persistence_pairs,
        "persistence_entropy": persistence_entropy
    }

def compute_entropy(pairs):
    """Compute Shannon entropy of persistence lifetimes."""
    if not pairs:
        return 0.0
    
    lifetimes = [p["persistence"] for p in pairs]
    total = sum(lifetimes)
    
    if total < 1e-10:
        return 0.0
    
    # Normalize to probabilities
    probs = [l / total for l in lifetimes]
    
    # Shannon entropy: H = -Σ p_i * log(p_i)
    entropy = 0.0
    for p in probs:
        if p > 1e-10:
            entropy -= p * np.log(p)
    
    return float(entropy)

def main():
    """CLI entry point."""
    if len(sys.argv) < 2:
        print(json.dumps({
            "error": "Usage: giotto_wrapper.py <input_json>",
            "betti_numbers": [0, 0, 0],
            "persistence_pairs": [],
            "persistence_entropy": 0.0
        }))
        sys.exit(1)
    
    try:
        # Parse input JSON
        input_data = json.loads(sys.argv[1])
        points = input_data.get("points", [])
        max_filtration = input_data.get("max_filtration", 2.0)
        
        if not points:
            print(json.dumps({
                "error": "No points provided",
                "betti_numbers": [0, 0, 0],
                "persistence_pairs": [],
                "persistence_entropy": 0.0
            }))
            sys.exit(1)
        
        # Compute persistence
        result = compute_persistence(points, max_filtration)
        
        # Output JSON
        print(json.dumps(result))
        
    except Exception as e:
        print(json.dumps({
            "error": f"Unexpected error: {str(e)}",
            "betti_numbers": [0, 0, 0],
            "persistence_pairs": [],
            "persistence_entropy": 0.0
        }))
        sys.exit(1)

if __name__ == "__main__":
    main()

