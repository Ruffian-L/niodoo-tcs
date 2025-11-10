"""
giotto-tda wrapper for approximate persistent homology computation.
Phase 2.1: TCSAnalyzer Acceleration - 60% speedup target.
"""
import numpy as np
from giotto.homology import VietorisRipsPersistence

def compute_approximate_persistence(points, max_filtration):
    """
    Compute approximate persistent homology using giotto-tda.
    
    Args:
        points: List of points (each point is a list/array of floats)
        max_filtration: Maximum filtration value
        
    Returns:
        dict with keys:
            - features: List of (birth, death, dimension) tuples
            - betti: List of Betti numbers [β₀, β₁, β₂]
            - entropy_weights: List of (dimension, persistence) tuples
    """
    # Convert to numpy array
    points_array = np.array(points, dtype=np.float32)
    
    # Create Vietoris-Rips persistence object
    vr_persistence = VietorisRipsPersistence(
        metric='euclidean',
        homology_dimensions=(0, 1, 2),
        max_edge_length=max_filtration,
        collapse_edges=True,  # Approximate mode for speed
    )
    
    # Reshape for giotto (expects 3D array: [n_samples, n_points, n_features])
    # For single point cloud, add batch dimension
    if len(points_array.shape) == 2:
        points_array = points_array.reshape(1, points_array.shape[0], points_array.shape[1])
    
    # Compute persistence diagram
    persistence_diagram = vr_persistence.fit_transform(points_array)
    
    # Extract features
    features = []
    betti = [0, 0, 0]
    entropy_weights = []
    total_weight = 0.0
    
    # Process persistence diagram (giotto returns as list of arrays)
    if len(persistence_diagram) > 0:
        diagram = persistence_diagram[0]  # First (and only) batch
        
        for point in diagram:
            if len(point) >= 3:
                birth = float(point[0])
                death = float(point[1])
                dimension = int(point[2])
                
                # Handle infinity (giotto uses large values or np.inf)
                if np.isinf(death) or death > max_filtration * 10:
                    death = float('inf')
                    if dimension < 3:
                        betti[dimension] += 1
                
                persistence = death - birth if np.isfinite(death) else float('inf')
                
                if persistence > 0 and np.isfinite(persistence):
                    entropy_weights.append((dimension, float(persistence)))
                    total_weight += persistence
                
                features.append((float(birth), float(death), dimension))
    
    # Normalize entropy weights
    if total_weight > 0:
        normalized_weights = [(dim, weight / total_weight) for dim, weight in entropy_weights]
    else:
        normalized_weights = []
    
    return {
        'features': features,
        'betti': betti,
        'entropy_weights': normalized_weights,
    }



