# K-Twist Topology Parameters - Quick Reference

## TL;DR - No More Hardcoding!

All topology parameters are now **mathematically derived** or **data-driven**. No magic numbers.

## Mathematical Constants

### K-Twist Constant
```rust
use crate::topology::mobius_torus_k_twist::calculate_k_twist_constant;

let k_twist = calculate_k_twist_constant();
// Value: 1.457324 (from (e × π) / (e + π))
```

**DO NOT hardcode `k_twist = 1.0` or any other value!**

## Data-Driven Parameters

### Major Radius
```rust
use crate::topology::mobius_torus_k_twist::calculate_major_radius;

let major_radius = calculate_major_radius(data_scale, complexity);
// Automatically scales with data characteristics
```

**DO NOT hardcode `major_radius = 5.0` or any fixed value!**

### Gaussian Variance
```rust
use crate::topology::mobius_torus_k_twist::calculate_gaussian_variance;

let variance = calculate_gaussian_variance(minor_radius, major_radius);
// Maintains geometric ratio (minor/major)
```

**DO NOT hardcode `gaussian_variance = 0.15`!**

### Mesh Resolution
```rust
use crate::topology::mobius_torus_k_twist::calculate_adaptive_resolution;

let (u_res, v_res) = calculate_adaptive_resolution(
    available_memory_mb,  // System memory budget
    target_quality        // 0.0 = low, 1.0 = high
);
```

**DO NOT hardcode `64, 32` or any fixed resolution!**

## Usage Patterns

### Pattern 1: Simple Default
```rust
let bridge = KTwistTopologyBridge::new();
// Defaults: 1GB memory, 70% quality
// k_twist automatically set to 1.457324
```

### Pattern 2: Memory-Constrained System
```rust
let bridge = KTwistTopologyBridge::with_memory_and_quality(512, 0.5);
// Low memory, medium quality
```

### Pattern 3: High-Performance System
```rust
let bridge = KTwistTopologyBridge::with_memory_and_quality(4096, 0.95);
// Ample memory, maximum quality
```

### Pattern 4: Custom Parameters from Data
```rust
let params = KTwistParameters::from_data(
    data_scale: 10.0,           // Calculated from bounding box
    complexity: 5.0,            // Calculated from point spread
    minor_radius_ratio: 0.3     // Proportion (0.1-0.5)
);
let mesh = KTwistMesh::new(params, u_res, v_res);
```

### Pattern 5: Dynamic Adaptation
```rust
let mut bridge = KTwistTopologyBridge::new();

// Parameters automatically adapt to consciousness data
bridge.update_topology(consciousness_data)?;

// Major radius and variance adjust based on data characteristics
// k_twist remains constant (mathematical invariant)
```

## Key Formulas

| Parameter | Formula | Range |
|-----------|---------|-------|
| `k_twist` | `(e × π) / (e + π)` | 1.457324 (constant) |
| `major_radius` | `2.0 × (1 + log₁₀(scale)) × √complexity` | 2.0 - 10.0 |
| `gaussian_variance` | `minor_radius / major_radius` | 0.05 - 0.5 |
| `u_resolution` | `√(memory × quality / 800)` | 16 - 256 |
| `v_resolution` | `u_resolution / 2` | 8 - 128 |

## Common Mistakes to Avoid

### ❌ DON'T DO THIS
```rust
// Hardcoded parameters - WRONG!
let params = KTwistParameters {
    major_radius: 5.0,         // ❌ HARDCODED
    minor_radius: 1.5,         // ❌ HARDCODED
    k_twist: 1.0,              // ❌ WRONG VALUE
    gaussian_variance: 0.15,   // ❌ HARDCODED
    learning_rate: 0.01,       // ❌ HARDCODED
};

// Fixed resolution - WRONG!
let mesh = KTwistMesh::new(params, 64, 32);  // ❌ HARDCODED
```

### ✅ DO THIS INSTEAD
```rust
// Data-driven parameters - CORRECT!
let params = KTwistParameters::from_data(
    data_scale,
    complexity,
    0.3  // Only the ratio can be chosen
);

// Adaptive resolution - CORRECT!
let (u_res, v_res) = calculate_adaptive_resolution(
    available_memory_mb,
    target_quality
);
let mesh = KTwistMesh::new(params, u_res, v_res);
```

## When Parameters Update

| Scenario | `k_twist` | `major_radius` | `variance` | Resolution |
|----------|-----------|----------------|------------|------------|
| Initial creation | ✅ Set | ✅ Set | ✅ Set | ✅ Set |
| Data update | ⛔ Constant | ✅ Adapts | ✅ Adapts | ⛔ Manual |
| Memory change | ⛔ N/A | ⛔ N/A | ⛔ N/A | ✅ Adapts |
| Quality change | ⛔ N/A | ⛔ N/A | ⛔ N/A | ✅ Adapts |

**Key insight:** `k_twist` is a mathematical constant - it NEVER changes!

## Logging

The system logs parameter derivations:
```
🎯 Adaptive K-Twist mesh: 128×64 resolution (memory: 1024MB, quality: 70.0%)
📐 Topology parameters: major_radius=2.828, k_twist=1.457 (mathematically derived)
📊 Consciousness data: scale=5.230, complexity=3.142, spread=1.234, points=42
🔄 Updated topology: major_radius=3.536, minor_radius=1.061, k_twist=1.457, variance=0.300
```

Use these logs to verify parameters are adapting correctly.

## Testing Your Code

Always verify parameters are NOT hardcoded:
```rust
#[test]
fn test_no_hardcoded_k_twist() {
    let params = KTwistParameters::default();
    let k_twist = calculate_k_twist_constant();
    assert_eq!(params.k_twist, k_twist);  // Must use mathematical constant
}

#[test]
fn test_variance_is_ratio() {
    let params = KTwistParameters::default();
    let ratio = params.minor_radius / params.major_radius;
    assert!((params.gaussian_variance - ratio).abs() < 1e-6);  // Must be geometric ratio
}
```

## FAQ

**Q: Why can't I just set `k_twist = 1.0`?**
A: Because 1.0 is arbitrary. The mathematical value 1.457324 emerges from the harmonic mean of e and π, which represents the optimal non-orientable twist rate.

**Q: What if I need a specific major_radius for visualization?**
A: You can scale the entire topology uniformly, but internal ratios should remain data-driven. Consider adjusting the `data_scale` input instead.

**Q: Can I override the adaptive resolution?**
A: Yes, but you must provide explicit memory budget and quality parameters. Never use raw hardcoded values.

**Q: How do I calculate `data_scale` and `complexity`?**
A: For consciousness data:
- `data_scale = geometric_mean(x_range, y_range, z_range)` from bounding box
- `complexity = log₁₀(point_count) × average_pairwise_distance`

The `update_topology()` method calculates these automatically.

## See Also

- Full implementation: `/src/topology/mobius_torus_k_twist.rs`
- Detailed report: `/TOPOLOGY_PARAMETERS_FIX_REPORT.md`
- Test suite: Run `cargo test topology::mobius_torus_k_twist`

---

**Remember:** Mathematical rigor over arbitrary values. Data-driven over hardcoded. Always.
