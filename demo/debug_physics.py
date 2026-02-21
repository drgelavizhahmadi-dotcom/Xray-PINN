"""
Debug script to test if physics constraints are actually filtering predictions.
"""
import sys
sys.path.insert(0, 'mdss_uncertainty_module/src')
sys.path.insert(0, 'src')

from uncertainty_module.core.physics_extremity import ExtremityPhysics
from uncertainty_module.core.physics_bone import BoneAgePhysics
import torch
import numpy as np

print("="*60)
print("PHYSICS CONSTRAINT DEBUG")
print("="*60)

# Test 1: Extremity Physics
print("\n1. Testing ExtremityPhysics (MURA - binary classification)")
print("-"*60)

physics = ExtremityPhysics()

# Case 1: Model predicts both classes with similar probability
print("\nCase 1: Ambiguous prediction [Normal=0.55, Abnormal=0.45]")
pred_set = [0, 1]  # Both classes
probs = np.array([0.55, 0.45])
print(f"  Input set: {pred_set}")
print(f"  Probs: {probs}")
filtered = physics.apply(pred_set, probs)
print(f"  Output set: {filtered}")
if set(filtered) == set(pred_set):
    print("  [WARN] Physics returned same set (no filtering)")
else:
    print(f"  [OK] Physics filtered! Removed: {set(pred_set) - set(filtered)}")

# Case 2: Clear abnormal prediction
print("\nCase 2: Clear abnormal [Normal=0.2, Abnormal=0.8]")
pred_set = [0, 1]
probs = np.array([0.2, 0.8])
print(f"  Input set: {pred_set}")
print(f"  Probs: {probs}")
filtered = physics.apply(pred_set, probs)
print(f"  Output set: {filtered}")
if len(filtered) == 1 and filtered[0] == 1:
    print("  [OK] Physics correctly identified abnormal")
else:
    print(f"  [INFO] Set after physics: {filtered}")

# Case 3: Clear normal prediction
print("\nCase 3: Clear normal [Normal=0.85, Abnormal=0.15]")
pred_set = [0, 1]
probs = np.array([0.85, 0.15])
print(f"  Input set: {pred_set}")
print(f"  Probs: {probs}")
filtered = physics.apply(pred_set, probs)
print(f"  Output set: {filtered}")
if len(filtered) == 1 and filtered[0] == 0:
    print("  [OK] Physics correctly identified normal")
else:
    print(f"  [INFO] Set after physics: {filtered}")

# Test 2: Bone Age Physics
print("\n" + "="*60)
print("2. Testing BoneAgePhysics (4-class classification)")
print("-"*60)

bone_physics = BoneAgePhysics()

# Case 1: Scattered predictions
print("\nCase 1: Scattered predictions [Infant, Toddler, Child, Adolescent]")
pred_set = [0, 1, 2, 3]
probs = np.array([0.3, 0.3, 0.2, 0.2])
print(f"  Input set: {pred_set} (Infant, Toddler, Child, Adolescent)")
print(f"  Probs: {probs}")
filtered = bone_physics.apply(pred_set, probs)
print(f"  Output set: {filtered}")
if len(filtered) < len(pred_set):
    print(f"  [OK] Physics reduced set from {len(pred_set)} to {len(filtered)} classes")
else:
    print("  [INFO] No reduction (scattered predictions allowed as uncertain)")

# Case 2: Non-contiguous predictions (should be filtered)
print("\nCase 2: Non-contiguous [Infant, Child] (skipping Toddler)")
pred_set = [0, 2]  # Infant and Child, no Toddler
probs = np.array([0.4, 0.0, 0.4, 0.0])  # Index 0 and 2
print(f"  Input set: {pred_set}")
print(f"  Probs: {probs}")
filtered = bone_physics.apply(pred_set, probs)
print(f"  Output set: {filtered}")
if set(filtered) != set(pred_set):
    print(f"  [OK] Physics corrected non-contiguous predictions")
else:
    print("  [INFO] Physics kept both (may be valid uncertainty)")

# Test 3: Summary
print("\n" + "="*60)
print("3. SUMMARY")
print("="*60)

print("""
Physics Constraint Behavior:
- ExtremityPhysics: Filters based on probability difference (>0.2 threshold)
  * If Normal >> Abnormal -> keep Normal only
  * If Abnormal >> Normal -> keep Abnormal only
  * If similar -> keep both (uncertain)

- BoneAgePhysics: Enforces developmental sequence
  * Removes non-contiguous predictions
  * Keeps consecutive age groups
  * e.g., [Infant, Toddler, Child] is OK
  * e.g., [Infant, Child] (skip Toddler) -> filtered

Key Insight:
Physics doesn't always reduce set size because:
1. It only filters when predictions are anatomically impossible
2. Uncertain cases (similar probs) are kept for human review
3. The goal is VALIDITY, not just compression
""")

# Test 4: Verify with actual numbers
print("\n4. QUANTITATIVE CHECK")
print("-"*60)

# Run multiple test cases
test_cases = [
    ([0, 1], [0.9, 0.1], "Clear normal"),
    ([0, 1], [0.1, 0.9], "Clear abnormal"),
    ([0, 1], [0.6, 0.4], "Uncertain"),
    ([0, 1], [0.55, 0.45], "Very uncertain"),
]

print("\nExtremityPhysics test cases:")
for pred_set, probs, desc in test_cases:
    filtered = physics.apply(pred_set, np.array(probs))
    reduction = len(pred_set) - len(filtered)
    print(f"  {desc:20s}: {pred_set} -> {filtered} (reduced by {reduction})")

print("\n" + "="*60)
print("Debug complete!")
print("="*60)
