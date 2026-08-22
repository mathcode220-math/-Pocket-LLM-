"""
Pocket-LLM: Projection Engine Unit Tests
=========================================

Validates that the hyperbolic projection engine correctly compresses
high-dimensional Euclidean matrices into low-dimensional Poincaré space
while preserving hierarchical relationships and boundary constraints.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from pocket_llm.projection_engine import HyperbolicCoralCompiler


def test_dimension_reduction():
    compiler = HyperbolicCoralCompiler(target_dimensions=2, curvature=1.0)
    flat = np.random.randn(10, 300)
    compressed = compiler.project_euclidean_to_poincare(flat)
    assert compressed.shape == (10, 2), f"Expected shape (10, 2), got {compressed.shape}"
    print("Test passed: Dimension reduction from 300 to 2 successful.")


def test_poincare_boundary():
    compiler = HyperbolicCoralCompiler(target_dimensions=5, curvature=1.0)
    flat = np.random.randn(5, 100) * 10.0  # Large values
    compressed = compiler.project_euclidean_to_poincare(flat)
    norms = np.linalg.norm(compressed, axis=1)
    assert np.all(norms < 1.0), "All compressed vectors must lie strictly inside the Poincaré disk (norm < 1)"
    print("Test passed: All vectors respect Poincaré disk boundary (norm < 1).")


def test_information_preservation():
    compiler = HyperbolicCoralCompiler(target_dimensions=2, curvature=1.0)
    # Create a simple hierarchy: similar words should stay close
    flat = np.array([
        [10.0, 5.0, 2.0] + [0.0]*297,   # Root concept
        [9.5,  4.8, 1.9] + [0.0]*297,   # Similar concept
        [1.0,  0.5, 0.1] + [0.0]*297,   # Distant concept
    ])
    compressed = compiler.project_euclidean_to_poincare(flat)
    # In hyperbolic space, similar vectors should remain closer than distant ones
    dist_01 = np.linalg.norm(compressed[0] - compressed[1])
    dist_02 = np.linalg.norm(compressed[0] - compressed[2])
    assert dist_01 < dist_02, "Hierarchical proximity must be preserved after compression"
    print("Test passed: Hierarchical relationships preserved in curved space.")


if __name__ == "__main__":
    test_dimension_reduction()
    test_poincare_boundary()
    test_information_preservation()
    print("\nAll projection engine tests passed successfully!")
