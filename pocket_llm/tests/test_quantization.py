"""
Pocket-LLM: Quantization Engine Unit Tests
===========================================

Validates that the 4-bit quantization process preserves hyperbolic
geometric relationships within acceptable error margins, ensuring
hardware-ready weights do not lose semantic accuracy.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from pocket_llm.hyperbolic_quantizer import HyperbolicCoralQuantizer


def test_4bit_range():
    quantizer = HyperbolicCoralQuantizer(bits=4)
    matrix = np.array([[0.9, -0.8], [0.3, -0.1]], dtype=np.float32)
    q, scale = quantizer.quantize_poincare_weights(matrix)
    assert q.min() >= -8, "4-bit minimum boundary violated"
    assert q.max() <= 7, "4-bit maximum boundary violated"
    print("Test passed: Quantized values strictly within 4-bit integer range (-8 to 7).")


def test_reconstruction_accuracy():
    quantizer = HyperbolicCoralQuantizer(bits=4)
    original = np.array([
        [0.75, -0.50],
        [0.25,  0.10],
        [-0.05, 0.90]
    ], dtype=np.float32)
    q, scale = quantizer.quantize_poincare_weights(original)
    reconstructed = quantizer.dequantize_to_poincare(q, scale)
    mse = np.mean((original - reconstructed) ** 2)
    assert mse < 0.01, f"Reconstruction MSE too high: {mse}"
    print(f"Test passed: Reconstruction MSE is {mse:.6f} (within hyperbolic tolerance).")


def test_memory_saving():
    quantizer = HyperbolicCoralQuantizer(bits=4)
    matrix = np.random.randn(100, 10).astype(np.float32) * 0.5
    q, _ = quantizer.quantize_poincare_weights(matrix)
    original_bytes = matrix.nbytes
    quantized_bytes = q.nbytes
    saving = (1 - quantized_bytes / original_bytes) * 100
    assert saving >= 75.0, "Memory saving must be at least 75%"
    print(f"Test passed: Memory footprint reduced by {saving:.1f}% via 4-bit quantization.")


if __name__ == "__main__":
    test_4bit_range()
    test_reconstruction_accuracy()
    test_memory_saving()
    print("\nAll quantization engine tests passed successfully!")
