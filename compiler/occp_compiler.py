"""
Pocket-LLM Hardware Compiler
=============================

This module compiles large language model weights into hardware-compatible
binary tiles for the OCCP co-processor's 2x2 systolic array.
"""

import numpy as np
import os
import argparse
import sys


class OCCPCompiler:
    """
    Hardware compiler for Pocket-LLM models targeting OCCP silicon.
    """
    
    def __init__(self, target_hardware_size=2, enable_quantization=False):
        self.hardware_size = target_hardware_size
        self.enable_quantization = enable_quantization
        self.tiles_generated = 0
        self.total_bytes = 0
        
        print(f"[OCCP Compiler] Target Core initialized for {self.hardware_size}x{self.hardware_size} Systolic Array.")
        print(f"[OCCP Compiler] Quantization: {'ENABLED (INT8)' if enable_quantization else 'DISABLED (Float32)'}")
    
    def load_mock_llm_weights(self, layer_name, rows, cols):
        print(f"[OCCP Compiler] Extracting weights for layer: '{layer_name}' ({rows}x{cols})...")
        weights = np.random.randn(rows, cols).astype(np.float32) * 0.1
        
        print(f"[OCCP Compiler] Weight statistics:")
        print(f"  - Min: {weights.min():.4f}")
        print(f"  - Max: {weights.max():.4f}")
        print(f"  - Mean: {weights.mean():.4f}")
        print(f"  - Std: {weights.std():.4f}")
        
        return weights
    
    def compile_and_tile_weights(self, weights):
        rows, cols = weights.shape
        print(f"[OCCP Compiler] Starting Tiling process for weights shape: {rows}x{cols}")
        
        tiles = []
        
        for r in range(0, rows, self.hardware_size):
            for c in range(0, cols, self.hardware_size):
                tile = np.zeros((self.hardware_size, self.hardware_size), dtype=np.float32)
                chunk = weights[r:r+self.hardware_size, c:c+self.hardware_size]
                tile[:chunk.shape[0], :chunk.shape[1]] = chunk
                tiles.append(tile)
        
        self.tiles_generated = len(tiles)
        print(f"[OCCP Compiler] Generated {len(tiles)} compiled hardware-compatible tiles.")
        
        return tiles
    
    def quantize_tiles(self, tiles):
        if not self.enable_quantization:
            return tiles
        
        print(f"[OCCP Compiler] Quantizing {len(tiles)} tiles to INT8...")
        
        quantized_tiles = []
        for tile in tiles:
            scale = 127.0 / max(abs(tile.min()), abs(tile.max()), 1e-8)
            quantized = np.clip(tile * scale, -128, 127).astype(np.int8)
            quantized_tiles.append(quantized)
        
        print(f"[OCCP Compiler] Quantization complete. Memory footprint reduced by 75%.")
        return quantized_tiles
    
    def export_to_binary(self, tiles, output_path="compiled_model.bin"):
        print(f"[OCCP Compiler] Exporting to binary: {output_path}")
        
        with open(output_path, "wb") as f:
            for tile in tiles:
                f.write(tile.tobytes())
                self.total_bytes += tile.nbytes
        
        file_size = os.path.getsize(output_path)
        print(f"[OCCP Compiler] Compilation success!")
        print(f"  - Tiles generated: {self.tiles_generated}")
        print(f"  - Binary file size: {file_size:,} bytes ({file_size / (1024*1024):.2f} MB)")
        print(f"  - Output path: {output_path}")
        
        return output_path
    
    def compile_model(self, layer_name="q_proj_layer_0", rows=4, cols=4, output_path="compiled_model.bin"):
        print("=" * 60)
        print("Pocket-LLM Hardware Compiler - OCCP Edition")
        print("=" * 60)
        
        weights = self.load_mock_llm_weights(layer_name, rows, cols)
        tiles = self.compile_and_tile_weights(weights)
        tiles = self.quantize_tiles(tiles)
        binary_path = self.export_to_binary(tiles, output_path)
        
        print("=" * 60)
        print("Compilation complete! Ready for hardware deployment.")
        print(f"Next step: Use the OCCP C driver to stream '{binary_path}' to silicon.")
        print("=" * 60)
        
        return binary_path


def main():
    parser = argparse.ArgumentParser(
        description="Pocket-LLM Hardware Compiler for OCCP Silicon"
    )
    
    parser.add_argument("--layer", type=str, default="q_proj_layer_0",
                        help="Name of the layer to compile")
    parser.add_argument("--rows", type=int, default=4,
                        help="Number of rows in weight matrix")
    parser.add_argument("--cols", type=int, default=4,
                        help="Number of columns in weight matrix")
    parser.add_argument("--output", type=str, default="compiled_model.bin",
                        help="Output binary file path")
    parser.add_argument("--hardware-size", type=int, default=2,
                        help="Target systolic array size")
    parser.add_argument("--quantize", action="store_true",
                        help="Enable INT8 quantization")
    
    args = parser.parse_args()
    
    compiler = OCCPCompiler(
        target_hardware_size=args.hardware_size,
        enable_quantization=args.quantize
    )
    
    try:
        compiler.compile_model(
            layer_name=args.layer,
            rows=args.rows,
            cols=args.cols,
            output_path=args.output
        )
    except Exception as e:
        print(f"[OCCP Compiler] ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
