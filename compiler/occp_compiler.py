#!/usr/bin/env python3
"""
OCCP Hardware Compiler
======================
Transforms high-level AI model weights into optimized binary/hex format
for the OCCP silicon co-processor.

Features:
- Load weights from real ONNX models
- Generate mock weights for testing
- Matrix tiling to match systolic array dimensions
- Optional INT8 quantization
- Binary (.bin) and HEX (.hex) export for simulation

Usage:
    # Mock data (no model file needed)
    python occp_compiler.py --rows 8 --cols 8 --output test_model

    # Real ONNX model
    python occp_compiler.py --onnx model.onnx --layer Conv_0 --quantize --output compiled

    # Export as HEX for testbench simulation
    python occp_compiler.py --onnx model.onnx --hex --output compiled
"""

import argparse
import struct
import sys
import os
import numpy as np

# Try to import ONNX library (optional - required only for real model loading)
try:
    import onnx
    from onnx import numpy_helper
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


class OCCPCompiler:
    """
    Compiler that transforms AI model weights into hardware-compatible
    binary/hex format for the OCCP systolic array co-processor.
    """

    def __init__(self, target_hardware_size=2, enable_quantization=False):
        """
        Initialize the OCCP compiler.

        Args:
            target_hardware_size (int): Size of the systolic array (NxN).
                                        Default is 2 for 2x2 matrix multiply unit.
            enable_quantization (bool): If True, quantize weights to INT8.
        """
        self.target_size = target_hardware_size
        self.enable_quantization = enable_quantization
        self.quant_scale = 1.0
        self.quant_zero_point = 0

        print("=" * 60)
        print("Pocket-LLM Hardware Compiler - OCCP Edition")
        print("=" * 60)
        print(f"[OCCP] Target Core: {target_hardware_size}x{target_hardware_size} Systolic Array")
        print(f"[OCCP] Quantization: {'ENABLED (INT8)' if enable_quantization else 'DISABLED (Float32)'}")

    def load_from_onnx(self, onnx_path, layer_name=None):
        """
        Load weights from a real ONNX model file.

        Args:
            onnx_path (str): Path to the .onnx model file.
            layer_name (str): Optional. Name of the specific layer to extract.
                              If None, extracts the first 2D weight matrix found.

        Returns:
            numpy.ndarray: The extracted weight matrix.

        Raises:
            RuntimeError: If ONNX library is not installed.
            FileNotFoundError: If the model file does not exist.
            ValueError: If no valid weight matrix is found.
        """
        if not ONNX_AVAILABLE:
            raise RuntimeError(
                "ONNX library is not installed. "
                "Please install it with: pip install onnx"
            )

        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"Model file not found: {onnx_path}")

        print(f"[OCCP] Loading ONNX model: {onnx_path}")
        model = onnx.load(onnx_path)

        # Search through model initializers (weight tensors)
        weights = None
        found_name = "Unknown"

        for init in model.graph.initializer:
            # Filter by layer name if specified
            if layer_name and layer_name not in init.name:
                continue

            candidate = numpy_helper.to_array(init)

            # Accept 2D matrices directly
            if candidate.ndim == 2:
                weights = candidate.astype(np.float32)
                found_name = init.name
                break
            # Flatten higher-dimensional tensors (e.g., 3D/4D conv weights)
            elif candidate.ndim > 2:
                weights = candidate.reshape(candidate.shape[0], -1).astype(np.float32)
                found_name = init.name
                break

        if weights is None:
            raise ValueError(
                f"No valid weight matrix found in model '{onnx_path}'. "
                f"Try specifying a different --layer name."
            )

        print(f"[OCCP] Extracted layer: '{found_name}'")
        print(f"[OCCP] Original shape: {weights.shape}")

        return weights

    def generate_mock_weights(self, rows, cols):
        """
        Generate random weight matrix for testing purposes.

        Args:
            rows (int): Number of rows.
            cols (int): Number of columns.

        Returns:
            numpy.ndarray: Random float32 weight matrix.
        """
        print(f"[OCCP] Generating MOCK weights: {rows}x{cols}")
        np.random.seed(42)  # Reproducible results for testing
        weights = np.random.randn(rows, cols).astype(np.float32)
        return weights

    def quantize_weights(self, weights):
        """
        Quantize float32 weights to INT8 for memory efficiency.

        Uses symmetric quantization: value_int8 = round(value_float / scale)

        Args:
            weights (numpy.ndarray): Input float32 weight matrix.

        Returns:
            numpy.ndarray: Quantized int8 weight matrix.
        """
        if not self.enable_quantization:
            return weights

        print("[OCCP] Applying INT8 Quantization...")

        max_abs_val = np.max(np.abs(weights))
        if max_abs_val == 0:
            max_abs_val = 1.0  # Avoid division by zero

        # Symmetric quantization scale
        self.quant_scale = max_abs_val / 127.0
        self.quant_zero_point = 0

        quantized = np.round(weights / self.quant_scale).astype(np.int8)

        print(f"[OCCP] Quantization Scale: {self.quant_scale:.6f}")
        print(f"[OCCP] Zero Point: {self.quant_zero_point}")

        return quantized

    def print_weight_stats(self, weights):
        """Print statistical summary of the weight matrix."""
        print("[OCCP] Weight statistics:")
        print(f"  - Min:  {np.min(weights):.6f}")
        print(f"  - Max:  {np.max(weights):.6f}")
        print(f"  - Mean: {np.mean(weights):.6f}")
        print(f"  - Std:  {np.std(weights):.6f}")

    def tile_weights(self, weights):
        """
        Split weight matrix into NxN tiles matching the systolic array size.
        Applies zero-padding if matrix dimensions are not multiples of tile size.

        Args:
            weights (numpy.ndarray): Input weight matrix.

        Returns:
            list: List of 2D numpy arrays (tiles).
        """
        rows, cols = weights.shape
        t_size = self.target_size

        # Calculate padding needed to make dimensions multiples of tile size
        pad_rows = (t_size - (rows % t_size)) % t_size
        pad_cols = (t_size - (cols % t_size)) % t_size

        if pad_rows > 0 or pad_cols > 0:
            print(f"[OCCP] Padding matrix: +{pad_rows} rows, +{pad_cols} cols")
            weights = np.pad(
                weights,
                ((0, pad_rows), (0, pad_cols)),
                mode='constant',
                constant_values=0
            )

        rows, cols = weights.shape
        tiles = []

        print(f"[OCCP] Tiling {rows}x{cols} matrix into {t_size}x{t_size} blocks...")

        for r in range(0, rows, t_size):
            for c in range(0, cols, t_size):
                tile = weights[r:r + t_size, c:c + t_size]
                tiles.append(tile)

        print(f"[OCCP] Generated {len(tiles)} tiles.")
        return tiles

    def export_binary(self, tiles, output_path):
        """
        Export tiles to binary file (.bin).
        Each tile is stored in row-major order.

        Binary format:
            Float32 mode: 4 bytes per weight, 16 bytes per 2x2 tile
            INT8 mode:    1 byte per weight, 4 bytes per 2x2 tile

        Args:
            tiles (list): List of 2D numpy arrays.
            output_path (str): Output file path.
        """
        with open(output_path, "wb") as f:
            for tile in tiles:
                flat_tile = tile.flatten()  # Row-major order

                if self.enable_quantization:
                    # INT8: 1 byte per weight
                    f.write(flat_tile.astype(np.int8).tobytes())
                else:
                    # Float32: 4 bytes per weight
                    f.write(flat_tile.astype(np.float32).tobytes())

        file_size = os.path.getsize(output_path)
        print(f"[OCCP] Binary export: {output_path}")
        print(f"[OCCP] Binary size: {file_size} bytes ({file_size / 1024:.2f} KB)")

    def export_hex(self, tiles, output_path):
        """
        Export tiles to HEX file (.hex) for SystemVerilog $readmemh.
        Each weight is stored as one line of hex digits.

        HEX format:
            Float32 mode: 8 hex digits per weight (32-bit IEEE 754)
            INT8 mode:    2 hex digits per weight (8-bit signed)

        Args:
            tiles (list): List of 2D numpy arrays.
            output_path (str): Output file path.
        """
        weight_count = 0

        with open(output_path, "w") as f:
            for tile in tiles:
                flat_tile = tile.flatten()  # Row-major order

                for weight in flat_tile:
                    if self.enable_quantization:
                        # INT8: Convert to unsigned 8-bit representation
                        # Handle negative values using two's complement
                        int_val = int(weight) & 0xFF
                        f.write(f"{int_val:02X}\n")
                    else:
                        # Float32: Convert to IEEE 754 32-bit representation
                        # struct.pack converts float to 4 bytes, then we interpret as int
                        packed = struct.pack('<f', float(weight))
                        hex_val = struct.unpack('<I', packed)[0]
                        f.write(f"{hex_val:08X}\n")

                    weight_count += 1

        file_size = os.path.getsize(output_path)
        print(f"[OCCP] HEX export: {output_path}")
        print(f"[OCCP] HEX size: {file_size} bytes ({file_size / 1024:.2f} KB)")
        print(f"[OCCP] Total weights exported: {weight_count}")

    def compile(self, args):
        """
        Main compilation pipeline: Load -> Quantize -> Tile -> Export.

        Args:
            args: Parsed command-line arguments.
        """
        # Step 1: Load weights (real ONNX or mock)
        if args.onnx_path:
            weights = self.load_from_onnx(args.onnx_path, args.layer_name)
        else:
            weights = self.generate_mock_weights(args.rows, args.cols)

        # Step 2: Print statistics
        self.print_weight_stats(weights)

        # Step 3: Quantize if enabled
        processed_weights = self.quantize_weights(weights)

        # Step 4: Tile into hardware-compatible blocks
        tiles = self.tile_weights(processed_weights)

        # Step 5: Export binary file (always)
        bin_path = args.output
        if not bin_path.endswith('.bin'):
            bin_path += '.bin'
        self.export_binary(tiles, bin_path)

        # Step 6: Export HEX file (if requested)
        if args.export_hex:
            hex_path = args.output
            if hex_path.endswith('.bin'):
                hex_path = hex_path[:-4]
            hex_path += '.hex'
            self.export_hex(tiles, hex_path)

        # Summary
        print("=" * 60)
        print("[OCCP] Compilation complete!")
        print(f"[OCCP] Binary output: {bin_path}")
        if args.export_hex:
            print(f"[OCCP] HEX output:    {hex_path}")
        print("[OCCP] Ready for hardware deployment.")
        print("=" * 60)


def main():
    """Command-line interface for the OCCP compiler."""
    parser = argparse.ArgumentParser(
        description="OCCP Hardware Compiler - Convert AI models to hardware format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate mock weights for testing
  python occp_compiler.py --rows 8 --cols 8 --output test_model

  # Compile a real ONNX model with INT8 quantization
  python occp_compiler.py --onnx model.onnx --quantize --output compiled

  # Compile with HEX output for SystemVerilog simulation
  python occp_compiler.py --onnx model.onnx --hex --output compiled

  # Target a 4x4 systolic array instead of default 2x2
  python occp_compiler.py --onnx model.onnx --size 4 --output compiled
        """
    )

    # Input source options
    input_group = parser.add_argument_group("Input Source")
    input_group.add_argument(
        "--onnx",
        dest="onnx_path",
        type=str,
        default=None,
        help="Path to ONNX model file (uses mock data if not specified)"
    )
    input_group.add_argument(
        "--layer",
        dest="layer_name",
        type=str,
        default=None,
        help="Name of specific layer to extract (default: first 2D weight)"
    )
    input_group.add_argument(
        "--rows",
        type=int,
        default=4,
        help="Number of rows for mock data generation (default: 4)"
    )
    input_group.add_argument(
        "--cols",
        type=int,
        default=4,
        help="Number of columns for mock data generation (default: 4)"
    )

    # Hardware target options
    hw_group = parser.add_argument_group("Hardware Target")
    hw_group.add_argument(
        "--size",
        type=int,
        default=2,
        help="Systolic array size NxN (default: 2)"
    )
    hw_group.add_argument(
        "--quantize",
        action="store_true",
        help="Enable INT8 quantization (reduces memory by 75%%)"
    )

    # Output options
    out_group = parser.add_argument_group("Output")
    out_group.add_argument(
        "--output",
        type=str,
        default="compiled_model",
        help="Output file base name (default: compiled_model)"
    )
    out_group.add_argument(
        "--hex",
        dest="export_hex",
        action="store_true",
        help="Also export HEX file for SystemVerilog testbench simulation"
    )

    args = parser.parse_args()

    # Initialize compiler
    compiler = OCCPCompiler(
        target_hardware_size=args.size,
        enable_quantization=args.quantize
    )

    # Run compilation pipeline
    try:
        compiler.compile(args)
    except Exception as e:
        print(f"\n[ERROR] Compilation failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
