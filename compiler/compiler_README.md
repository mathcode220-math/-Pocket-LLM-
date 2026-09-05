# OCCP Hardware Compiler

The **OCCP Hardware Compiler** (`occp_compiler.py`) is the native software-compilation layer of the [Pocket-LLM](https://github.com/mathcode220-math/-Pocket-LLM-) framework. It transforms high-level AI model weights into optimized binary (`.bin`) and hexadecimal (`.hex`) formats ready for deployment on the [Open Cognitive Core Project (OCCP)](https://github.com/mathcode220-math/open-cognitive-core) systolic-array co-processor.

While Pocket-LLM compresses heavy LLMs using Hyperbolic Coral Mathematics, the compiler is responsible for the final step: slicing, quantizing, and exporting weight tensors into hardware-aligned micro-binaries that fit directly into the OCCP chip's **SRAM Skew Buffers**.

---

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Mock Weights (No Model File)](#mock-weights-no-model-file)
  - [Real ONNX Models](#real-onnx-models)
  - [INT8 Quantization](#int8-quantization)
  - [HEX Export for SystemVerilog](#hex-export-for-systemverilog)
  - [Custom Systolic Array Size](#custom-systolic-array-size)
- [Command-Line Reference](#command-line-reference)
- [How It Works](#how-it-works)
  - [1. Weight Loading](#1-weight-loading)
  - [2. Quantization](#2-quantization)
  - [3. Tiling](#3-tiling)
  - [4. Export](#4-export)
- [Output Formats](#output-formats)
  - [Binary (.bin)](#binary-bin)
  - [Hexadecimal (.hex)](#hexadecimal-hex)
- [Hardware Alignment](#hardware-alignment)
- [Troubleshooting](#troubleshooting)
- [Integration with Pocket-LLM & OCCP](#integration-with-pocket-llm--occp)
- [License](#license)

---

## Features

- **ONNX Model Ingestion**: Load weights directly from standard `.onnx` files (optional dependency).
- **Mock Weight Generation**: Generate reproducible random matrices for rapid testing without external model files.
- **Symmetric INT8 Quantization**: Reduce memory footprint by ~75% with a simple max-abs symmetric quantizer — critical for fitting compressed edge models into on-chip SRAM.
- **Systolic Array Tiling**: Automatically splits matrices into N×N tiles and applies zero-padding when dimensions are not exact multiples, matching the OCCP `systolic_array_param.sv` core dimensions.
- **Binary Export**: Produces `.bin` files in row-major order, ready for firmware flashing into the co-processor's adaptive weight memory.
- **HEX Export**: Produces `.hex` files compatible with SystemVerilog `$readmemh` for RTL simulation and testbench verification of the systolic array pipeline.
- **Layer Filtering**: Extract a specific layer by name when a model contains multiple weight tensors.

---

## Requirements

### Mandatory
- Python 3.8 or higher
- NumPy

### Optional
- `onnx` — required **only** if you intend to compile real ONNX models.

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/mathcode220-math/-Pocket-LLM-.git
cd -Pocket-LLM-/compiler

# 2. Install mandatory dependency
pip install numpy

# 3. (Optional) Install ONNX support
pip install onnx
```

---

## Quick Start

```bash
# Generate a 4×4 mock weight matrix, tile it for a 2×2 systolic array, and export as binary
python occp_compiler.py --rows 4 --cols 4 --output my_test

# Compile a real ONNX model with INT8 quantization + HEX export
python occp_compiler.py --onnx model.onnx --quantize --hex --output compiled_model
```

---

## Usage

### Mock Weights (No Model File)

Use this mode when you want to test the compiler pipeline or the OCCP RTL testbench without downloading a full AI model.

```bash
python occp_compiler.py --rows 8 --cols 8 --size 4 --output test_matrix
```

- `--rows` and `--cols` define the shape of the random matrix.
- `--size` defines the target systolic array dimensions (default: 2).

### Real ONNX Models

To compile a model exported from PyTorch, TensorFlow, or any ONNX-compatible framework:

```bash
python occp_compiler.py --onnx resnet18.onnx --output compiled
```

By default, the compiler scans the ONNX graph and extracts the **first 2D weight matrix** it finds. Higher-dimensional tensors (e.g., 4D convolution kernels) are automatically flattened to 2D.

#### Extract a Specific Layer

```bash
python occp_compiler.py --onnx resnet18.onnx --layer conv1.weight --output compiled
```

### INT8 Quantization

Enable quantization to shrink the binary size by roughly 75% compared to float32. This is essential for fitting Small Language Models (SLMs) and highly quantized 1B–3B parameter profiles into the OCCP co-processor's limited on-chip SRAM.

```bash
python occp_compiler.py --onnx model.onnx --quantize --output compiled_int8
```

The compiler uses **symmetric quantization**:

```
scale = max(|weights|) / 127
quantized = round(weights / scale)
```

This maps the dynamic range of the weights into signed 8-bit integers (`-127` to `+127`) with zero point fixed at `0`.

### HEX Export for SystemVerilog

For RTL simulation of the OCCP systolic array, generate a `.hex` file that can be read by a testbench via `$readmemh`:

```bash
python occp_compiler.py --onnx model.onnx --hex --output sim_weights
```

This creates both:
- `sim_weights.bin` — raw binary for firmware flashing
- `sim_weights.hex` — one hex value per line for SystemVerilog simulation

### Custom Systolic Array Size

If your OCCP hardware target uses a larger systolic array (e.g., 8×8 instead of the default 2×2):

```bash
python occp_compiler.py --onnx model.onnx --size 8 --output compiled_8x8
```

---

## Command-Line Reference

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--onnx` | `str` | `None` | Path to an ONNX model file. If omitted, mock data is generated. |
| `--layer` | `str` | `None` | Name of a specific weight initializer to extract. If omitted, the first 2D tensor is used. |
| `--rows` | `int` | `4` | Number of rows for mock weight generation. |
| `--cols` | `int` | `4` | Number of columns for mock weight generation. |
| `--size` | `int` | `2` | Target systolic array size N×N. |
| `--quantize` | `flag` | `False` | Enable symmetric INT8 quantization. |
| `--hex` | `flag` | `False` | Also export a `.hex` file for SystemVerilog simulation. |
| `--output` | `str` | `compiled_model` | Base name for output files. |

---

## How It Works

The compiler pipeline consists of four sequential stages:

### 1. Weight Loading

- **ONNX path**: The model is parsed with `onnx.load()`. Initializers are scanned for the first tensor with `ndim >= 2`. If `--layer` is provided, only initializers whose name contains the given substring are considered. Tensors with `ndim > 2` are reshaped to 2D via `reshape(shape[0], -1)`.
- **Mock path**: A reproducible random matrix is generated via `np.random.randn(rows, cols)` with `seed=42`.

### 2. Quantization

If `--quantize` is enabled, the compiler computes a per-tensor symmetric scale factor and converts float32 values to `int8`.

### 3. Tiling

The weight matrix is padded with zeros so that both dimensions become exact multiples of the target tile size (`--size`). It is then sliced into non-overlapping N×N blocks in row-major order.

### 4. Export

- **Binary**: Each tile is flattened in row-major order and appended to a `.bin` file. In float32 mode each weight occupies 4 bytes; in INT8 mode it occupies 1 byte.
- **HEX**: Each weight is written as one line of uppercase hex. In float32 mode the value is packed as little-endian IEEE 754 (`<f`) and reinterpreted as a 32-bit unsigned integer (`08X`). In INT8 mode negative values are converted to their two's-complement unsigned representation (`02X`).

---

## Output Formats

### Binary (.bin)

- **Float32 mode**: 4 bytes per weight, IEEE 754 little-endian.
- **INT8 mode**: 1 byte per weight, signed two's complement.
- Tiles are stored sequentially, each tile flattened row-major.

### Hexadecimal (.hex)

- One value per line — compatible with Verilog/SystemVerilog `$readmemh`.
- **Float32**: 8 hex digits per line (e.g., `3F800000`).
- **INT8**: 2 hex digits per line (e.g., `FF` for `-1`).

---

## Hardware Alignment

The compiler is designed to produce outputs that align with the OCCP hardware architecture:

| Compiler Output | OCCP Hardware Block |
|-----------------|---------------------|
| `.bin` (INT8) | Loaded into **SRAM Skew Buffers** (`sram_skew_buffer.sv`) and **Adaptive Neural Synapses** (RRAM / Flash weight memory). |
| `.hex` | Fed into SystemVerilog testbenches to verify `systolic_array_param.sv`, `matrix_multiply_2x2.sv`, and `softmax_core.sv`. |
| Tile size (`--size`) | Must match the `systolic_array_param.sv` parameter `N` (default 2). |
| Row-major flattening | Matches the data skew pattern expected by the SRAM skew buffer before entering the systolic array. |

---

## Troubleshooting

| Symptom | Cause | Solution |
|---------|-------|----------|
| `ONNX library is not installed` | The `onnx` package is missing. | Run `pip install onnx`. |
| `Model file not found` | The path passed to `--onnx` does not exist. | Verify the file path. |
| `No valid weight matrix found` | The ONNX model contains only 1D tensors (e.g., biases), or `--layer` filtered out everything. | Omit `--layer` to auto-select, or check initializer names with an ONNX inspector. |
| Output `.bin` is larger than expected | Float32 mode is active. | Add `--quantize` to switch to INT8. |
| HEX file has wrong values in simulation | Endianness mismatch. | The compiler uses little-endian (`<f`). Ensure your SystemVerilog testbench expects the same. |

---

## Integration with Pocket-LLM & OCCP

The OCCP Compiler is **Phase 2** of the [Pocket-LLM](https://github.com/mathcode220-math/-Pocket-LLM-) roadmap: *The Compiler Engine*.

In the full Pocket-LLM → OCCP pipeline:

1. **Pocket-LLM Core** compresses a high-dimensional LLM (e.g., Llama, Mistral) into a low-dimensional hyperbolic representation.
2. **OCCP Compiler** (this tool) ingests the compressed weights — either directly or via an ONNX bridge — and performs tiling, quantization, and hardware-aligned export.
3. **OCCP Co-Processor** ([open-cognitive-core](https://github.com/mathcode220-math/open-cognitive-core)) loads the resulting `.bin` into its **SRAM Skew Buffers** and executes native non-Euclidean operations offline via the hardwired systolic array, softmax core, and AXI4-Lite control interface.

For details on the hyperbolic mathematics, see the [main Pocket-LLM README](../README.md).  
For details on the silicon architecture, see the [OCCP Hardware Repository](https://github.com/mathcode220-math/open-cognitive-core).

---

## License

This compiler is part of the Pocket-LLM project and is licensed under the **CERN Open Hardware Licence v2 — Weakly Reciprocal (CERN-OHL-W)**.
