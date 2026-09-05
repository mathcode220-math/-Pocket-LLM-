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
