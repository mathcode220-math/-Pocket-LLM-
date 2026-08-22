"""
Pocket-LLM: Hyperbolic Coral Compiler for Edge AI
==================================================

An open-source AI compilation framework designed to compress heavy Large Language Models (LLMs)
into pocket-sized devices using Non-Euclidean Coral Mathematics.
"""

__version__ = "1.0.0"
__author__ = "Open Cognitive Core Project (OCCP)"
__license__ = "CERN-OHL-W"

from .hyperbolic_quantizer import HyperbolicCoralQuantizer
from .projection_engine import HyperbolicCoralCompiler
from .weight_exporter import HyperbolicCoralExporter

__all__ = [
    "HyperbolicCoralQuantizer",
    "HyperbolicCoralCompiler",
    "HyperbolicCoralExporter",
]