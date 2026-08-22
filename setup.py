"""
Pocket-LLM: Hyperbolic Coral Compiler for Edge AI
==================================================

An open-source AI compilation framework designed to compress heavy Large Language Models (LLMs)
into pocket-sized devices using Non-Euclidean Coral Mathematics.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="pocket-llm",
    version="1.0.0",
    author="Open Cognitive Core Project (OCCP)",
    description="Hyperbolic Coral Compiler for Edge AI - Compress LLMs using Non-Euclidean Geometry",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/occp/pocket-llm",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: CERN Open Hardware Licence v2 (CERN-OHL-W)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Compilers",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "pocket-compile=pocket_llm.compiler.projection_engine:main",
            "pocket-quantize=pocket_llm.compiler.hyperbolic_quantizer:main",
            "pocket-export=pocket_llm.compiler.weight_exporter:main",
        ],
    },
    include_package_data=True,
    package_data={
        "pocket_llm": ["runtime/include/*.h", "runtime/src/*.cpp"],
    },
)
