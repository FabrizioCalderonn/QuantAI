"""
Setup script para el sistema de trading cuantitativo.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Leer README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Leer requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file, 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="quantia-trading-system",
    version="1.0.0",
    author="QuantIA Team",
    author_email="team@quantia.com",
    description="Sistema de Trading Cuantitativo Institucional-Grade",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/quantia/trading-system",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-asyncio>=0.21.1",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "mypy>=1.5.1",
        ],
        "gpu": [
            "cupy-cuda11x>=12.2.0",
            "cuml>=23.08.00",
        ],
        "ml": [
            "torch>=2.0.1",
            "transformers>=4.32.1",
        ]
    },
    entry_points={
        "console_scripts": [
            "quantia=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["configs/*.yaml", "*.md"],
    },
    zip_safe=False,
)

