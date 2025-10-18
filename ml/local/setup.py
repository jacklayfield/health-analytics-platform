#!/usr/bin/env python3
"""
Setup script for the Health Analytics ML Pipeline.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = []
with open("requirements.txt", "r") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="health-analytics-ml-pipeline",
    version="1.0.0",
    author="Health Analytics Team",
    author_email="team@healthanalytics.com",
    description="Production-ready ML pipeline for health analytics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/health-analytics-platform",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.12.0",
            "black>=23.0.0",
            "flake8>=6.1.0",
            "mypy>=1.7.0",
            "pre-commit>=3.5.0",
        ],
        "mlflow": [
            "mlflow>=2.12.0",
        ],
        "monitoring": [
            "great-expectations>=0.18.0",
            "prometheus-client>=0.19.0",
        ],
        "serving": [
            "fastapi>=0.104.0",
            "uvicorn[standard]>=0.24.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "health-ml=cli.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["config/*.yaml", "*.yaml"],
    },
    zip_safe=False,
)

