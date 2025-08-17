from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip() for line in fh if line.strip() and not line.startswith("#")
    ]

setup(
    name="dte-dspy",
    version="0.1.0",
    author="DTE Research Team",
    author_email="dte@research.dev",
    description="Disagreement-Triggered Escalation framework using DSPy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/evalops/dte-dspy",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "ruff>=0.1.0",
            "black>=22.0.0", 
            "isort>=5.0.0",
            "mypy>=1.0.0",
        ],
        "analysis": [
            "scipy>=1.9.0",
            "plotly>=5.0.0",
            "jupyter>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "dte=dte.main:main",
        ],
    },
    keywords="disagreement escalation dspy llm verification fact-checking",
    project_urls={
        "Bug Reports": "https://github.com/evalops/dte-dspy/issues",
        "Source": "https://github.com/evalops/dte-dspy",
        "Documentation": "https://github.com/evalops/dte-dspy",
    },
)