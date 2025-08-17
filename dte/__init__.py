"""
Disagreement-Triggered Escalation (DTE) Framework

A framework for studying disagreement-triggered escalation in multi-agent fact verification.
Instead of optimizing for agreement (which can amplify errors), DTE treats disagreement
as a signal to escalate to stronger verification methods.

Key Components:
- Core: DTESystem for orchestrating verifier disagreements
- Config: Configuration management for experiments  
- Evaluation: Comprehensive evaluation harness
- Data: Dataset loading and management
- Metrics: Performance and cost tracking
"""

__version__ = "0.1.0"
__author__ = "DTE Research Team"

from .core import DTESystem, DTEResult, VerificationResult
from .config import DTEConfig
from .evaluation import DTEEvaluator
from .data import load_dataset, create_test_dataset
from .metrics import calculate_metrics

__all__ = [
    "DTESystem",
    "DTEResult", 
    "VerificationResult",
    "DTEConfig",
    "DTEEvaluator",
    "load_dataset",
    "create_test_dataset",
    "calculate_metrics",
]