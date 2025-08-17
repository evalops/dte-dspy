"""
Configuration management for DTE experiments.

This module provides configuration classes and utilities for managing
DTE system parameters, model configurations, and experiment settings.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import dspy


@dataclass
class DTEConfig:
    """Configuration for DTE system experiments."""
    
    # Model configuration
    verifier_a_model: str = "llama3.2:latest"
    verifier_b_model: str = "llama3:8b"
    referee_model: str = "llama3.1:8b"
    api_base: str = "http://localhost:11434"
    
    # Model parameters
    temperature_a: float = 0.1  # Low temp for consistency
    temperature_b: float = 0.9  # High temp for diversity
    temperature_referee: float = 0.0  # Deterministic referee
    max_tokens: int = 512
    
    # DTE parameters
    gamma: float = 0.7  # Confidence threshold for escalation
    use_cot: bool = True  # Use Chain of Thought reasoning
    
    # Experiment parameters
    num_trials: int = 1  # Number of runs per configuration
    random_seed: Optional[int] = None
    
    # Evaluation parameters
    dataset_name: str = "default"
    dataset_size: Optional[int] = None  # Limit dataset size
    
    # Output configuration
    output_dir: str = "results"
    save_detailed_results: bool = True
    verbose: bool = False
    
    # Caching
    enable_cache: bool = True
    cache_dir: str = ".cache"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.gamma < 0 or self.gamma > 1:
            raise ValueError("gamma must be between 0 and 1")
            
        if self.temperature_a < 0 or self.temperature_b < 0 or self.temperature_referee < 0:
            raise ValueError("temperatures must be non-negative")
    
    @classmethod
    def from_env(cls) -> "DTEConfig":
        """Create configuration from environment variables."""
        return cls(
            verifier_a_model=os.getenv("DTE_VERIFIER_A_MODEL", cls.verifier_a_model),
            verifier_b_model=os.getenv("DTE_VERIFIER_B_MODEL", cls.verifier_b_model),
            referee_model=os.getenv("DTE_REFEREE_MODEL", cls.referee_model),
            api_base=os.getenv("DTE_API_BASE", cls.api_base),
            gamma=float(os.getenv("DTE_GAMMA", cls.gamma)),
            use_cot=os.getenv("DTE_USE_COT", "true").lower() == "true",
            random_seed=int(os.getenv("DTE_RANDOM_SEED")) if os.getenv("DTE_RANDOM_SEED") else None,
            output_dir=os.getenv("DTE_OUTPUT_DIR", cls.output_dir),
            verbose=os.getenv("DTE_VERBOSE", "false").lower() == "true",
        )
    
    def create_language_models(self) -> tuple[dspy.LM, dspy.LM, dspy.LM]:
        """Create configured language model instances."""
        verifier_a_lm = dspy.LM(
            model=f"ollama/{self.verifier_a_model}",
            api_base=self.api_base,
            temperature=self.temperature_a,
            max_tokens=self.max_tokens
        )
        
        verifier_b_lm = dspy.LM(
            model=f"ollama/{self.verifier_b_model}",
            api_base=self.api_base,
            temperature=self.temperature_b,
            max_tokens=self.max_tokens
        )
        
        referee_lm = dspy.LM(
            model=f"ollama/{self.referee_model}",
            api_base=self.api_base,
            temperature=self.temperature_referee,
            max_tokens=self.max_tokens
        )
        
        return verifier_a_lm, verifier_b_lm, referee_lm
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "verifier_a_model": self.verifier_a_model,
            "verifier_b_model": self.verifier_b_model,
            "referee_model": self.referee_model,
            "api_base": self.api_base,
            "gamma": self.gamma,
            "use_cot": self.use_cot,
            "temperature_a": self.temperature_a,
            "temperature_b": self.temperature_b,
            "temperature_referee": self.temperature_referee,
            "max_tokens": self.max_tokens,
        }


def setup_logging(level: str = "INFO") -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Set DSPy logging to WARNING to reduce noise
    logging.getLogger("dspy").setLevel(logging.WARNING)


def validate_ollama_connection(api_base: str = "http://localhost:11434") -> bool:
    """Validate that Ollama is running and accessible."""
    import requests
    
    try:
        response = requests.get(f"{api_base}/api/tags", timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False


def list_available_models(api_base: str = "http://localhost:11434") -> list[str]:
    """List available Ollama models."""
    import requests
    
    try:
        response = requests.get(f"{api_base}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
        return []
    except requests.RequestException:
        return []