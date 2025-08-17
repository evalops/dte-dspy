"""
Ollama Integration for DTE System

This module provides integration with Ollama for running local language models
as verifiers and referees in the DTE system.
"""

import dspy
import requests
import json
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class OllamaLM(dspy.LM):
    """DSPy Language Model adapter for Ollama."""
    
    def __init__(self, 
                 model: str, 
                 base_url: str = "http://localhost:11434",
                 temperature: float = 0.7,
                 max_tokens: int = 1000,
                 **kwargs):
        """
        Initialize Ollama language model.
        
        Args:
            model: Ollama model name (e.g., "llama2", "mistral")
            base_url: Ollama server URL
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
        """
        super().__init__(model=model, **kwargs)
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._verify_connection()
    
    def _verify_connection(self):
        """Verify Ollama server is accessible."""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = [m['name'] for m in response.json().get('models', [])]
                if self.model not in models:
                    logger.warning(f"Model {self.model} not found in Ollama. Available: {models}")
            else:
                logger.error(f"Cannot connect to Ollama at {self.base_url}")
        except requests.RequestException as e:
            logger.error(f"Ollama connection error: {e}")
    
    def basic_request(self, prompt: str, **kwargs) -> str:
        """Make a basic request to Ollama."""
        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.temperature),
                "num_predict": kwargs.get("max_tokens", self.max_tokens)
            }
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=data,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except requests.RequestException as e:
            logger.error(f"Ollama request failed: {e}")
            return ""


def setup_ollama_models(base_url: str = "http://localhost:11434") -> Dict[str, OllamaLM]:
    """
    Set up standard Ollama models for DTE system.
    
    Args:
        base_url: Ollama server URL
        
    Returns:
        Dictionary with configured models
    """
    models = {
        # Verifier A: Lower temperature for consistency
        "verifier_a": OllamaLM(
            model="llama2:7b", 
            base_url=base_url,
            temperature=0.1,
            max_tokens=500
        ),
        
        # Verifier B: Higher temperature for diversity
        "verifier_b": OllamaLM(
            model="mistral:7b",
            base_url=base_url, 
            temperature=0.9,
            max_tokens=500
        ),
        
        # Judge: More powerful model with low temperature
        "judge": OllamaLM(
            model="llama2:13b",
            base_url=base_url,
            temperature=0.0,
            max_tokens=500
        )
    }
    
    return models


def get_available_ollama_models(base_url: str = "http://localhost:11434") -> List[str]:
    """Get list of available models from Ollama server."""
    try:
        response = requests.get(f"{base_url.rstrip('/')}/api/tags")
        if response.status_code == 200:
            return [m['name'] for m in response.json().get('models', [])]
        else:
            logger.error(f"Failed to get models from Ollama: {response.status_code}")
            return []
    except requests.RequestException as e:
        logger.error(f"Error connecting to Ollama: {e}")
        return []


def pull_ollama_model(model: str, base_url: str = "http://localhost:11434") -> bool:
    """Pull a model from Ollama registry."""
    try:
        response = requests.post(
            f"{base_url.rstrip('/')}/api/pull",
            json={"name": model},
            stream=True,
            timeout=300
        )
        
        if response.status_code == 200:
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if data.get("status"):
                        print(f"Pulling {model}: {data['status']}")
                    if data.get("error"):
                        logger.error(f"Error pulling {model}: {data['error']}")
                        return False
            return True
        else:
            logger.error(f"Failed to pull model {model}: {response.status_code}")
            return False
            
    except requests.RequestException as e:
        logger.error(f"Error pulling model {model}: {e}")
        return False


class OllamaModelManager:
    """Utility class for managing Ollama models."""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip('/')
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all available models with details."""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                return response.json().get('models', [])
            return []
        except requests.RequestException as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    def model_exists(self, model: str) -> bool:
        """Check if a model exists locally."""
        models = [m['name'] for m in self.list_models()]
        return model in models
    
    def ensure_models(self, required_models: List[str]) -> bool:
        """Ensure all required models are available, pulling if necessary."""
        available = [m['name'] for m in self.list_models()]
        missing = [m for m in required_models if m not in available]
        
        if not missing:
            logger.info("All required models are available")
            return True
        
        logger.info(f"Pulling missing models: {missing}")
        for model in missing:
            logger.info(f"Pulling {model}...")
            if not pull_ollama_model(model, self.base_url):
                logger.error(f"Failed to pull {model}")
                return False
            logger.info(f"Successfully pulled {model}")
        
        return True
    
    def get_model_info(self, model: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific model."""
        models = self.list_models()
        for m in models:
            if m['name'] == model:
                return m
        return None


def create_ollama_dte_system(
    verifier_a_model: str = "llama2:7b",
    verifier_b_model: str = "mistral:7b", 
    judge_model: str = "llama2:13b",
    base_url: str = "http://localhost:11434",
    ensure_models: bool = True
):
    """
    Create a DTE system using Ollama models.
    
    Args:
        verifier_a_model: Model name for verifier A
        verifier_b_model: Model name for verifier B
        judge_model: Model name for judge
        base_url: Ollama server URL
        ensure_models: Whether to automatically pull missing models
        
    Returns:
        Configured DTESystem instance
    """
    from scripts.dte_core import DTESystem
    
    manager = OllamaModelManager(base_url)
    
    if ensure_models:
        required = [verifier_a_model, verifier_b_model, judge_model]
        if not manager.ensure_models(required):
            raise RuntimeError("Failed to ensure all required models are available")
    
    # Create model instances
    verifier_a = OllamaLM(verifier_a_model, base_url, temperature=0.1)
    verifier_b = OllamaLM(verifier_b_model, base_url, temperature=0.9)
    judge = OllamaLM(judge_model, base_url, temperature=0.0)
    
    return DTESystem(verifier_a, verifier_b, judge)


if __name__ == "__main__":
    # Example usage
    print("Available Ollama models:")
    models = get_available_ollama_models()
    for model in models:
        print(f"  - {model}")
    
    if models:
        print("\nCreating DTE system with Ollama models...")
        try:
            dte_system = create_ollama_dte_system()
            print("DTE system created successfully!")
            
            # Test with a simple claim
            test_claim = "The capital of France is Paris."
            result = dte_system.evaluate_claim(test_claim)
            print(f"\nTest evaluation:")
            print(f"Claim: {test_claim}")
            print(f"Result: {result.final_prediction}")
            print(f"Escalated: {result.escalated}")
            
        except Exception as e:
            print(f"Error creating DTE system: {e}")
    else:
        print("No Ollama models found. Please install Ollama and pull some models first.")
        print("Example: ollama pull llama2:7b")