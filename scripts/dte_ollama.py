#!/usr/bin/env python3
"""
Real Ollama-based DTE implementation using DSPy.

This implements Disagreement-Triggered Escalation with actual Ollama models,
following the patterns from the working folie-a-deux repository.
"""

import dspy
import requests
import json
import logging
import random
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VerifyClaim(dspy.Signature):
    """Decide if a claim is factually correct. Output strictly 'yes' or 'no'."""
    claim = dspy.InputField(desc="The factual claim to verify")
    verdict = dspy.OutputField(desc="either 'yes' or 'no'")


class Verifier(dspy.Module):
    """A factual claim verifier using DSPy with Ollama."""
    
    def __init__(self, use_cot: bool = True):
        super().__init__()
        self.use_cot = use_cot
        self.step = (dspy.ChainOfThought if use_cot else dspy.Predict)(VerifyClaim)
        
    def forward(self, claim: str) -> dspy.Prediction:
        """Verify a factual claim."""
        try:
            out = self.step(claim=claim)  # type: ignore
            
            # Normalize verdict
            raw_verdict = (out.verdict or "").strip().lower()  # type: ignore
            
            if "yes" in raw_verdict and "no" not in raw_verdict:
                normalized_verdict = "yes"
            elif "no" in raw_verdict and "yes" not in raw_verdict:
                normalized_verdict = "no"
            else:
                logger.warning(f"Ambiguous verdict '{out.verdict}' for claim: {claim}")  # type: ignore
                normalized_verdict = random.choice(["yes", "no"])
            
            out.verdict = normalized_verdict  # type: ignore
            return out  # type: ignore
            
        except Exception as e:
            logger.error(f"Error verifying claim '{claim}': {e}")
            return dspy.Prediction(verdict=random.choice(["yes", "no"]))


@dataclass 
class DTEResult:
    """Result from DTE evaluation."""
    claim: str
    final_verdict: str  # "yes" or "no"
    escalated: bool
    verifier_a_verdict: str
    verifier_b_verdict: str  
    judge_verdict: Optional[str] = None
    gamma_threshold: float = 0.0


class DTESystem:
    """Disagreement-Triggered Escalation system with Ollama models."""
    
    def __init__(self, 
                 verifier_a_model: str = "llama2:7b",
                 verifier_b_model: str = "mistral:7b", 
                 judge_model: str = "llama2:13b",
                 base_url: str = "http://localhost:11434",
                 gamma: float = 0.7):
        """
        Initialize DTE system with Ollama models.
        
        Args:
            verifier_a_model: Model for verifier A
            verifier_b_model: Model for verifier B (should be different from A)
            judge_model: Model for judge (typically stronger)
            base_url: Ollama server URL
            gamma: Confidence threshold for escalation
        """
        self.gamma = gamma
        self.base_url = base_url
        
        # Initialize verifiers and judge
        self.verifier_a = Verifier(use_cot=True)
        self.verifier_b = Verifier(use_cot=True) 
        self.judge = Verifier(use_cot=True)
        
        # Set up Ollama language models
        self.lm_a = dspy.LM(model=f"ollama/{verifier_a_model}", api_base=base_url, temperature=0.1)
        self.lm_b = dspy.LM(model=f"ollama/{verifier_b_model}", api_base=base_url, temperature=0.9)
        self.lm_judge = dspy.LM(model=f"ollama/{judge_model}", api_base=base_url, temperature=0.0)
        
        # Verify models are available
        self._verify_models([verifier_a_model, verifier_b_model, judge_model])
        
        # Metrics tracking
        self.metrics: dict[str, int | float] = {}
        self.reset_metrics()
    
    def _verify_models(self, model_names: List[str]):
        """Verify that required models are available in Ollama."""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                available = [m['name'] for m in response.json().get('models', [])]
                missing = [m for m in model_names if m not in available]
                if missing:
                    logger.warning(f"Missing models: {missing}")
                    logger.info(f"Available models: {available}")
                    logger.info("Pull missing models with: ollama pull <model_name>")
            else:
                logger.error(f"Cannot connect to Ollama at {self.base_url}")
        except requests.RequestException as e:
            logger.error(f"Ollama connection error: {e}")
    
    def reset_metrics(self):
        """Reset metrics counters."""
        self.metrics: dict[str, int | float] = {
            'total_evaluations': 0,
            'escalations': 0,
            'agreements': 0,
            'disagreements': 0,
            'correct_escalations': 0,
            'correct_non_escalations': 0,
            'verifier_a_calls': 0,
            'verifier_b_calls': 0,
            'judge_calls': 0
        }
    
    def evaluate_claim(self, claim: str, ground_truth: Optional[str] = None) -> DTEResult:
        """
        Evaluate a claim using DTE protocol.
        
        Args:
            claim: The claim to evaluate
            ground_truth: Optional ground truth ("yes"/"no") for metrics
            
        Returns:
            DTEResult with evaluation outcome
        """
        self.metrics['total_evaluations'] += 1
        
        # Get verdicts from both verifiers
        with dspy.context(lm=self.lm_a):
            a_prediction = self.verifier_a(claim=claim)  # type: ignore
            self.metrics['verifier_a_calls'] += 1
            
        with dspy.context(lm=self.lm_b):
            b_prediction = self.verifier_b(claim=claim)  # type: ignore
            self.metrics['verifier_b_calls'] += 1
        
        a_verdict = a_prediction.verdict  # type: ignore
        b_verdict = b_prediction.verdict  # type: ignore
        
        # Check for agreement
        verifiers_agree = (a_verdict == b_verdict)
        
        if verifiers_agree:
            self.metrics['agreements'] += 1
            # For simplicity, assume high confidence when verifiers agree
            # In practice, you might extract confidence from the reasoning
            should_escalate = False  # High confidence agreement
        else:
            self.metrics['disagreements'] += 1
            should_escalate = True  # Disagreement triggers escalation
        
        if should_escalate:
            # Escalate to judge
            self.metrics['escalations'] += 1
            self.metrics['judge_calls'] += 1
            
            with dspy.context(lm=self.lm_judge):
                judge_prediction = self.judge(claim=claim)  # type: ignore
                
            final_verdict = judge_prediction.verdict  # type: ignore
            judge_verdict = judge_prediction.verdict  # type: ignore
            escalated = True
        else:
            # Accept consensus
            final_verdict = a_verdict  # They agree, so either works
            judge_verdict = None
            escalated = False
        
        # Track accuracy if ground truth provided
        if ground_truth:
            is_correct = (final_verdict == ground_truth)
            if escalated:
                if is_correct:
                    self.metrics['correct_escalations'] += 1
            else:
                if is_correct:
                    self.metrics['correct_non_escalations'] += 1
        
        return DTEResult(
            claim=claim,
            final_verdict=final_verdict,
            escalated=escalated,
            verifier_a_verdict=a_verdict,
            verifier_b_verdict=b_verdict,
            judge_verdict=judge_verdict,
            gamma_threshold=self.gamma
        )
    
    def evaluate_dataset(self, 
                        claims: List[str], 
                        ground_truth: Optional[List[str]] = None,
                        verbose: bool = True) -> List[DTEResult]:
        """
        Evaluate a dataset of claims.
        
        Args:
            claims: List of claims to evaluate
            ground_truth: Optional ground truth labels
            verbose: Whether to show progress
            
        Returns:
            List of DTEResult objects
        """
        results = []
        
        if ground_truth:
            assert len(claims) == len(ground_truth), "Claims and ground truth must have same length"
        
        for i, claim in enumerate(claims):
            if verbose and i % 10 == 0:
                print(f"Evaluating claim {i+1}/{len(claims)}")
                
            gt = ground_truth[i] if ground_truth else None
            result = self.evaluate_claim(claim, gt)
            results.append(result)
            
            if verbose:
                status = "✓" if not result.escalated else "↗"
                print(f"  {status} {claim[:60]}... → {result.final_verdict}")
        
        return results
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics with computed rates."""
        m = self.metrics.copy()
        
        if m['total_evaluations'] > 0:
            m['escalation_rate'] = m['escalations'] / m['total_evaluations']
            m['disagreement_rate'] = m['disagreements'] / m['total_evaluations']
            m['agreement_rate'] = m['agreements'] / m['total_evaluations']
            
        if m['escalations'] > 0:
            m['escalation_accuracy'] = m['correct_escalations'] / m['escalations']
            
        if (m['total_evaluations'] - m['escalations']) > 0:
            m['non_escalation_accuracy'] = m['correct_non_escalations'] / (m['total_evaluations'] - m['escalations'])
            
        # Overall accuracy
        total_correct = m['correct_escalations'] + m['correct_non_escalations']
        if m['total_evaluations'] > 0:
            m['overall_accuracy'] = total_correct / m['total_evaluations']
            
        # Cost metrics
        m['total_model_calls'] = m['verifier_a_calls'] + m['verifier_b_calls'] + m['judge_calls']
        if m['total_evaluations'] > 0:
            m['avg_calls_per_evaluation'] = m['total_model_calls'] / m['total_evaluations']
            
        return m


def create_test_dataset() -> Tuple[List[str], List[str]]:
    """Create a test dataset of factual claims."""
    claims = [
        "The capital of France is Paris.",
        "The capital of France is London.",
        "Water boils at 100 degrees Celsius at sea level.",
        "Water boils at 50 degrees Celsius at sea level.", 
        "The Earth is round.",
        "The Earth is flat.",
        "Python is a programming language.",
        "Python is only a type of snake.",
        "The sun is a star.",
        "The sun is a planet.",
        "Humans have two eyes.",
        "Humans have three eyes.",
        "Gold is a chemical element.",
        "Gold is made of plastic.",
        "Shakespeare wrote Romeo and Juliet.",
        "Shakespeare wrote Harry Potter.",
        "The Pacific Ocean is the largest ocean.",
        "The Arctic Ocean is the largest ocean.",
        "DNA stands for Deoxyribonucleic acid.",
        "DNA stands for Digital Network Access."
    ]
    
    ground_truth = [
        "yes", "no", "yes", "no", "yes", "no", "yes", "no", "yes", "no",
        "yes", "no", "yes", "no", "yes", "no", "yes", "no", "yes", "no"
    ]
    
    return claims, ground_truth


def main():
    """Run DTE system with Ollama."""
    print("=== Disagreement-Triggered Escalation with Ollama ===")
    
    # Check if Ollama is running
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code != 200:
            print("❌ Ollama is not running. Please start it with: ollama serve")
            return 1
    except requests.RequestException:
        print("❌ Cannot connect to Ollama. Please start it with: ollama serve")
        return 1
    
    print("✓ Ollama is running")
    
    # Create DTE system using available models
    try:
        dte = DTESystem(
            verifier_a_model="llama3:8b",           # Use available llama3:8b
            verifier_b_model="llama3.2:latest",    # Different model for diversity
            judge_model="llama3.3:latest",       # Larger model as referee
            gamma=0.7
        )
        print("✓ DTE system initialized")
    except Exception as e:
        print(f"❌ Failed to initialize DTE system: {e}")
        return 1
    
    # Create test dataset
    claims, ground_truth = create_test_dataset()
    print(f"✓ Created test dataset with {len(claims)} claims")
    
    # Evaluate dataset
    print("\n=== Evaluating Claims ===")
    results = dte.evaluate_dataset(claims, ground_truth)
    
    # Show results
    print(f"\n=== Results ===")
    metrics = dte.get_metrics()
    
    print(f"Total evaluations: {metrics['total_evaluations']}")
    print(f"Escalation rate: {metrics.get('escalation_rate', 0):.2%}")
    print(f"Agreement rate: {metrics.get('agreement_rate', 0):.2%}")
    print(f"Overall accuracy: {metrics.get('overall_accuracy', 0):.2%}")
    print(f"Average calls per evaluation: {metrics.get('avg_calls_per_evaluation', 0):.1f}")
    
    if metrics.get('escalation_accuracy') is not None:
        print(f"Escalated case accuracy: {metrics['escalation_accuracy']:.2%}")
    if metrics.get('non_escalation_accuracy') is not None:
        print(f"Non-escalated case accuracy: {metrics['non_escalation_accuracy']:.2%}")
    
    # Show some example results
    print(f"\n=== Example Results ===")
    for i, result in enumerate(results[:5]):
        status = "ESCALATED" if result.escalated else "CONSENSUS"
        print(f"{result.claim[:50]}...")
        print(f"  A: {result.verifier_a_verdict}, B: {result.verifier_b_verdict}")
        if result.escalated:
            print(f"  → Judge: {result.judge_verdict} [{status}]")
        else:
            print(f"  → Final: {result.final_verdict} [{status}]")
        print()
    
    return 0


if __name__ == "__main__":
    exit(main())