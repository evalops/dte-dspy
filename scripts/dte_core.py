"""
Disagreement-Triggered Escalation (DTE) Core Implementation

This module implements the core DTE logic where two heterogeneous verifiers
(A and B) evaluate claims, and disagreement triggers escalation to a referee (R).
"""

import dspy
from typing import Dict, Any, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Result from a verifier including prediction and confidence."""
    prediction: int  # 0 or 1 for binary classification
    confidence: float  # confidence score [0, 1]
    reasoning: str = ""  # optional reasoning text


@dataclass
class DTEResult:
    """Complete DTE evaluation result."""
    final_prediction: int
    escalated: bool
    verifier_a_result: VerificationResult
    verifier_b_result: VerificationResult
    referee_result: Optional[VerificationResult] = None
    gamma_threshold: float = 0.0


class VerifyClaim(dspy.Signature):
    """Decide if a claim is factually correct. Output strictly 'yes' or 'no'."""
    claim = dspy.InputField(desc="The factual claim to verify")
    verdict = dspy.OutputField(desc="either 'yes' or 'no'")


class DTESystem:
    """
    Disagreement-Triggered Escalation System
    
    Uses two heterogeneous verifiers and escalates to a referee when:
    1. Verifiers disagree, OR
    2. Both verifiers agree but confidence is below threshold
    """
    
    def __init__(self, 
                 verifier_a_lm: dspy.LM,
                 verifier_b_lm: dspy.LM, 
                 referee_lm: dspy.LM,
                 default_gamma: float = 0.7):
        """
        Initialize DTE system with three language models.
        
        Args:
            verifier_a_lm: Language model for verifier A
            verifier_b_lm: Language model for verifier B  
            referee_lm: Language model for referee R
            default_gamma: Default confidence threshold for escalation
        """
        self.verifier_a = dspy.ChainOfThought(VerifyClaim)
        self.verifier_b = dspy.ChainOfThought(VerifyClaim)
        self.referee = dspy.ChainOfThought(VerifyClaim)
        
        self.verifier_a_lm = verifier_a_lm
        self.verifier_b_lm = verifier_b_lm
        self.referee_lm = referee_lm
        
        self.default_gamma = default_gamma
        
        # Metrics tracking
        self.reset_metrics()
    
    def reset_metrics(self):
        """Reset all metrics counters."""
        self.metrics = {
            'total_evaluations': 0,
            'escalations': 0,
            'agreements': 0,
            'disagreements': 0,
            'high_confidence_agreements': 0,
            'low_confidence_agreements': 0,
            'verifier_a_calls': 0,
            'verifier_b_calls': 0,
            'referee_calls': 0
        }
    
    def _parse_verifier_output(self, output) -> VerificationResult:
        """Parse verifier output into VerificationResult."""
        try:
            # Get the verdict and normalize it
            raw_verdict = (output.verdict or "").strip().lower()
            
            if "yes" in raw_verdict and "no" not in raw_verdict:
                prediction = 1
                confidence = 0.8  # Assume reasonable confidence for clear answers
            elif "no" in raw_verdict and "yes" not in raw_verdict:
                prediction = 0 
                confidence = 0.8
            else:
                # Ambiguous response - log and choose randomly
                logger.warning(f"Ambiguous verdict '{output.verdict}' - defaulting to 'no'")
                prediction = 0
                confidence = 0.5
            
            reasoning = str(getattr(output, 'reasoning', ''))
            
            return VerificationResult(prediction, confidence, reasoning)
        except (ValueError, AttributeError) as e:
            logger.warning(f"Error parsing verifier output: {e}")
            return VerificationResult(0, 0.5, "Parse error")
    
    def evaluate_claim(self, claim: str, gamma: Optional[float] = None) -> DTEResult:
        """
        Evaluate a claim using the DTE protocol.
        
        Args:
            claim: The factual claim to verify
            gamma: Confidence threshold (uses default if None)
            
        Returns:
            DTEResult containing the evaluation outcome
        """
        if gamma is None:
            gamma = self.default_gamma
            
        self.metrics['total_evaluations'] += 1
        
        # Get predictions from both verifiers
        with dspy.context(lm=self.verifier_a_lm):
            a_output = self.verifier_a(claim=claim)
            self.metrics['verifier_a_calls'] += 1
            
        with dspy.context(lm=self.verifier_b_lm):
            b_output = self.verifier_b(claim=claim)
            self.metrics['verifier_b_calls'] += 1
        
        a_result = self._parse_verifier_output(a_output)
        b_result = self._parse_verifier_output(b_output)
        
        # Check for agreement
        verifiers_agree = (a_result.prediction == b_result.prediction)
        min_confidence = min(a_result.confidence, b_result.confidence)
        
        if verifiers_agree:
            self.metrics['agreements'] += 1
            if min_confidence >= gamma:
                self.metrics['high_confidence_agreements'] += 1
                # High confidence agreement - accept consensus
                return DTEResult(
                    final_prediction=a_result.prediction,
                    escalated=False,
                    verifier_a_result=a_result,
                    verifier_b_result=b_result,
                    gamma_threshold=gamma
                )
            else:
                self.metrics['low_confidence_agreements'] += 1
        else:
            self.metrics['disagreements'] += 1
        
        # Escalate to referee
        self.metrics['escalations'] += 1
        self.metrics['referee_calls'] += 1
        
        with dspy.context(lm=self.referee_lm):
            r_output = self.referee(claim=claim)
            
        r_result = self._parse_verifier_output(r_output)
        
        return DTEResult(
            final_prediction=r_result.prediction,
            escalated=True,
            verifier_a_result=a_result,
            verifier_b_result=b_result,
            referee_result=r_result,
            gamma_threshold=gamma
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics with computed rates."""
        metrics = self.metrics.copy()
        
        if metrics['total_evaluations'] > 0:
            metrics['escalation_rate'] = metrics['escalations'] / metrics['total_evaluations']
            metrics['disagreement_rate'] = metrics['disagreements'] / metrics['total_evaluations']
            metrics['agreement_rate'] = metrics['agreements'] / metrics['total_evaluations']
        
        if metrics['agreements'] > 0:
            metrics['high_conf_agreement_rate'] = metrics['high_confidence_agreements'] / metrics['agreements']
            
        # Cost metrics (total model calls)
        metrics['total_model_calls'] = (metrics['verifier_a_calls'] + 
                                      metrics['verifier_b_calls'] + 
                                      metrics['referee_calls'])
        
        if metrics['total_evaluations'] > 0:
            metrics['avg_calls_per_evaluation'] = metrics['total_model_calls'] / metrics['total_evaluations']
            
        return metrics


def create_simple_test_models():
    """Create simple test models for demonstration using mock/dummy models."""
    # Create dummy models that will work for testing structure
    # In practice, these would be real LM instances
    
    import os
    
    # Try to use OpenAI if available, otherwise create dummy models
    if os.getenv("OPENAI_API_KEY"):
        model_a = dspy.LM(model="gpt-3.5-turbo", temperature=0.1)
        model_b = dspy.LM(model="gpt-3.5-turbo", temperature=0.9)
        referee = dspy.LM(model="gpt-4", temperature=0.0)
    else:
        # Use dummy models for testing structure
        model_a = dspy.LM(model="dummy")
        model_b = dspy.LM(model="dummy") 
        referee = dspy.LM(model="dummy")
    
    return model_a, model_b, referee


if __name__ == "__main__":
    # Example usage
    model_a, model_b, referee = create_simple_test_models()
    dte = DTESystem(model_a, model_b, referee, default_gamma=0.7)
    
    test_claims = [
        "The capital of France is Paris.",
        "The Earth is flat.",
        "Python was created by Guido van Rossum.",
        "The moon is made of cheese."
    ]
    
    for claim in test_claims:
        result = dte.evaluate_claim(claim)
        print(f"Claim: {claim}")
        print(f"Final prediction: {result.final_prediction}")
        print(f"Escalated: {result.escalated}")
        print(f"Gamma: {result.gamma_threshold}")
        print("---")
    
    print("Final metrics:", dte.get_metrics())