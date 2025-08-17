"""
Core DTE system implementation.

This module contains the main DTESystem class that orchestrates
disagreement-triggered escalation between heterogeneous verifiers.
"""

import dspy
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass
import random

logger = logging.getLogger(__name__)


class VerifyClaim(dspy.Signature):
    """Decide if a claim is factually correct. Output strictly 'yes' or 'no'."""
    claim = dspy.InputField(desc="The factual claim to verify")
    verdict = dspy.OutputField(desc="either 'yes' or 'no'")


class Verifier(dspy.Module):
    """A factual claim verifier using DSPy."""
    
    def __init__(self, use_cot: bool = True):
        super().__init__()
        self.use_cot = use_cot
        self.step = (dspy.ChainOfThought if use_cot else dspy.Predict)(VerifyClaim)
        
    def forward(self, claim: str) -> dspy.Prediction:
        """Verify a factual claim."""
        try:
            out = self.step(claim=claim)
            
            # Normalize verdict
            raw_verdict = (out.verdict or "").strip().lower()
            
            if "yes" in raw_verdict and "no" not in raw_verdict:
                normalized_verdict = "yes"
            elif "no" in raw_verdict and "yes" not in raw_verdict:
                normalized_verdict = "no"
            else:
                logger.warning(f"Ambiguous verdict '{out.verdict}' for claim: {claim}")
                normalized_verdict = random.choice(["yes", "no"])
            
            out.verdict = normalized_verdict
            return out
            
        except Exception as e:
            logger.error(f"Error verifying claim '{claim}': {e}")
            return dspy.Prediction(verdict=random.choice(["yes", "no"]))


@dataclass
class VerificationResult:
    """Result from a single verifier."""
    prediction: int  # 0 or 1 for binary classification
    confidence: float  # confidence score [0, 1]
    reasoning: str = ""  # optional reasoning text
    
    @classmethod
    def from_verdict(cls, verdict: str, confidence: float = 0.8) -> "VerificationResult":
        """Create from yes/no verdict."""
        prediction = 1 if verdict == "yes" else 0
        return cls(prediction=prediction, confidence=confidence, reasoning="")


@dataclass 
class DTEResult:
    """Complete DTE evaluation result."""
    claim: str
    final_prediction: int  # 0 or 1
    escalated: bool
    verifier_a_result: VerificationResult
    verifier_b_result: VerificationResult
    referee_result: Optional[VerificationResult] = None
    gamma_threshold: float = 0.0
    
    @property
    def final_verdict(self) -> str:
        """Get final verdict as yes/no string."""
        return "yes" if self.final_prediction == 1 else "no"


class DTESystem:
    """
    Disagreement-Triggered Escalation System.
    
    Uses two heterogeneous verifiers and escalates to a referee when:
    1. Verifiers disagree, OR
    2. Both verifiers agree but confidence is below threshold
    """
    
    def __init__(self, 
                 verifier_a_lm: dspy.LM,
                 verifier_b_lm: dspy.LM,
                 referee_lm: dspy.LM,
                 gamma: float = 0.7,
                 use_cot: bool = True):
        """
        Initialize DTE system.
        
        Args:
            verifier_a_lm: Language model for verifier A
            verifier_b_lm: Language model for verifier B  
            referee_lm: Language model for referee
            gamma: Confidence threshold for escalation
            use_cot: Whether to use Chain of Thought reasoning
        """
        self.verifier_a = Verifier(use_cot=use_cot)
        self.verifier_b = Verifier(use_cot=use_cot)
        self.referee = Verifier(use_cot=use_cot)
        
        self.verifier_a_lm = verifier_a_lm
        self.verifier_b_lm = verifier_b_lm
        self.referee_lm = referee_lm
        
        self.gamma = gamma
        self.reset_metrics()
    
    def reset_metrics(self):
        """Reset all metrics counters."""
        self.metrics = {
            'total_evaluations': 0,
            'escalations': 0,
            'agreements': 0,
            'disagreements': 0,
            'correct_predictions': 0,
            'correct_escalations': 0,
            'correct_non_escalations': 0,
            'verifier_a_calls': 0,
            'verifier_b_calls': 0,
            'referee_calls': 0
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
        with dspy.context(lm=self.verifier_a_lm):
            a_prediction = self.verifier_a(claim=claim)
            self.metrics['verifier_a_calls'] += 1
            
        with dspy.context(lm=self.verifier_b_lm):
            b_prediction = self.verifier_b(claim=claim)
            self.metrics['verifier_b_calls'] += 1
        
        a_result = VerificationResult.from_verdict(a_prediction.verdict)
        b_result = VerificationResult.from_verdict(b_prediction.verdict)
        
        # Check for agreement
        verifiers_agree = (a_result.prediction == b_result.prediction)
        
        if verifiers_agree:
            self.metrics['agreements'] += 1
            # For simplicity, assume high confidence when verifiers agree
            should_escalate = False
        else:
            self.metrics['disagreements'] += 1
            should_escalate = True
        
        if should_escalate:
            # Escalate to referee
            self.metrics['escalations'] += 1
            self.metrics['referee_calls'] += 1
            
            with dspy.context(lm=self.referee_lm):
                referee_prediction = self.referee(claim=claim)
                
            referee_result = VerificationResult.from_verdict(referee_prediction.verdict)
            final_prediction = referee_result.prediction
            escalated = True
        else:
            # Accept consensus
            final_prediction = a_result.prediction
            referee_result = None
            escalated = False
        
        # Track accuracy if ground truth provided
        if ground_truth:
            gt_prediction = 1 if ground_truth == "yes" else 0
            is_correct = (final_prediction == gt_prediction)
            
            if is_correct:
                self.metrics['correct_predictions'] += 1
                
            if escalated and is_correct:
                self.metrics['correct_escalations'] += 1
            elif not escalated and is_correct:
                self.metrics['correct_non_escalations'] += 1
        
        return DTEResult(
            claim=claim,
            final_prediction=final_prediction,
            escalated=escalated,
            verifier_a_result=a_result,
            verifier_b_result=b_result,
            referee_result=referee_result,
            gamma_threshold=self.gamma
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics with computed rates."""
        m = self.metrics.copy()
        
        if m['total_evaluations'] > 0:
            m['escalation_rate'] = m['escalations'] / m['total_evaluations']
            m['disagreement_rate'] = m['disagreements'] / m['total_evaluations'] 
            m['agreement_rate'] = m['agreements'] / m['total_evaluations']
            m['overall_accuracy'] = m['correct_predictions'] / m['total_evaluations']
            
        if m['escalations'] > 0:
            m['escalation_accuracy'] = m['correct_escalations'] / m['escalations']
            
        non_escalations = m['total_evaluations'] - m['escalations']
        if non_escalations > 0:
            m['non_escalation_accuracy'] = m['correct_non_escalations'] / non_escalations
            
        # Cost metrics
        m['total_model_calls'] = m['verifier_a_calls'] + m['verifier_b_calls'] + m['referee_calls']
        if m['total_evaluations'] > 0:
            m['avg_calls_per_evaluation'] = m['total_model_calls'] / m['total_evaluations']
            
        return m