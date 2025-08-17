"""
Core DTE system implementation.

This module contains the main DTESystem class that orchestrates
disagreement-triggered escalation between heterogeneous verifiers.
"""

import dspy
import logging
import re
from typing import Optional, Dict, Any
from dataclasses import dataclass
import random

logger = logging.getLogger(__name__)


class VerifyClaim(dspy.Signature):
    """Decide if a claim is factually correct with calibrated confidence."""
    claim = dspy.InputField(desc="The factual claim to verify")
    verdict = dspy.OutputField(desc="either 'yes' or 'no'")
    confidence = dspy.OutputField(desc="numeric confidence score from 0.0 to 1.0")
    reasoning = dspy.OutputField(desc="brief explanation for the verdict")


class Verifier(dspy.Module):
    """A factual claim verifier using DSPy."""
    
    def __init__(self, use_cot: bool = True):
        super().__init__()
        self.use_cot = use_cot
        self.step = (dspy.ChainOfThought if use_cot else dspy.Predict)(VerifyClaim)
        
    def forward(self, claim: str) -> dspy.Prediction:
        """Verify a factual claim with robust error handling."""
        if not claim or not claim.strip():
            logger.warning("Empty claim provided")
            return dspy.Prediction(verdict="no", reasoning="Empty claim")
            
        try:
            out = self.step(claim=claim)
            
            # Validate output structure
            if not hasattr(out, 'verdict') or out.verdict is None:
                logger.warning(f"Model output missing verdict for claim: {claim}")
                return dspy.Prediction(verdict="no", reasoning="Model output error")
            
            # Normalize verdict with better handling
            raw_verdict = str(out.verdict).strip().lower()
            
            # Enhanced verdict parsing
            if any(word in raw_verdict for word in ['yes', 'true', 'correct', 'valid']):
                if any(word in raw_verdict for word in ['no', 'false', 'incorrect', 'invalid']):
                    # Conflicting signals - check which is more prominent
                    yes_count = sum(raw_verdict.count(word) for word in ['yes', 'true', 'correct'])
                    no_count = sum(raw_verdict.count(word) for word in ['no', 'false', 'incorrect'])
                    normalized_verdict = "yes" if yes_count > no_count else "no"
                    logger.warning(f"Conflicting verdict '{out.verdict}' resolved to '{normalized_verdict}' for claim: {claim}")
                else:
                    normalized_verdict = "yes"
            elif any(word in raw_verdict for word in ['no', 'false', 'incorrect', 'invalid']):
                normalized_verdict = "no"
            else:
                logger.warning(f"Ambiguous verdict '{out.verdict}' for claim: {claim}")
                # Use reasoning to make a decision if available
                reasoning = getattr(out, 'reasoning', '').lower()
                if reasoning and any(word in reasoning for word in ['correct', 'true', 'accurate']):
                    normalized_verdict = "yes"
                elif reasoning and any(word in reasoning for word in ['incorrect', 'false', 'wrong']):
                    normalized_verdict = "no"
                else:
                    normalized_verdict = "no"  # Conservative default
            
            out.verdict = normalized_verdict
            return out
            
        except Exception as e:
            logger.error(f"Error verifying claim '{claim}': {e}")
            return dspy.Prediction(verdict="no", reasoning=f"Error: {str(e)}")


@dataclass
class VerificationResult:
    """Result from a single verifier."""
    prediction: int  # 0 or 1 for binary classification
    confidence: float  # confidence score [0, 1]
    reasoning: str = ""  # optional reasoning text
    raw_output: str = ""  # raw model output for debugging
    
    @classmethod
    def from_verdict(cls, verdict: str, reasoning: str = "", raw_output: str = "", structured_confidence: float = None) -> "VerificationResult":
        """Create from yes/no verdict with confidence extraction."""
        prediction = 1 if verdict == "yes" else 0
        confidence = cls._extract_confidence(reasoning, raw_output, verdict, structured_confidence)
        return cls(
            prediction=prediction, 
            confidence=confidence, 
            reasoning=reasoning,
            raw_output=raw_output
        )
    
    @staticmethod
    def _extract_confidence(reasoning: str, raw_output: str, verdict: str, structured_confidence: float = None) -> float:
        """Extract confidence, preferring structured output over pattern matching."""
        # Use structured confidence if available (from JSON schema)
        if structured_confidence is not None:
            try:
                conf = float(structured_confidence)
                return min(max(conf, 0.0), 1.0)  # Clamp to [0, 1]
            except (ValueError, TypeError):
                logger.warning(f"Invalid structured confidence: {structured_confidence}")
        
        # Fallback to pattern matching for legacy compatibility
        text = f"{reasoning} {raw_output}".lower()
        
        # Look for explicit confidence patterns
        confidence_patterns = [
            r'([0-9.]+)%\s*confident',
            r'confident.*?([0-9.]+)%',
            r'([0-9.]+)%\s*confidence',
            r'confidence.*?([0-9.]+)%',
            r'([0-9.]+)%\s*certain',
            r'certain.*?([0-9.]+)%',
            r'([0-9.]+)%\s*sure',
            r'sure.*?([0-9.]+)%',
            r'probability.*?([0-9.]+)%',
            r'([0-9.]+)%\s*probability'
        ]
        
        for pattern in confidence_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    value = float(match.group(1))
                    return min(value / 100 if value > 1 else value, 1.0)
                except ValueError:
                    continue
        
        # Conservative fallback - avoid overconfident defaults
        logger.warning(f"No confidence found, using conservative default for verdict: {verdict}")
        return 0.6  # Moderate confidence when uncertain


@dataclass 
class DTEResult:
    """Complete DTE evaluation result."""
    claim: str
    final_prediction: int  # 0 or 1
    escalated: bool
    verifier_a_result: VerificationResult
    verifier_b_result: VerificationResult
    judge_result: Optional[VerificationResult] = None
    gamma_threshold: float = 0.0
    
    @property
    def final_verdict(self) -> str:
        """Get final verdict as yes/no string."""
        return "yes" if self.final_prediction == 1 else "no"


class DTESystem:
    """
    Disagreement-Triggered Escalation System.
    
    Uses two heterogeneous verifiers and escalates to a judge when:
    1. Verifiers disagree, OR
    2. Both verifiers agree but confidence is below threshold
    """
    
    def __init__(self, 
                 verifier_a_lm: dspy.LM,
                 verifier_b_lm: dspy.LM,
                 judge_lm: dspy.LM,
                 gamma: float = 0.7,
                 use_cot: bool = True):
        """
        Initialize DTE system.
        
        Args:
            verifier_a_lm: Language model for verifier A
            verifier_b_lm: Language model for verifier B  
            judge_lm: Language model for judge
            gamma: Confidence threshold for escalation
            use_cot: Whether to use Chain of Thought reasoning
        """
        self.verifier_a = Verifier(use_cot=use_cot)
        self.verifier_b = Verifier(use_cot=use_cot)
        self.judge = Verifier(use_cot=use_cot)
        
        self.verifier_a_lm = verifier_a_lm
        self.verifier_b_lm = verifier_b_lm
        self.judge_lm = judge_lm
        
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
            'judge_calls': 0
        }
    
    def evaluate_claim(self, claim: str, ground_truth: Optional[str] = None) -> DTEResult:
        """
        Evaluate a claim using DTE protocol with robust error handling.
        
        Args:
            claim: The claim to evaluate
            ground_truth: Optional ground truth ("yes"/"no") for metrics
            
        Returns:
            DTEResult with evaluation outcome
            
        Raises:
            ValueError: If claim is empty or invalid
            RuntimeError: If all models fail
        """
        if not claim or not claim.strip():
            raise ValueError("Claim cannot be empty")
            
        self.metrics['total_evaluations'] += 1
        logger.debug(f"Evaluating claim: {claim[:100]}...")
        
        try:
            # Get verdicts from both verifiers
            with dspy.context(lm=self.verifier_a_lm):
                a_prediction = self.verifier_a(claim=claim)
                self.metrics['verifier_a_calls'] += 1
                
            with dspy.context(lm=self.verifier_b_lm):
                b_prediction = self.verifier_b(claim=claim)
                self.metrics['verifier_b_calls'] += 1
            
            # Extract reasoning and create results with dynamic confidence
            a_reasoning = getattr(a_prediction, 'reasoning', '')
            b_reasoning = getattr(b_prediction, 'reasoning', '')
            
            # Extract structured confidence if available
            a_confidence = getattr(a_prediction, 'confidence', None)
            b_confidence = getattr(b_prediction, 'confidence', None)
            
            a_result = VerificationResult.from_verdict(
                a_prediction.verdict, 
                reasoning=a_reasoning,
                raw_output=str(a_prediction),
                structured_confidence=a_confidence
            )
            b_result = VerificationResult.from_verdict(
                b_prediction.verdict,
                reasoning=b_reasoning, 
                raw_output=str(b_prediction),
                structured_confidence=b_confidence
            )
            
            # Check for agreement and confidence
            verifiers_agree = (a_result.prediction == b_result.prediction)
            min_confidence = min(a_result.confidence, b_result.confidence)
            
            if verifiers_agree:
                self.metrics['agreements'] += 1
                # Escalate if agreement but low confidence
                should_escalate = min_confidence < self.gamma
                if not should_escalate:
                    logger.debug(f"High confidence agreement: {min_confidence:.2f} >= {self.gamma}")
                else:
                    logger.debug(f"Low confidence agreement: {min_confidence:.2f} < {self.gamma}")
            else:
                self.metrics['disagreements'] += 1
                should_escalate = True
                logger.debug(f"Disagreement: A={a_result.prediction}, B={b_result.prediction}")
            
            if should_escalate:
                # Escalate to judge
                self.metrics['escalations'] += 1
                self.metrics['judge_calls'] += 1
                
                with dspy.context(lm=self.judge_lm):
                    judge_prediction = self.judge(claim=claim)
                    
                judge_reasoning = getattr(judge_prediction, 'reasoning', '')
                judge_confidence = getattr(judge_prediction, 'confidence', None)
                judge_result = VerificationResult.from_verdict(
                    judge_prediction.verdict,
                    reasoning=judge_reasoning,
                    raw_output=str(judge_prediction),
                    structured_confidence=judge_confidence
                )
                final_prediction = judge_result.prediction
                escalated = True
            else:
                # Accept consensus
                final_prediction = a_result.prediction
                judge_result = None
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
                judge_result=judge_result,
                gamma_threshold=self.gamma
            )
            
        except Exception as e:
            logger.error(f"Critical error evaluating claim '{claim}': {e}")
            # Return a safe fallback result
            fallback_result = VerificationResult(0, 0.1, f"Error: {str(e)}")
            return DTEResult(
                claim=claim,
                final_prediction=0,  # Conservative default
                escalated=True,  # Mark as escalated to indicate uncertainty
                verifier_a_result=fallback_result,
                verifier_b_result=fallback_result,
                judge_result=fallback_result,
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
        m['total_model_calls'] = m['verifier_a_calls'] + m['verifier_b_calls'] + m['judge_calls']
        if m['total_evaluations'] > 0:
            m['avg_calls_per_evaluation'] = m['total_model_calls'] / m['total_evaluations']
            
        return m