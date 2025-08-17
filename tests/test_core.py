"""Tests for core DTE functionality."""

import pytest
from dte.core import DTESystem, VerificationResult, DTEResult, Verifier


class TestVerificationResult:
    """Test VerificationResult dataclass."""
    
    def test_creation(self):
        result = VerificationResult(1, 0.8, "reasoning")
        assert result.prediction == 1
        assert result.confidence == 0.8
        assert result.reasoning == "reasoning"
    
    def test_from_verdict_yes(self):
        result = VerificationResult.from_verdict("yes", 0.9)
        assert result.prediction == 1
        assert result.confidence == 0.9
    
    def test_from_verdict_no(self):
        result = VerificationResult.from_verdict("no", 0.7)
        assert result.prediction == 0
        assert result.confidence == 0.7


class TestDTEResult:
    """Test DTEResult dataclass."""
    
    def test_creation(self, sample_dte_results):
        result = sample_dte_results[0]
        assert result.claim == "Test claim 1"
        assert result.final_prediction == 1
        assert not result.escalated
        assert result.gamma_threshold == 0.7
    
    def test_final_verdict_property(self, sample_dte_results):
        result_yes = sample_dte_results[0]  # final_prediction = 1
        result_no = sample_dte_results[1]   # final_prediction = 0
        
        assert result_yes.final_verdict == "yes"
        assert result_no.final_verdict == "no"


class TestDTESystem:
    """Test DTESystem core functionality."""
    
    def test_initialization(self, dte_system):
        assert dte_system.gamma == 0.7
        assert dte_system.metrics['total_evaluations'] == 0
    
    def test_reset_metrics(self, dte_system):
        # Modify metrics
        dte_system.metrics['total_evaluations'] = 5
        dte_system.metrics['escalations'] = 2
        
        # Reset
        dte_system.reset_metrics()
        
        assert dte_system.metrics['total_evaluations'] == 0
        assert dte_system.metrics['escalations'] == 0
    
    def test_evaluate_claim_disagreement(self, dte_system):
        """Test escalation when verifiers disagree."""
        result = dte_system.evaluate_claim("Test claim", "yes")
        
        # Should escalate because mock_lm_yes returns 'yes', mock_lm_no returns 'no'
        assert result.escalated
        assert result.verifier_a_result.prediction == 1  # yes
        assert result.verifier_b_result.prediction == 0  # no
        assert result.judge_result is not None
        assert result.final_prediction == result.judge_result.prediction
    
    def test_evaluate_claim_consensus(self, consensus_dte_system):
        """Test consensus when verifiers agree."""
        result = consensus_dte_system.evaluate_claim("Test claim", "yes")
        
        # Should not escalate because both return 'yes'
        assert not result.escalated
        assert result.verifier_a_result.prediction == 1  # yes
        assert result.verifier_b_result.prediction == 1  # yes
        assert result.judge_result is None
        assert result.final_prediction == 1
    
    def test_metrics_tracking(self, dte_system):
        """Test that metrics are properly tracked."""
        initial_metrics = dte_system.get_metrics()
        assert initial_metrics['total_evaluations'] == 0
        
        # Evaluate a claim (should escalate due to disagreement)
        dte_system.evaluate_claim("Test claim", "yes")
        
        metrics = dte_system.get_metrics()
        assert metrics['total_evaluations'] == 1
        assert metrics['escalations'] == 1
        assert metrics['disagreements'] == 1
        assert metrics['verifier_a_calls'] == 1
        assert metrics['verifier_b_calls'] == 1
        assert metrics['judge_calls'] == 1
    
    def test_get_metrics_calculations(self, dte_system):
        """Test that metric calculations are correct."""
        # Evaluate multiple claims
        dte_system.evaluate_claim("Claim 1", "yes")
        dte_system.evaluate_claim("Claim 2", "no")
        
        metrics = dte_system.get_metrics()
        
        # Check rate calculations
        assert metrics['escalation_rate'] == 1.0  # Both should escalate
        assert metrics['disagreement_rate'] == 1.0
        assert metrics['avg_calls_per_evaluation'] == 3.0  # 2 verifiers + 1 referee each
    
    def test_accuracy_tracking(self, dte_system):
        """Test accuracy tracking with ground truth."""
        # Evaluator returns 'yes', ground truth is 'yes' -> correct
        result1 = dte_system.evaluate_claim("Claim 1", "yes")
        
        # Evaluator returns 'yes' (referee), ground truth is 'no' -> incorrect  
        result2 = dte_system.evaluate_claim("Claim 2", "no")
        
        metrics = dte_system.get_metrics()
        
        # Should have 1 correct out of 2
        assert metrics['correct_predictions'] == 1
        assert metrics['overall_accuracy'] == 0.5


class TestVerifier:
    """Test Verifier component."""
    
    def test_verifier_initialization(self):
        verifier = Verifier(use_cot=True)
        assert verifier.use_cot is True
    
    def test_verifier_forward_mock(self, mock_lm_yes):
        """Test verifier with mock LM."""
        verifier = Verifier(use_cot=False)
        
        # Mock the step method
        verifier.step = lambda claim: mock_lm_yes()
        
        result = verifier.forward("Test claim")
        assert result.verdict == "yes"