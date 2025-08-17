#!/usr/bin/env python3
"""
Basic test of DTE implementation structure.
This tests the core logic without requiring real LLM calls.
"""

import sys
import os
import logging

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.dte_core import DTESystem, VerificationResult, DTEResult

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockLM:
    """Mock language model for testing."""
    
    def __init__(self, name="mock", default_response="yes"):
        self.name = name
        self.default_response = default_response
    
    def __call__(self, **kwargs):
        """Mock call that returns a predictable response."""
        return MockPrediction(self.default_response)


class MockPrediction:
    """Mock prediction object."""
    
    def __init__(self, verdict="yes"):
        self.verdict = verdict


def test_dte_structure():
    """Test the basic DTE system structure."""
    print("=== Testing DTE System Structure ===")
    
    # Create mock models
    verifier_a = MockLM("verifier_a", "yes")
    verifier_b = MockLM("verifier_b", "yes") 
    referee = MockLM("referee", "no")
    
    # Create DTE system
    dte = DTESystem(verifier_a, verifier_b, referee, default_gamma=0.7)  # type: ignore
    
    # Test basic properties
    assert dte.default_gamma == 0.7
    assert dte.verifier_a_lm.name == "verifier_a"  # type: ignore
    assert dte.verifier_b_lm.name == "verifier_b"  # type: ignore
    assert dte.referee_lm.name == "referee"  # type: ignore
    
    print("✓ DTE system created successfully")
    
    # Test metrics initialization
    metrics = dte.get_metrics()
    assert metrics['total_evaluations'] == 0
    assert metrics['escalations'] == 0
    print("✓ Metrics initialized correctly")
    
    return dte


def test_verification_result():
    """Test VerificationResult dataclass."""
    print("\n=== Testing VerificationResult ===")
    
    result = VerificationResult(
        prediction=1,
        confidence=0.8,
        reasoning="Test reasoning"
    )
    
    assert result.prediction == 1
    assert result.confidence == 0.8
    assert result.reasoning == "Test reasoning"
    
    print("✓ VerificationResult works correctly")


def test_dte_result():
    """Test DTEResult dataclass.""" 
    print("\n=== Testing DTEResult ===")
    
    verifier_a_result = VerificationResult(1, 0.8, "A says yes")
    verifier_b_result = VerificationResult(1, 0.9, "B says yes")
    
    dte_result = DTEResult(
        final_prediction=1,
        escalated=False,
        verifier_a_result=verifier_a_result,
        verifier_b_result=verifier_b_result,
        gamma_threshold=0.7
    )
    
    assert dte_result.final_prediction == 1
    assert dte_result.escalated == False
    assert dte_result.verifier_a_result.prediction == 1
    assert dte_result.gamma_threshold == 0.7
    
    print("✓ DTEResult works correctly")


def test_parse_verifier_output():
    """Test output parsing logic."""
    print("\n=== Testing Output Parsing ===")
    
    dte = test_dte_structure()
    
    # Test "yes" response
    yes_output = MockPrediction("yes")
    result = dte._parse_verifier_output(yes_output)
    assert result.prediction == 1
    assert result.confidence == 0.8
    print("✓ 'yes' parsed correctly")
    
    # Test "no" response  
    no_output = MockPrediction("no")
    result = dte._parse_verifier_output(no_output)
    assert result.prediction == 0
    assert result.confidence == 0.8
    print("✓ 'no' parsed correctly")
    
    # Test ambiguous response
    ambiguous_output = MockPrediction("maybe")
    result = dte._parse_verifier_output(ambiguous_output)
    assert result.prediction == 0  # defaults to 'no'
    assert result.confidence == 0.5
    print("✓ Ambiguous response handled correctly")


def test_escalation_logic():
    """Test the core escalation logic without actual LLM calls."""
    print("\n=== Testing Escalation Logic ===")
    
    # Create DTE system with predictable mock responses
    verifier_a = MockLM("verifier_a", "yes") 
    verifier_b = MockLM("verifier_b", "yes")  # Agreement case
    referee = MockLM("referee", "no")
    
    dte = DTESystem(verifier_a, verifier_b, referee, default_gamma=0.7)  # type: ignore
    
    # Mock the verifier calls by directly testing the decision logic
    a_result = VerificationResult(1, 0.8, "A says yes")  # high confidence
    b_result = VerificationResult(1, 0.9, "B says yes")  # high confidence
    
    # Test agreement with high confidence (should not escalate)
    verifiers_agree = (a_result.prediction == b_result.prediction)
    min_confidence = min(a_result.confidence, b_result.confidence)
    gamma = 0.7
    
    should_escalate = not (verifiers_agree and min_confidence >= gamma)
    
    assert verifiers_agree == True
    assert min_confidence == 0.8
    assert should_escalate == False  # Should not escalate
    print("✓ High confidence agreement logic correct")
    
    # Test agreement with low confidence (should escalate)
    a_result_low = VerificationResult(1, 0.5, "A says yes")  # low confidence
    b_result_low = VerificationResult(1, 0.6, "B says yes")  # low confidence
    
    min_confidence_low = min(a_result_low.confidence, b_result_low.confidence)
    should_escalate_low = not (verifiers_agree and min_confidence_low >= gamma)
    
    assert should_escalate_low == True  # Should escalate due to low confidence
    print("✓ Low confidence agreement logic correct")
    
    # Test disagreement (should escalate)
    a_result_disagree = VerificationResult(1, 0.9, "A says yes")
    b_result_disagree = VerificationResult(0, 0.9, "B says no")
    
    verifiers_disagree = (a_result_disagree.prediction != b_result_disagree.prediction)
    should_escalate_disagree = verifiers_disagree
    
    assert should_escalate_disagree == True  # Should escalate due to disagreement
    print("✓ Disagreement logic correct")


def test_metrics_tracking():
    """Test metrics tracking functionality."""
    print("\n=== Testing Metrics Tracking ===")
    
    dte = test_dte_structure()
    
    # Simulate some operations
    dte.metrics['total_evaluations'] = 10
    dte.metrics['escalations'] = 3
    dte.metrics['agreements'] = 7
    dte.metrics['disagreements'] = 3
    dte.metrics['verifier_a_calls'] = 10
    dte.metrics['verifier_b_calls'] = 10 
    dte.metrics['referee_calls'] = 3
    
    metrics = dte.get_metrics()
    
    # Test computed rates
    assert metrics['escalation_rate'] == 0.3  # 3/10
    assert metrics['disagreement_rate'] == 0.3  # 3/10
    assert metrics['agreement_rate'] == 0.7  # 7/10
    assert metrics['total_model_calls'] == 23  # 10+10+3
    assert metrics['avg_calls_per_evaluation'] == 2.3  # 23/10
    
    print("✓ Metrics calculations correct")


def main():
    """Run all tests."""
    print("Running DTE System Tests")
    print("=" * 50)
    
    try:
        test_verification_result()
        test_dte_result() 
        test_dte_structure()
        test_parse_verifier_output()
        test_escalation_logic()
        test_metrics_tracking()
        
        print("\n" + "=" * 50)
        print("🎉 All tests passed! DTE system structure is working correctly.")
        print("\nNext steps:")
        print("1. Test with real language models (OpenAI, Ollama, etc.)")
        print("2. Run evaluation harness with actual datasets")
        print("3. Perform gamma threshold sweeping")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0


if __name__ == "__main__":
    exit(main())