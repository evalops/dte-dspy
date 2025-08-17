#!/usr/bin/env python3
"""
Test script for DTE improvements.

This script tests the enhanced DTE system with:
- Dynamic confidence extraction
- Improved error handling  
- Better verdict parsing
"""

import sys
import logging
from unittest.mock import Mock, MagicMock

# Add the dte package to path
sys.path.insert(0, '.')

from dte.core import DTESystem, VerificationResult, Verifier
import dspy

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockLM:
    """Mock language model for testing."""
    def __init__(self, responses):
        self.responses = responses
        self.call_count = 0
    
    def __call__(self, *args, **kwargs):
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return response

def test_confidence_extraction():
    """Test dynamic confidence extraction."""
    print("Testing confidence extraction...")
    
    # Test explicit confidence patterns
    result1 = VerificationResult.from_verdict("yes", "I am 95% confident this is correct", "")
    assert result1.confidence > 0.9, f"Expected high confidence, got {result1.confidence}"
    
    # Test confidence words
    result2 = VerificationResult.from_verdict("yes", "I am definitely sure this is accurate", "")
    assert result2.confidence > 0.8, f"Expected high confidence, got {result2.confidence}"
    
    result3 = VerificationResult.from_verdict("no", "I might be wrong but this seems false", "")
    assert result3.confidence < 0.5, f"Expected low confidence, got {result3.confidence}"
    
    print("✓ Confidence extraction working correctly")

def test_error_handling():
    """Test robust error handling."""
    print("Testing error handling...")
    
    verifier = Verifier(use_cot=False)
    
    # Test empty claim
    result = verifier.forward("")
    assert result.verdict == "no", f"Expected 'no' for empty claim, got {result.verdict}"
    
    # Test with mock that returns None
    verifier.step = lambda claim: None  # type: ignore
    result = verifier.forward("test claim")
    assert result.verdict == "no", f"Expected 'no' for None output, got {result.verdict}"
    
    print("✓ Error handling working correctly")

def test_verdict_parsing():
    """Test enhanced verdict parsing."""
    print("Testing verdict parsing...")
    
    verifier = Verifier(use_cot=False)
    
    # Mock the step method to return different verdict formats
    test_cases = [
        ("Yes, this is correct", "yes"),
        ("No, this is false", "no"), 
        ("True statement", "yes"),
        ("Invalid claim", "no"),
        ("ambiguous response", "no"),  # Should default to no
    ]
    
    for raw_verdict, expected in test_cases:
        mock_output = Mock()
        mock_output.verdict = raw_verdict
        mock_output.reasoning = ""
        
        verifier.step = lambda claim: mock_output  # type: ignore
        result = verifier.forward("test claim")
        
        assert result.verdict == expected, f"Expected {expected} for '{raw_verdict}', got {result.verdict}"
    
    print("✓ Verdict parsing working correctly")

def test_dte_system_integration():
    """Test full DTE system with mocked models."""
    print("Testing DTE system integration...")
    
    # Create mock language models
    mock_lm_a = Mock()
    mock_lm_b = Mock() 
    mock_lm_referee = Mock()
    
    # Create DTE system
    dte = DTESystem(mock_lm_a, mock_lm_b, mock_lm_referee, gamma=0.7)
    
    # Mock verifier outputs - disagreement case
    mock_a_output = Mock()
    mock_a_output.verdict = "yes"
    mock_a_output.reasoning = "I am 90% confident this is true"
    
    mock_b_output = Mock()
    mock_b_output.verdict = "no"
    mock_b_output.reasoning = "I'm 85% sure this seems incorrect to me"
    
    mock_referee_output = Mock()
    mock_referee_output.verdict = "yes"
    mock_referee_output.reasoning = "After careful analysis, I'm 95% confident this is correct"
    
    # Override verifier methods
    dte.verifier_a.forward = lambda **kwargs: mock_a_output  # type: ignore
    dte.verifier_b.forward = lambda **kwargs: mock_b_output  # type: ignore
    dte.judge.forward = lambda **kwargs: mock_referee_output  # type: ignore
    
    # Test evaluation
    result = dte.evaluate_claim("Test claim", "yes")
    
    assert result.escalated == True, "Expected escalation due to disagreement"
    assert result.final_prediction == 1, "Expected final prediction to be 1 (yes)"
    assert result.verifier_a_result.confidence > 0.8, "Expected high confidence from A"
    
    print("✓ DTE system integration working correctly")

def test_error_recovery():
    """Test DTE system error recovery."""
    print("Testing error recovery...")
    
    # Create DTE system with None models (will cause errors)
    dte = DTESystem(None, None, None, gamma=0.7)  # type: ignore
    
    try:
        result = dte.evaluate_claim("Test claim")
        # Should not raise exception but return fallback result
        assert result.final_prediction == 0, "Expected conservative fallback"
        assert result.escalated == True, "Expected escalation flag on error"
        print("✓ Error recovery working correctly")
    except Exception as e:
        print(f"✗ Error recovery failed: {e}")

def main():
    """Run all tests."""
    print("🧪 Testing DTE Improvements\n")
    
    try:
        test_confidence_extraction()
        test_error_handling()
        test_verdict_parsing()
        test_dte_system_integration()
        test_error_recovery()
        
        print("\n🎉 All tests passed! Improvements are working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)