"""Pytest configuration and fixtures for DTE tests."""

import pytest
import dspy
from unittest.mock import Mock, MagicMock
from dte.core import DTESystem, VerificationResult, DTEResult
from dte.config import DTEConfig


class MockLM:
    """Mock language model for testing."""
    
    def __init__(self, name="mock", default_verdict="yes"):
        self.name = name
        self.default_verdict = default_verdict
        self.call_count = 0
    
    def __call__(self, **kwargs):
        self.call_count += 1
        return MockPrediction(self.default_verdict)


class MockPrediction:
    """Mock prediction object."""
    
    def __init__(self, verdict="yes"):
        self.verdict = verdict


@pytest.fixture
def mock_lm_yes():
    """Mock LM that always returns 'yes'."""
    return MockLM("mock_yes", "yes")


@pytest.fixture
def mock_lm_no():
    """Mock LM that always returns 'no'."""
    return MockLM("mock_no", "no")


@pytest.fixture
def mock_lm_neutral():
    """Mock LM for neutral referee."""
    return MockLM("mock_referee", "yes")


@pytest.fixture
def test_config():
    """Test configuration."""
    return DTEConfig(
        verifier_a_model="test_model_a",
        verifier_b_model="test_model_b", 
        judge_model="test_referee",
        gamma=0.7,
        use_cot=True,
        api_base="http://test:11434"
    )


@pytest.fixture
def dte_system(mock_lm_yes, mock_lm_no, mock_lm_neutral):
    """DTE system with mock language models."""
    return DTESystem(
        verifier_a_lm=mock_lm_yes,
        verifier_b_lm=mock_lm_no,
        judge_lm=mock_lm_neutral,
        gamma=0.7
    )


@pytest.fixture
def consensus_dte_system(mock_lm_yes, mock_lm_neutral):
    """DTE system where verifiers agree."""
    mock_lm_agree = MockLM("mock_agree", "yes")
    return DTESystem(  # type: ignore
        verifier_a_lm=mock_lm_yes,
        verifier_b_lm=mock_lm_agree,  # type: ignore
        judge_lm=mock_lm_neutral,
        gamma=0.7
    )


@pytest.fixture
def sample_claims():
    """Sample claims for testing."""
    return [
        "The capital of France is Paris.",
        "The Earth is flat.",
        "Python is a programming language."
    ]


@pytest.fixture
def sample_ground_truth():
    """Ground truth for sample claims."""
    return ["yes", "no", "yes"]


@pytest.fixture
def sample_dte_results():
    """Sample DTE results for testing."""
    return [
        DTEResult(
            claim="Test claim 1",
            final_prediction=1,
            escalated=False,
            verifier_a_result=VerificationResult(1, 0.9),
            verifier_b_result=VerificationResult(1, 0.8),
            gamma_threshold=0.7
        ),
        DTEResult(
            claim="Test claim 2", 
            final_prediction=0,
            escalated=True,
            verifier_a_result=VerificationResult(1, 0.8),
            verifier_b_result=VerificationResult(0, 0.9),
            judge_result=VerificationResult(0, 0.95),
            gamma_threshold=0.7
        )
    ]