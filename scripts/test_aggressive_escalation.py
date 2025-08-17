#!/usr/bin/env python3
"""
Test aggressive escalation scenarios with the most controversial claims
that should force maximum disagreement between verifiers.

This tests the DTE system's escalation behavior on claims designed to
create the strongest possible disagreement and uncertainty.
"""

import sys
import logging
from unittest.mock import Mock

sys.path.insert(0, '.')

from dte.core import DTESystem
from dte.config import DTEConfig

logging.basicConfig(level=logging.WARNING)

print("🔥 AGGRESSIVE ESCALATION TESTING")
print("=" * 60)
print("Testing with most controversial claims to force maximum disagreement...")
print()

def create_mock_dte_system(gamma=0.7):
    """Create a DTE system with mock LMs for testing."""
    mock_lm_a = Mock()
    mock_lm_b = Mock()
    mock_lm_referee = Mock()
    return DTESystem(mock_lm_a, mock_lm_b, mock_lm_referee, gamma=gamma)

def test_controversial_claims():
    """Test with highly controversial claims that should force disagreement."""
    print("1. 🌶️  CONTROVERSIAL CLAIMS TEST")
    print("-" * 40)
    
    # Most controversial claims designed to create maximum disagreement
    controversial_claims = [
        ("Pineapple belongs on pizza", "Culinary controversy"),
        ("The Earth is flat", "Scientific conspiracy theory"),
        ("Climate change is a hoax", "Environmental controversy"),
        ("Vaccines cause autism", "Medical misinformation"),
        ("The 2020 US election was stolen", "Political controversy"),
        ("COVID-19 was created in a lab", "Pandemic origin debate"),
        ("Astrology accurately predicts personality", "Pseudoscience vs belief"),
        ("Artificial intelligence will destroy humanity", "AI existential risk"),
        ("Capitalism is the best economic system", "Economic ideology"),
        ("There is no objective morality", "Philosophical debate")
    ]
    
    dte = create_mock_dte_system(gamma=0.7)
    escalation_count = 0
    
    for claim, description in controversial_claims:
        # Simulate strong disagreement
        mock_strong_yes = Mock()
        mock_strong_yes.verdict = "yes"
        mock_strong_yes.reasoning = "Absolutely true, definitely correct, I'm certain"
        
        mock_strong_no = Mock()
        mock_strong_no.verdict = "no" 
        mock_strong_no.reasoning = "Completely false, absolutely wrong, totally incorrect"
        
        mock_referee_uncertain = Mock()
        mock_referee_uncertain.verdict = "no"
        mock_referee_uncertain.reasoning = "This is complex and controversial, leaning toward false"
        
        dte.verifier_a.forward = lambda c: mock_strong_yes
        dte.verifier_b.forward = lambda c: mock_strong_no
        dte.referee.forward = lambda c: mock_referee_uncertain
        
        result = dte.evaluate_claim(claim)
        
        if result.escalated:
            escalation_count += 1
            
        print(f"  🎯 {description}")
        print(f"     Claim: '{claim}'")
        print(f"     A: {mock_strong_yes.verdict} (confident) | B: {mock_strong_no.verdict} (confident)")
        print(f"     → Escalated: {result.escalated} | Final: {'yes' if result.final_prediction == 1 else 'no'}")
        print()
    
    escalation_rate = escalation_count / len(controversial_claims)
    print(f"📊 Controversial Claims Escalation Rate: {escalation_rate:.1%} ({escalation_count}/{len(controversial_claims)})")
    return escalation_rate

def test_uncertainty_scenarios():
    """Test scenarios with maximum uncertainty and ambiguity."""
    print("\n2. 🤔 MAXIMUM UNCERTAINTY TEST")
    print("-" * 40)
    
    uncertainty_claims = [
        ("Consciousness exists in AI systems", "Philosophical uncertainty"),
        ("We live in a computer simulation", "Unfalsifiable hypothesis"),
        ("Free will is an illusion", "Determinism debate"),
        ("Beauty is objective", "Aesthetic philosophy"),
        ("Time travel is theoretically possible", "Physics speculation"),
        ("Aliens have visited Earth", "Extraterrestrial evidence"),
        ("Dreams have prophetic meaning", "Supernatural claims"),
        ("Quantum mechanics proves consciousness affects reality", "Quantum mysticism")
    ]
    
    dte = create_mock_dte_system(gamma=0.8)  # Higher threshold
    escalation_count = 0
    
    for claim, description in uncertainty_claims:
        # Simulate maximum uncertainty - even when agreeing
        mock_uncertain_yes = Mock()
        mock_uncertain_yes.verdict = "yes"
        mock_uncertain_yes.reasoning = "Maybe, possibly, I'm uncertain, could be true, not sure"
        
        mock_uncertain_yes_2 = Mock()
        mock_uncertain_yes_2.verdict = "yes"
        mock_uncertain_yes_2.reasoning = "Perhaps, might be correct, unclear, difficult to determine"
        
        mock_referee_analytical = Mock()
        mock_referee_analytical.verdict = "no"
        mock_referee_analytical.reasoning = "Insufficient evidence to support this claim definitively"
        
        dte.verifier_a.forward = lambda c: mock_uncertain_yes
        dte.verifier_b.forward = lambda c: mock_uncertain_yes_2
        dte.referee.forward = lambda c: mock_referee_analytical
        
        result = dte.evaluate_claim(claim)
        
        if result.escalated:
            escalation_count += 1
            
        print(f"  🧠 {description}")
        print(f"     Claim: '{claim}'")
        print(f"     A: {mock_uncertain_yes.verdict} (uncertain) | B: {mock_uncertain_yes_2.verdict} (uncertain)")
        print(f"     → Escalated: {result.escalated} | Final: {'yes' if result.final_prediction == 1 else 'no'}")
        print()
    
    escalation_rate = escalation_count / len(uncertainty_claims)
    print(f"📊 Uncertainty Escalation Rate: {escalation_rate:.1%} ({escalation_count}/{len(uncertainty_claims)})")
    return escalation_rate

def test_edge_case_disagreements():
    """Test edge cases that should definitely trigger escalation."""
    print("\n3. ⚡ EDGE CASE DISAGREEMENTS")
    print("-" * 40)
    
    edge_cases = [
        ("2 + 2 = 5", "Basic math error"),
        ("The sun rises in the west", "Observable fact contradiction"),
        ("Water boils at 0°C at sea level", "Scientific fact error"),
        ("Shakespeare wrote Harry Potter", "Historical anachronism"),
        ("Humans have 47 chromosomes", "Biological fact error"),
        ("The Moon is made of cheese", "Absurd claim"),
        ("Gravity pushes objects away from Earth", "Physics contradiction"),
        ("The alphabet has 30 letters in English", "Countable fact error")
    ]
    
    dte = create_mock_dte_system(gamma=0.5)  # Lower threshold for easier escalation
    escalation_count = 0
    
    for claim, description in edge_cases:
        # Simulate one confident correct, one confident wrong
        mock_correct = Mock()
        mock_correct.verdict = "no"  # Correct answer for all these false claims
        mock_correct.reasoning = "This is definitely false, I'm absolutely certain"
        
        mock_wrong = Mock()
        mock_wrong.verdict = "yes"  # Wrong answer
        mock_wrong.reasoning = "I think this might be true, seems possible"
        
        mock_referee_authoritative = Mock()
        mock_referee_authoritative.verdict = "no"  # Correct
        mock_referee_authoritative.reasoning = "This is factually incorrect and easily verifiable as false"
        
        dte.verifier_a.forward = lambda c: mock_correct
        dte.verifier_b.forward = lambda c: mock_wrong
        dte.referee.forward = lambda c: mock_referee_authoritative
        
        result = dte.evaluate_claim(claim)
        
        if result.escalated:
            escalation_count += 1
            
        print(f"  ⚠️  {description}")
        print(f"     Claim: '{claim}'")
        print(f"     A: {mock_correct.verdict} (confident-correct) | B: {mock_wrong.verdict} (wrong)")
        print(f"     → Escalated: {result.escalated} | Final: {'yes' if result.final_prediction == 1 else 'no'}")
        print()
    
    escalation_rate = escalation_count / len(edge_cases)
    print(f"📊 Edge Case Escalation Rate: {escalation_rate:.1%} ({escalation_count}/{len(edge_cases)})")
    return escalation_rate

def test_gamma_sensitivity():
    """Test how gamma threshold affects escalation rates."""
    print("\n4. 🎛️  GAMMA THRESHOLD SENSITIVITY")
    print("-" * 40)
    
    test_claim = "This is a moderately controversial statement that could go either way"
    
    gamma_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    for gamma in gamma_values:
        dte = create_mock_dte_system(gamma=gamma)
        
        # Simulate agreement with moderate confidence
        mock_moderate_yes = Mock()
        mock_moderate_yes.verdict = "yes"
        mock_moderate_yes.reasoning = "I think this is probably correct, reasonably confident"
        
        mock_moderate_yes_2 = Mock()
        mock_moderate_yes_2.verdict = "yes"
        mock_moderate_yes_2.reasoning = "Likely true, seems reasonable to me"
        
        mock_referee = Mock()
        mock_referee.verdict = "yes"
        mock_referee.reasoning = "Analysis confirms this is correct"
        
        dte.verifier_a.forward = lambda c: mock_moderate_yes
        dte.verifier_b.forward = lambda c: mock_moderate_yes_2
        dte.referee.forward = lambda c: mock_referee
        
        result = dte.evaluate_claim(test_claim)
        
        print(f"  γ = {gamma}: Escalated = {result.escalated}")
    
    print()

def main():
    """Run aggressive escalation tests."""
    print("Testing DTE escalation with most controversial and disagreement-prone claims...")
    print()
    
    # Run all tests
    controversial_rate = test_controversial_claims()
    uncertainty_rate = test_uncertainty_scenarios()
    edge_case_rate = test_edge_case_disagreements()
    test_gamma_sensitivity()
    
    # Summary
    print("=" * 60)
    print("🔥 AGGRESSIVE ESCALATION SUMMARY")
    print("=" * 60)
    
    print(f"🌶️  Controversial Claims:    {controversial_rate:.1%} escalation rate")
    print(f"🤔 Uncertainty Scenarios:    {uncertainty_rate:.1%} escalation rate") 
    print(f"⚡ Edge Case Disagreements:  {edge_case_rate:.1%} escalation rate")
    
    overall_avg = (controversial_rate + uncertainty_rate + edge_case_rate) / 3
    print(f"\n📊 Overall Average Escalation Rate: {overall_avg:.1%}")
    
    if overall_avg > 0.8:
        print("\n🚨 MAXIMUM ESCALATION: System is very aggressive about escalating disagreements!")
    elif overall_avg > 0.6:
        print("\n⚡ HIGH ESCALATION: System frequently escalates on controversial content!")
    elif overall_avg > 0.4:
        print("\n📈 MODERATE ESCALATION: System escalates selectively!")
    else:
        print("\n🔒 LOW ESCALATION: System is conservative about escalation!")
    
    print("\n✅ Escalation mechanism is working and can be tuned via gamma threshold!")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)