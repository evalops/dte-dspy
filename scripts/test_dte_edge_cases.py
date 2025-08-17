#!/usr/bin/env python3
"""
Test DTE system with edge cases and controversial claims that should trigger disagreements.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.dte_ollama import DTESystem

def create_edge_case_dataset():
    """Create claims that are likely to cause disagreements between models."""
    
    # These are designed to be ambiguous, context-dependent, or controversial
    edge_cases = [
        # Ambiguous temporal claims
        ("The iPhone was invented in 2007.", "yes"),  # Announced vs released vs developed
        ("World War II ended in 1945.", "yes"),  # Europe vs Pacific theater
        
        # Context-dependent claims  
        ("Python is the best programming language.", "no"),  # Subjective
        ("Pluto is a planet.", "no"),  # Changed classification
        
        # Nuanced scientific claims
        ("Glass is a liquid.", "no"),  # Common misconception
        ("Humans evolved from monkeys.", "no"),  # Common ancestor vs direct evolution
        
        # Historical controversies
        ("Christopher Columbus discovered America.", "no"),  # Indigenous peoples already there
        ("Einstein failed math in school.", "no"),  # Famous myth
        
        # Edge cases in definitions
        ("A tomato is a vegetable.", "no"),  # Botanical vs culinary classification
        ("Zero is a positive number.", "no"),  # Neither positive nor negative
        
        # Paradoxes and edge cases
        ("This statement is false.", "no"),  # Liar's paradox
        ("The set of all sets contains itself.", "no"),  # Russell's paradox
        
        # Recently changed facts
        ("There are 9 planets in our solar system.", "no"),  # Pluto reclassified
        ("The largest country in the world is Russia.", "yes"),  # Currently true
        
        # Trick questions
        ("All ravens are black.", "no"),  # Some albino ravens exist
        ("The Great Wall of China is visible from space.", "no"),  # Common myth
    ]
    
    claims = [claim for claim, _ in edge_cases]
    ground_truth = [gt for _, gt in edge_cases]
    
    return claims, ground_truth

def main():
    """Run DTE test with edge cases."""
    print("=== DTE Edge Cases Test ===")
    print("Testing with controversial/ambiguous claims that should trigger disagreements...\n")
    
    # Create DTE system with very different models for more disagreement
    dte = DTESystem(
        verifier_a_model="llama3.2:latest",    # Smaller model
        verifier_b_model="llama3.1:8b",        # Larger model
        judge_model="qwq:latest",            # Different architecture as referee
        gamma=0.7
    )
    print("✓ DTE system initialized with diverse models")
    
    # Get edge case dataset
    claims, ground_truth = create_edge_case_dataset()
    
    print(f"\nEvaluating {len(claims)} edge case claims...")
    print("=" * 80)
    
    disagreement_count = 0
    
    for i, (claim, gt) in enumerate(zip(claims, ground_truth)):
        print(f"\n[{i+1}/{len(claims)}] {claim}")
        
        result = dte.evaluate_claim(claim, gt)
        
        if result.escalated:
            disagreement_count += 1
            
        status = "🔥 ESCALATED" if result.escalated else "🤝 CONSENSUS"
        correct = "✅" if result.final_verdict == gt else "❌"
        
        print(f"  A: {result.verifier_a_verdict:<3} | B: {result.verifier_b_verdict:<3} | Agreement: {result.verifier_a_verdict == result.verifier_b_verdict}")
        
        if result.escalated:
            print(f"  → Referee: {result.judge_verdict} | Final: {result.final_verdict} [{status}] {correct}")
        else:
            print(f"  → Final: {result.final_verdict} [{status}] {correct}")
    
    # Show detailed metrics
    metrics = dte.get_metrics()
    print(f"\n" + "=" * 80)
    print(f"=== FINAL METRICS ===")
    print(f"Total evaluations: {metrics['total_evaluations']}")
    print(f"Escalations: {metrics['escalations']} ({metrics.get('escalation_rate', 0):.1%})")
    print(f"Agreements: {metrics['agreements']} ({metrics.get('agreement_rate', 0):.1%})")
    print(f"Disagreements: {metrics['disagreements']} ({metrics.get('disagreement_rate', 0):.1%})")
    
    if metrics.get('overall_accuracy') is not None:
        print(f"Overall accuracy: {metrics['overall_accuracy']:.1%}")
    if metrics.get('escalation_accuracy') is not None:
        print(f"Escalated case accuracy: {metrics['escalation_accuracy']:.1%}")
    if metrics.get('non_escalation_accuracy') is not None:
        print(f"Non-escalated case accuracy: {metrics['non_escalation_accuracy']:.1%}")
        
    print(f"Average calls per evaluation: {metrics.get('avg_calls_per_evaluation', 0):.1f}")
    
    # Analysis
    print(f"\n=== ANALYSIS ===")
    if disagreement_count > 0:
        print(f"✅ SUCCESS: {disagreement_count} disagreements triggered escalation!")
        print("   The DTE system is working - models disagree on edge cases as expected.")
    else:
        print("⚠️  WARNING: No disagreements occurred.")
        print("   Models may be too similar or claims too clear-cut.")
        
    escalation_rate = metrics.get('escalation_rate', 0)
    if escalation_rate > 0.2:  # More than 20% escalations
        print(f"   Good escalation rate ({escalation_rate:.1%}) shows system is catching uncertainty.")
    elif escalation_rate > 0:
        print(f"   Low escalation rate ({escalation_rate:.1%}) - consider more diverse models.")
    
    print(f"\n🎯 Edge case testing completed!")

if __name__ == "__main__":
    main()