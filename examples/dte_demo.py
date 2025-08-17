#!/usr/bin/env python3
"""
Demo showing DTE system with both consensus and escalation cases.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dte_ollama import DTESystem

def main():
    """Demo the DTE system."""
    print("🎯 DISAGREEMENT-TRIGGERED ESCALATION DEMO")
    print("=" * 60)
    
    # Create DTE system
    dte = DTESystem(
        verifier_a_model="llama3.2:latest",
        verifier_b_model="llama3:8b", 
        judge_model="llama3.1:8b",
        gamma=0.7
    )
    
    print("✅ DTE System Ready")
    print(f"   Verifier A: llama3.2:latest (temp=0.1)")
    print(f"   Verifier B: llama3:8b (temp=0.9)")  
    print(f"   Judge: llama3.1:8b (temp=0.0)")
    print(f"   Gamma threshold: {dte.gamma}")
    print()
    
    # Test cases designed to show different behaviors
    test_cases = [
        ("The capital of France is Paris.", "yes", "Should get consensus"),
        ("The Earth is flat.", "no", "Should get consensus"),
        ("Pluto is a planet.", "no", "May cause disagreement (reclassified in 2006)")
    ]
    
    for i, (claim, expected, note) in enumerate(test_cases, 1):
        print(f"[TEST {i}] {claim}")
        print(f"Expected: {expected} ({note})")
        print("-" * 50)
        
        result = dte.evaluate_claim(claim, expected)
        
        print(f"Verifier A verdict: {result.verifier_a_verdict}")
        print(f"Verifier B verdict: {result.verifier_b_verdict}")
        print(f"Agreement: {'YES' if result.verifier_a_verdict == result.verifier_b_verdict else 'NO'}")
        
        if result.escalated:
            print(f"🔥 ESCALATED to judge")
            print(f"Judge verdict: {result.judge_verdict}")
            print(f"Final decision: {result.final_verdict}")
            print(f"Cost: 3 model calls")
        else:
            print(f"🤝 CONSENSUS reached")  
            print(f"Final decision: {result.final_verdict}")
            print(f"Cost: 2 model calls")
            
        correct = "✅ CORRECT" if result.final_verdict == expected else "❌ INCORRECT"
        print(f"Result: {correct}")
        print()
    
    # Show final metrics
    metrics = dte.get_metrics()
    print("=" * 60)
    print("📊 FINAL METRICS")
    print(f"Total evaluations: {metrics['total_evaluations']}")
    print(f"Escalations: {metrics['escalations']} ({metrics.get('escalation_rate', 0):.0%})")
    print(f"Accuracy: {metrics.get('overall_accuracy', 0):.0%}")
    print(f"Avg cost: {metrics.get('avg_calls_per_evaluation', 0):.1f} calls per claim")
    
    print("\n🎉 Demo complete! The DTE system successfully:")
    print("   • Detects when verifiers agree (consensus)")
    print("   • Escalates when verifiers disagree") 
    print("   • Uses judge to resolve conflicts")
    print("   • Tracks accuracy and cost metrics")

if __name__ == "__main__":
    main()