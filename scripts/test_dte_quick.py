#!/usr/bin/env python3
"""
Quick test of DTE system with just a few claims.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.dte_ollama import DTESystem

def main():
    """Run a quick DTE test."""
    print("=== Quick DTE Test ===")
    
    # Create DTE system
    dte = DTESystem(
        verifier_a_model="llama3.2:latest",    # Small, fast model
        verifier_b_model="llama3:8b",          # Different model for diversity  
        judge_model="llama3.1:8b",           # Referee model
        gamma=0.7
    )
    print("✓ DTE system initialized")
    
    # Test just a few claims
    test_claims = [
        "The capital of France is Paris.",
        "The Earth is flat.", 
        "Water boils at 100 degrees Celsius at sea level."
    ]
    
    ground_truth = ["yes", "no", "yes"]
    
    print(f"\nEvaluating {len(test_claims)} claims...")
    
    for i, (claim, gt) in enumerate(zip(test_claims, ground_truth)):
        print(f"\n[{i+1}/{len(test_claims)}] {claim}")
        
        result = dte.evaluate_claim(claim, gt)
        
        status = "ESCALATED" if result.escalated else "CONSENSUS"
        correct = "✓" if result.final_verdict == gt else "✗"
        
        print(f"  Verifier A: {result.verifier_a_verdict}")
        print(f"  Verifier B: {result.verifier_b_verdict}")
        
        if result.escalated:
            print(f"  Referee: {result.judge_verdict}")
            print(f"  → Final: {result.final_verdict} [{status}] {correct}")
        else:
            print(f"  → Final: {result.final_verdict} [{status}] {correct}")
    
    # Show metrics
    metrics = dte.get_metrics()
    print(f"\n=== Metrics ===")
    print(f"Total evaluations: {metrics['total_evaluations']}")
    print(f"Escalations: {metrics['escalations']}")
    print(f"Agreements: {metrics['agreements']}")
    print(f"Disagreements: {metrics['disagreements']}")
    print(f"Overall accuracy: {metrics.get('overall_accuracy', 0):.2%}")
    print(f"Avg calls per evaluation: {metrics.get('avg_calls_per_evaluation', 0):.1f}")
    
    print("\n✅ Quick test completed!")

if __name__ == "__main__":
    main()