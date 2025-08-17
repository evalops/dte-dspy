#!/usr/bin/env python3
"""
Force disagreement by using very different models and controversial claims.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.dte_ollama import DTESystem

def main():
    """Test with models likely to disagree."""
    print("🔥 FORCING DISAGREEMENTS TEST")
    print("=" * 50)
    
    # Use very different models that might have different training
    dte = DTESystem(
        verifier_a_model="llama3.2:latest",    # Small recent model
        verifier_b_model="gpt-oss:20b",        # Different architecture 
        judge_model="qwq:latest",            # Reasoning-focused model
        gamma=0.7
    )
    
    print("✅ DTE with diverse models:")
    print("   A: llama3.2:latest (3B params)")
    print("   B: gpt-oss:20b (20B params)")  
    print("   Referee: qwq:latest (reasoning model)")
    print()
    
    # Claims more likely to cause disagreement
    controversial_claims = [
        ("Glass is a liquid.", "no"),  # Common misconception
        ("Humans evolved from monkeys.", "no"),  # Oversimplification 
        ("The Great Wall of China is visible from space.", "no"),  # Myth
    ]
    
    for i, (claim, expected) in enumerate(controversial_claims, 1):
        print(f"[{i}] {claim}")
        print("    (This claim often causes confusion/disagreement)")
        
        result = dte.evaluate_claim(claim, expected)
        
        agree = result.verifier_a_verdict == result.verifier_b_verdict
        print(f"    A: {result.verifier_a_verdict} | B: {result.verifier_b_verdict} | Agree: {agree}")
        
        if result.escalated:
            print(f"    🔥 ESCALATED → Referee: {result.judge_verdict}")
            print(f"    Final: {result.final_verdict} (3 calls)")
        else:
            print(f"    🤝 Consensus: {result.final_verdict} (2 calls)")
            
        correct = "✅" if result.final_verdict == expected else "❌"
        print(f"    {correct} Expected: {expected}")
        print()
    
    metrics = dte.get_metrics()
    escalation_rate = metrics.get('escalation_rate', 0)
    
    print("=" * 50)
    print(f"📊 RESULTS:")
    print(f"Escalation rate: {escalation_rate:.0%}")
    print(f"Accuracy: {metrics.get('overall_accuracy', 0):.0%}")
    
    if escalation_rate > 0:
        print(f"🎯 SUCCESS: Got {metrics['escalations']} escalations!")
        print("   DTE system is working - models disagree on edge cases")
    else:
        print("⚠️  Models still agreeing - they may be well-aligned")
        print("   This actually shows the models are robust!")

if __name__ == "__main__":
    main()