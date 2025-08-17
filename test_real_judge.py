#!/usr/bin/env python3
"""
Real DTE LLM-as-Judge Demonstration
Shows the system working with actual language models for disagreement resolution.
"""

from scripts.dte_ollama import DTESystem

def test_llm_as_judge():
    """Demonstrate LLM-as-Judge functionality with real models."""
    print("🧠 REAL LLM-AS-JUDGE DEMONSTRATION")
    print("=" * 60)
    
    # Configure DTE with real models
    dte = DTESystem(
        verifier_a_model="llama3.2:latest",    # Conservative verifier 
        verifier_b_model="mistral:7b",         # Different perspective
        judge_model="qwq:latest",              # Strong reasoning model
        gamma=0.4                              # Low threshold = more escalations
    )
    
    print(f"✅ DTE System Ready")
    print(f"   Verifier A: llama3.2:latest (temp=0.1)")
    print(f"   Verifier B: mistral:7b (temp=0.8)")  
    print(f"   Judge: qwq:latest (temp=0.0)")
    print(f"   Gamma threshold: 0.4 (low = aggressive escalation)")
    print()
    
    # Test controversial claims likely to trigger disagreement
    test_claims = [
        {
            "claim": "Pineapple belongs on pizza",
            "note": "Culinary controversy - likely to cause disagreement"
        },
        {
            "claim": "Nuclear energy is safer than solar power",
            "note": "Complex technical claim requiring expert judgment"
        },
        {
            "claim": "Python is better than JavaScript for web development",
            "note": "Technical opinion that different models might view differently"
        }
    ]
    
    for i, test in enumerate(test_claims, 1):
        claim = test["claim"]
        note = test["note"]
        
        print(f"[TEST {i}] {claim}")
        print(f"Note: {note}")
        print("-" * 50)
        
        result = dte.evaluate_claim(claim)
        
        print(f"Verifier A verdict: {result.verifier_a_verdict}")
        print()
        
        print(f"Verifier B verdict: {result.verifier_b_verdict}")
        print()
        
        if result.escalated:
            print("🚨 DISAGREEMENT DETECTED - ESCALATING TO LLM JUDGE")
            print(f"Judge verdict: {result.judge_verdict}")
            print()
            print(f"🏛️ **FINAL DECISION BY LLM JUDGE: {result.final_verdict.upper()}**")
            print(f"Cost: 3 model calls (2 verifiers + 1 judge)")
        else:
            print("🤝 CONSENSUS REACHED - No escalation needed")
            print(f"✅ **FINAL DECISION BY CONSENSUS: {result.final_verdict.upper()}**")
            print(f"Cost: 2 model calls (2 verifiers)")
        
        print()
        print("=" * 60)
        print()

if __name__ == "__main__":
    test_llm_as_judge()
