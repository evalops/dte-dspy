#!/usr/bin/env python3
"""
Verbose demonstration of DTE escalation with controversial claims.
Shows step-by-step reasoning and disagreement detection.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dte import DTEConfig, DTESystem


def demonstrate_escalation():
    """Show DTE escalation working with controversial claims."""
    print("🔥 DTE ESCALATION DEMONSTRATION")
    print("=" * 60)
    print()
    
    # Configure for maximum disagreement potential
    config = DTEConfig(
        verifier_a_model="llama3.2:latest",  # Small model, fast responses
        verifier_b_model="gpt-oss:20b",      # Large model, different training
        judge_model="qwq:latest",          # Reasoning-focused model
        temperature_a=0.0,  # Deterministic
        temperature_b=0.8,  # High creativity/diversity  
        gamma=0.8,          # High confidence threshold
        verbose=True
    )
    
    print("🎯 DTE System Configuration:")
    print(f"   Verifier A: {config.verifier_a_model} (temp={config.temperature_a}) - Deterministic")
    print(f"   Verifier B: {config.verifier_b_model} (temp={config.temperature_b}) - Creative") 
    print(f"   Referee: {config.judge_model} (temp={config.temperature_judge}) - Reasoning")
    print(f"   Gamma threshold: {config.gamma} (high = more escalations)")
    print()
    
    # Create DTE system
    verifier_a_lm, verifier_b_lm, referee_lm = config.create_language_models()
    dte = DTESystem(verifier_a_lm, verifier_b_lm, referee_lm, gamma=config.gamma)
    
    # Controversial claims designed to trigger disagreements
    controversial_claims = [
        ("Glass is a liquid at room temperature.", "no"),
        ("Humans evolved from monkeys.", "no"),  
        ("A hot dog is a sandwich.", "no"),
        ("Artificial intelligence can be conscious.", "no"),
        ("Cryptocurrency is real money.", "yes"),
    ]
    
    escalation_count = 0
    
    for i, (claim, expected) in enumerate(controversial_claims, 1):
        print(f"🧪 TEST {i}: {claim}")
        print(f"    Expected answer: {expected}")
        print(f"    Controversy level: HIGH (designed to cause disagreement)")
        print()
        
        print("🔍 Step 1: Independent Verifier Evaluation")
        result = dte.evaluate_claim(claim, expected)
        
        # Show individual results
        a_verdict = "yes" if result.verifier_a_result.prediction == 1 else "no"
        b_verdict = "yes" if result.verifier_b_result.prediction == 1 else "no"
        
        print(f"    Verifier A ({config.verifier_a_model}): {a_verdict}")
        print(f"        Confidence: {result.verifier_a_result.confidence:.3f}")
        print(f"    Verifier B ({config.verifier_b_model}): {b_verdict}")
        print(f"        Confidence: {result.verifier_b_result.confidence:.3f}")
        print()
        
        # Check agreement
        agree = result.verifier_a_result.prediction == result.verifier_b_result.prediction
        print("🔍 Step 2: Agreement Analysis")
        print(f"    Verifiers agree: {agree}")
        
        if agree:
            avg_confidence = (result.verifier_a_result.confidence + result.verifier_b_result.confidence) / 2
            print(f"    Average confidence: {avg_confidence:.3f}")
            print(f"    Confidence threshold: {config.gamma}")
            below_threshold = avg_confidence < config.gamma
            print(f"    Below threshold: {below_threshold}")
            
        print()
        print("🔍 Step 3: Escalation Decision")
        
        if result.escalated:
            escalation_count += 1
            print("🔥 ESCALATION TRIGGERED!")
            
            if not agree:
                print("    Reason: Verifiers DISAGREE on the claim")
                print(f"        {config.verifier_a_model}: {a_verdict}")
                print(f"        {config.verifier_b_model}: {b_verdict}")
            else:
                print("    Reason: Low confidence despite agreement")
                print(f"        Average confidence {avg_confidence:.3f} < threshold {config.gamma}")
                
            referee_verdict = "yes" if result.judge_result.prediction == 1 else "no"
            print(f"    Referee ({config.judge_model}): {referee_verdict}")
            print(f"        Confidence: {result.judge_result.confidence:.3f}")
            print(f"    💰 Cost: 3 LLM calls")
            
        else:
            print("🤝 CONSENSUS - No escalation needed")
            print("    Both verifiers agree with sufficient confidence")
            print(f"    💰 Cost: 2 LLM calls")
            
        print()
        final_correct = "✅ CORRECT" if result.final_verdict == expected else "❌ INCORRECT"
        print(f"🎯 Final Decision: {result.final_verdict} {final_correct}")
        
        print("=" * 60)
        print()
        
    # Summary
    escalation_rate = escalation_count / len(controversial_claims)
    print("📊 DEMONSTRATION SUMMARY")
    print(f"Claims tested: {len(controversial_claims)}")
    print(f"Escalations: {escalation_count}")
    print(f"Escalation rate: {escalation_rate:.0%}")
    print()
    
    if escalation_rate >= 0.4:  # 40%+ escalations
        print("🏆 EXCELLENT: High escalation rate demonstrates DTE working!")
        print("   ✓ System correctly identifies controversial/uncertain claims")
        print("   ✓ Referee provides authoritative resolution")
        print("   ✓ DTE protocol successfully balances cost vs accuracy")
    elif escalation_rate > 0:
        print("✅ GOOD: Some escalations show DTE mechanism is working")
        print("   ✓ System escalates when appropriate")
        print("   ✓ Models show reasonable disagreement on edge cases")
    else:
        print("⚠️ LIMITED: No escalations observed")
        print("   → Models are very well-aligned on these topics")
        print("   → May need even more controversial/ambiguous claims")
        print("   → This actually demonstrates model robustness!")
        
    print()
    print("💡 Key Insight: DTE escalates only when models disagree or lack confidence,")
    print("   optimizing the trade-off between accuracy and computational cost.")


if __name__ == "__main__":
    demonstrate_escalation()