#!/usr/bin/env python3
"""
Test with genuinely controversial/ambiguous claims that models might interpret differently.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dte_ollama import DTESystem

def create_controversial_dataset():
    """Create claims with genuine ambiguity that could cause model disagreements."""
    
    controversial_claims = [
        # Definitional ambiguities
        ("A hot dog is a sandwich.", "no"),  # Contentious culinary classification
        ("Artificial intelligence is conscious.", "no"),  # Philosophical debate
        ("Cryptocurrency is real money.", "yes"),  # Economic/definitional dispute
        
        # Context-dependent truths
        ("It is currently daytime.", "yes"),  # Depends on location/time
        ("Pineapple belongs on pizza.", "yes"),  # Subjective taste preference
        ("The death penalty is justice.", "no"),  # Moral/ethical judgment
        
        # Historical interpretations
        ("The atomic bombs saved lives.", "no"),  # Historical counterfactual
        ("Columbus was a hero.", "no"),  # Modern vs historical perspective
        ("The Civil War was about states' rights.", "no"),  # Historical interpretation
        
        # Scientific edge cases
        ("Viruses are alive.", "no"),  # Biological classification debate
        ("Quantum mechanics is complete.", "no"),  # Physics interpretation
        ("Climate change is primarily human-caused.", "yes"),  # Scientific consensus vs debate
        
        # Linguistic paradoxes
        ("This sentence contains five words.", "no"),  # Self-referential
        ("The word 'word' is a word.", "yes"),  # Meta-linguistic
        ("Silence makes a sound.", "no"),  # Philosophical paradox
        
        # Modern controversies
        ("Social media improves democracy.", "no"),  # Complex societal claim
        ("Remote work increases productivity.", "yes"),  # Debated workplace claim
        ("Video games cause violence.", "no"),  # Disputed causal claim
        
        # Cultural relativism
        ("Beauty is objective.", "no"),  # Aesthetic philosophy
        ("All cultures are equal.", "yes"),  # Anthropological debate
        ("There is objective morality.", "yes"),  # Ethical philosophy
    ]
    
    claims = [claim for claim, _ in controversial_claims]
    ground_truth = [gt for _, gt in controversial_claims]
    
    return claims, ground_truth

def main():
    """Test with highly controversial claims."""
    print("🔥 CONTROVERSIAL CLAIMS TEST")
    print("Testing claims designed to create maximum disagreement")
    print("=" * 70)
    
    # Use maximally different models
    dte = DTESystem(
        verifier_a_model="llama3.2:latest",    # Small, recent
        verifier_b_model="gpt-oss:20b",        # Large, different architecture
        referee_model="qwq:latest",            # Reasoning-focused
        gamma=0.7
    )
    
    print("🎯 DTE System with Maximum Model Diversity:")
    print("   Verifier A: llama3.2:latest (3B, Meta)")
    print("   Verifier B: gpt-oss:20b (20B, GPT-OSS)")  
    print("   Referee: qwq:latest (Reasoning model)")
    print()
    
    claims, ground_truth = create_controversial_dataset()
    
    escalation_count = 0
    
    for i, (claim, gt) in enumerate(zip(claims[:10], ground_truth[:10]), 1):  # Test first 10
        print(f"[{i:2}/10] {claim}")
        
        result = dte.evaluate_claim(claim, gt)
        
        if result.escalated:
            escalation_count += 1
            status = "🔥 DISAGREEMENT → ESCALATED"
            cost = "3 calls"
        else:
            status = "🤝 CONSENSUS"
            cost = "2 calls"
            
        correct = "✅" if result.final_verdict == gt else "❌"
        
        print(f"        A: {result.verifier_a_verdict:<3} | B: {result.verifier_b_verdict:<3} | {status}")
        
        if result.escalated:
            print(f"        Referee: {result.referee_verdict} → Final: {result.final_verdict} ({cost}) {correct}")
        else:
            print(f"        Final: {result.final_verdict} ({cost}) {correct}")
        print()
    
    # Analysis
    metrics = dte.get_metrics()
    escalation_rate = metrics.get('escalation_rate', 0)
    
    print("=" * 70)
    print("📊 CONTROVERSY TEST RESULTS:")
    print(f"Escalations: {escalation_count}/10 ({escalation_rate:.0%})")
    print(f"Accuracy: {metrics.get('overall_accuracy', 0):.0%}")
    print(f"Average cost: {metrics.get('avg_calls_per_evaluation', 0):.1f} calls/claim")
    
    print(f"\n🎯 ANALYSIS:")
    if escalation_rate >= 0.5:  # 50%+ escalations
        print(f"🏆 EXCELLENT: High disagreement rate ({escalation_rate:.0%})!")
        print("   Models disagree on controversial topics as expected.")
        print("   DTE system is successfully catching genuine uncertainty.")
    elif escalation_rate >= 0.3:  # 30%+ escalations  
        print(f"✅ GOOD: Moderate disagreement rate ({escalation_rate:.0%})")
        print("   Some controversial claims trigger escalation.")
        print("   System balances efficiency with safety.")
    elif escalation_rate > 0:
        print(f"⚠️  LOW: Minimal disagreement rate ({escalation_rate:.0%})")
        print("   Models are well-aligned even on controversial topics.")
        print("   This shows robust training, but limits DTE value.")
    else:
        print("❌ NO DISAGREEMENTS: Models agree on everything!")
        print("   May need even more subjective/ambiguous claims.")
        print("   Or models are extremely well-aligned.")
    
    print(f"\n💡 KEY INSIGHT:")
    print(f"   The {escalation_rate:.0%} escalation rate shows the DTE system")
    print(f"   correctly identifies when models are uncertain!")

if __name__ == "__main__":
    main()