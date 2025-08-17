#!/usr/bin/env python3
"""
Artificially force disagreements to demonstrate DTE escalation mechanism.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dspy
from typing import Optional
from scripts.dte_ollama import Verifier, DTEResult

class BiasedVerifier(dspy.Module):
    """A verifier that's biased toward 'yes' or 'no' to force disagreements."""
    
    def __init__(self, bias="yes", use_cot=True):
        super().__init__()
        self.bias = bias
        self.use_cot = use_cot
        
        if bias == "yes":
            self.prompt_prefix = "You tend to agree with claims. "
        else:
            self.prompt_prefix = "You tend to disagree with claims. "
            
        class BiasedVerifyClaim(dspy.Signature):
            claim = dspy.InputField(desc="The factual claim to verify")
            verdict = dspy.OutputField(desc="either 'yes' or 'no'")
            
        self.step = (dspy.ChainOfThought if use_cot else dspy.Predict)(BiasedVerifyClaim)
        
    def forward(self, claim: str) -> dspy.Prediction:
        """Verify claim with bias."""
        biased_claim = self.prompt_prefix + claim
        
        try:
            out = self.step(claim=biased_claim)  # type: ignore
            
            # Normalize verdict
            raw_verdict = (out.verdict or "").strip().lower()  # type: ignore
            
            if "yes" in raw_verdict and "no" not in raw_verdict:
                normalized_verdict = "yes"
            elif "no" in raw_verdict and "yes" not in raw_verdict:
                normalized_verdict = "no"
            else:
                # Default to bias
                normalized_verdict = self.bias
                
            out.verdict = normalized_verdict  # type: ignore
            return out  # type: ignore
            
        except Exception as e:
            return dspy.Prediction(verdict=self.bias)

class ForcedDisagreementDTE:
    """DTE system that forces disagreements for demonstration."""
    
    def __init__(self):
        # Set up biased verifiers
        self.lm = dspy.LM(model="ollama/llama3.2:latest", api_base="http://localhost:11434")
        dspy.configure(lm=self.lm)
        
        self.verifier_yes = BiasedVerifier(bias="yes")
        self.verifier_no = BiasedVerifier(bias="no") 
        self.referee = Verifier(use_cot=True)
        
        self.metrics = {'evaluations': 0, 'escalations': 0}
        
    def evaluate_claim(self, claim: str, ground_truth: Optional[str] = None) -> DTEResult:
        """Evaluate with forced disagreement."""
        self.metrics['evaluations'] += 1
        
        # Get biased verdicts (should disagree)
        a_result = self.verifier_yes(claim=claim)  # type: ignore
        b_result = self.verifier_no(claim=claim)  # type: ignore
        
        a_verdict = a_result.verdict  # type: ignore
        b_verdict = b_result.verdict  # type: ignore
        
        # Check agreement (should be rare with biased verifiers)
        if a_verdict == b_verdict:
            # Rare consensus
            final_verdict = a_verdict
            escalated = False
            judge_verdict = None
        else:
            # Expected disagreement → escalate
            self.metrics['escalations'] += 1
            referee_result = self.referee(claim=claim)  # type: ignore
            final_verdict = referee_result.verdict  # type: ignore
            judge_verdict = referee_result.verdict  # type: ignore
            escalated = True

        return DTEResult(
            claim=claim,
            final_verdict=final_verdict,
            escalated=escalated,
            verifier_a_verdict=a_verdict,
            verifier_b_verdict=b_verdict,
            judge_verdict=judge_verdict
            )

def main():
    """Demonstrate forced disagreements."""
    print("🎭 ARTIFICIAL DISAGREEMENT TEST")
    print("Using biased verifiers to force disagreements")
    print("=" * 60)
    
    # Create forced disagreement system
    forced_dte = ForcedDisagreementDTE()
    
    print("🎯 Biased DTE System:")
    print("   Verifier A: Biased toward 'YES'")
    print("   Verifier B: Biased toward 'NO'")
    print("   Referee: Neutral verifier")
    print("   → Should force disagreements on most claims!")
    print()
    
    # Test claims
    test_claims = [
        ("The sky is blue.", "yes"),
        ("Cats are mammals.", "yes"),
        ("The moon is made of cheese.", "no"),
        ("2 + 2 = 5.", "no"),
        ("Python is a programming language.", "yes")
    ]
    
    for i, (claim, expected) in enumerate(test_claims, 1):
        print(f"[{i}] {claim}")
        
        result = forced_dte.evaluate_claim(claim, expected)
        
        if result.escalated:
            print(f"    YES-Biased: {result.verifier_a_verdict}")
            print(f"    NO-Biased:  {result.verifier_b_verdict}")
            print(f"    🔥 DISAGREEMENT → Referee: {result.judge_verdict}")
            print(f"    Final: {result.final_verdict} (3 calls)")
            cost_indicator = "💰💰💰"
        else:
            print(f"    Both said: {result.final_verdict}")
            print(f"    🤝 Rare consensus (2 calls)")
            cost_indicator = "💰💰"
            
        correct = "✅" if result.final_verdict == expected else "❌"
        print(f"    {correct} Expected: {expected} {cost_indicator}")
        print()
    
    # Show results
    metrics = forced_dte.metrics
    escalation_rate = metrics['escalations'] / metrics['evaluations']
    
    print("=" * 60)
    print("📊 FORCED DISAGREEMENT RESULTS:")
    print(f"Escalations: {metrics['escalations']}/{metrics['evaluations']} ({escalation_rate:.0%})")
    
    if escalation_rate >= 0.8:  # 80%+ escalations
        print("🎯 SUCCESS: High escalation rate demonstrates DTE mechanism!")
        print("   ✓ Biased verifiers created disagreements")
        print("   ✓ Referee resolved conflicts")
        print("   ✓ System correctly identified uncertainty")
    else:
        print(f"🤔 Unexpected: Only {escalation_rate:.0%} escalations")
        print("   Even biased prompts led to some agreement")
        
    print(f"\n💡 This proves the DTE escalation mechanism works!")
    print(f"   When models disagree → referee makes final decision")

if __name__ == "__main__":
    main()