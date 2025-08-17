#!/usr/bin/env python3
"""
DTE Language Program Evaluation with Multi-Agent Architecture.

Systematic evaluation of disagreement-triggered escalation using
heterogeneous language models in a structured verification framework.

Multi-Agent Configuration:
- Verifier Agent A: llama3.1:8b (LLaMA family - instruction tuned)
- Verifier Agent B: mistral:7b (Mistral architecture - diverse reasoning)
- Judge Model: qwq:latest (QwQ reasoning specialist - final arbitration)
"""

import sys
import time
import logging
import random
import json
import os
import signal
from contextlib import contextmanager

sys.path.insert(0, '.')

from dte import DTESystem, DTEConfig
from dte.config import validate_ollama_connection, validate_models_present, generate_pull_commands

logging.basicConfig(level=logging.WARNING)  # Reduce noise

@contextmanager
def timeout_context(seconds: int):
    """Context manager for adding timeouts to operations."""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")

    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

def test_real_models(seed: int = 42):
    """Test with real models - fast version with deterministic results."""
    # Set deterministic seed
    random.seed(seed)

    print("🧪 REAL MODEL DTE TEST (Fast Version)")
    print("=" * 50)
    print(f"Seed: {seed} (for reproducibility)")
    print()

    # Check connection
    if not validate_ollama_connection():
        print("❌ Ollama not accessible. Please start: ollama serve")
        return False

    # Configure with diverse models for better disagreement detection
    config = DTEConfig(
        verifier_a_model="llama3.1:8b",      # LLaMA family - instruction tuned
        verifier_b_model="mistral:7b",       # Mistral - different architecture/training
        judge_model="qwq:latest",          # QwQ - reasoning specialist
        temperature_a=0.1,                   # Consistent
        temperature_b=0.8,                   # More diverse
        temperature_judge=0.0,             # Deterministic
        gamma=0.5,
        random_seed=seed                     # Deterministic results
    )

    # Validate all models are present before starting
    required_models = [config.verifier_a_model, config.verifier_b_model, config.judge_model]
    models_present, missing_models = validate_models_present(required_models)

    if not models_present:
        print(f"❌ Missing models: {missing_models}")
        print(generate_pull_commands(missing_models))
        return False

    print(f"Model Configuration:")
    print(f"  Verifier A: {config.verifier_a_model} (temp={config.temperature_a})")
    print(f"  Verifier B: {config.verifier_b_model} (temp={config.temperature_b})")
    print(f"  Judge:    {config.judge_model} (temp={config.temperature_judge})")
    print(f"  Gamma:      {config.gamma}")
    print()

    # Create DTE system
    try:
        verifier_a_lm, verifier_b_lm, judge_lm = config.create_language_models()
        dte = DTESystem(verifier_a_lm, verifier_b_lm, judge_lm, gamma=config.gamma)
        print("✅ DTE system initialized with real models")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return False

    # Add labeled factual claims for accuracy measurement (BoolQ/FEVER style)
    labeled_claims = [
        ("The capital of France is Paris", "yes"),
        ("Water boils at 100°C at sea level", "yes"),
        ("The Earth has two moons", "no"),
        ("2 + 2 equals 4", "yes"),
        ("The Pacific Ocean is the smallest ocean", "no"),
        ("Humans have 46 chromosomes", "yes"),
        ("The sun is a planet", "no"),
        ("Shakespeare wrote Romeo and Juliet", "yes"),
        ("1 + 1 equals 3", "no"),
        ("Tokyo is the capital of Japan", "yes")
    ]

    # Test contested factual claims (verifiable but disputed)
    contested_factual_claims = [
        "Nuclear energy produces fewer carbon emissions than solar power over its lifetime",
        "GMO foods have been proven safe for human consumption by scientific consensus",
        "Climate change is primarily caused by human activities according to 97% of climate scientists",
        "Vaccines do not cause autism according to multiple large-scale studies",
        "The gender pay gap in the US is approximately 20% when controlling for job type and experience",
        "Artificial intelligence will likely surpass human cognitive abilities within 30 years",
        "Dark matter comprises approximately 85% of all matter in the universe"
    ]

    print(f"\n📚 Testing {len(labeled_claims)} labeled factual claims:")
    print("-" * 50)

    # Test labeled claims first for accuracy measurement
    labeled_results = []
    labeled_escalations = 0
    correct_predictions = 0

    for i, (claim, expected) in enumerate(labeled_claims, 1):
        print(f"\n[{i}] Claim: '{claim}' (Expected: {expected})")

        start_time = time.time()

        try:
            # Add timeout guardrail for slow claims
            with timeout_context(60):  # 60 second timeout per claim
                result = dte.evaluate_claim(claim)
            elapsed = time.time() - start_time

            if result.escalated:
                labeled_escalations += 1

            # Check accuracy
            predicted = result.final_verdict
            correct = (predicted == expected)
            if correct:
                correct_predictions += 1

            # Get confidence scores
            a_conf = result.verifier_a_result.confidence
            b_conf = result.verifier_b_result.confidence
            judge_conf = result.judge_result.confidence if result.judge_result else None

            labeled_result = {
                'claim': claim,
                'expected': expected,
                'predicted': predicted,
                'correct': correct,
                'escalated': result.escalated,
                'time': elapsed,
                'a_confidence': a_conf,
                'b_confidence': b_conf,
                'ref_confidence': judge_conf
            }
            labeled_results.append(labeled_result)

            print(f"    Expected: {expected} | Got: {predicted} | Correct: {correct} | Escalated: {result.escalated} | Time: {elapsed:.1f}s")

        except TimeoutError as e:
            print(f"    ⏰ Timeout: {e}")
            labeled_results.append({'claim': claim, 'expected': expected, 'error': 'timeout', 'correct': False, 'escalated': False})
        except Exception as e:
            print(f"    ❌ Error: {e}")
            labeled_results.append({'claim': claim, 'expected': expected, 'error': str(e), 'correct': False, 'escalated': False})

    # Calculate accuracy metrics
    labeled_accuracy = correct_predictions / len(labeled_claims) if labeled_claims else 0
    labeled_escalation_rate = labeled_escalations / len(labeled_claims) if labeled_claims else 0

    print(f"\n📊 LABELED CLAIMS SUMMARY:")
    print(f"  Accuracy: {labeled_accuracy:.1%} ({correct_predictions}/{len(labeled_claims)})")
    print(f"  Escalation rate: {labeled_escalation_rate:.1%}")

    print(f"\n🌶️  Testing {len(contested_factual_claims)} contested factual claims:")
    print("-" * 50)

    escalation_count = 0
    results = []

    for i, claim in enumerate(contested_factual_claims, 1):
        print(f"\n[{i}] Claim: '{claim}'")

        start_time = time.time()

        try:
            # Add timeout guardrail for slow claims
            with timeout_context(60):  # 60 second timeout per claim
                result = dte.evaluate_claim(claim)
            elapsed = time.time() - start_time

            if result.escalated:
                escalation_count += 1

            # Get confidence scores
            a_conf = result.verifier_a_result.confidence
            b_conf = result.verifier_b_result.confidence
            judge_conf = result.judge_result.confidence if result.judge_result else None

            # Store result with reasoning to avoid double-compute
            claim_result = {
                'claim': claim,
                'verifier_a': result.verifier_a_result.prediction,
                'verifier_b': result.verifier_b_result.prediction,
                'final': result.final_prediction,
                'escalated': result.escalated,
                'a_confidence': a_conf,
                'b_confidence': b_conf,
                'ref_confidence': judge_conf,
                'time': elapsed,
                'a_reasoning': result.verifier_a_result.reasoning[:150] + "..." if len(result.verifier_a_result.reasoning) > 150 else result.verifier_a_result.reasoning,
                'b_reasoning': result.verifier_b_result.reasoning[:150] + "..." if len(result.verifier_b_result.reasoning) > 150 else result.verifier_b_result.reasoning
            }
            results.append(claim_result)

            # Display result - use consistent API
            a_verdict = "yes" if result.verifier_a_result.prediction == 1 else "no"
            b_verdict = "yes" if result.verifier_b_result.prediction == 1 else "no"
            final_verdict = result.final_verdict  # This is the property method

            print(f"    Verifier A: {a_verdict} (confidence: {a_conf:.2f})")
            print(f"    Verifier B: {b_verdict} (confidence: {b_conf:.2f})")
            if result.judge_result:
                judge_verdict = "yes" if result.judge_result.prediction == 1 else "no"
                print(f"    Judge:    {judge_verdict} (confidence: {judge_conf:.2f})")
            print(f"    → Final: {final_verdict} | Escalated: {result.escalated} | Time: {elapsed:.1f}s")

        except TimeoutError as e:
            print(f"    ⏰ Timeout: {e}")
            results.append({'claim': claim, 'error': 'timeout', 'escalated': False})
        except Exception as e:
            print(f"    ❌ Error: {e}")
            results.append({'claim': claim, 'error': str(e), 'escalated': False})

    # Calculate metrics
    successful_results = [r for r in results if 'error' not in r]
    escalation_rate = escalation_count / len(successful_results) if successful_results else 0

    avg_time = sum(r['time'] for r in successful_results) / len(successful_results) if successful_results else 0

    # System metrics
    metrics = dte.get_metrics()

    print(f"\n📊 COMPREHENSIVE TEST RESULTS:")
    print("=" * 50)
    print(f"LABELED CLAIMS (Accuracy Measurement):")
    print(f"  Total: {len(labeled_claims)}")
    print(f"  Accuracy: {labeled_accuracy:.1%}")
    print(f"  Escalation rate: {labeled_escalation_rate:.1%}")
    print(f"\nCONTESTED FACTUAL CLAIMS (Disagreement Detection):")
    print(f"  Total: {len(contested_factual_claims)}")
    print(f"  Successful evaluations: {len(successful_results)}")
    print(f"  Escalations: {escalation_count}")
    print(f"  Escalation rate: {escalation_rate:.1%}")
    print(f"  Average time per claim: {avg_time:.1f}s")
    print(f"\nSYSTEM METRICS:")
    print(f"  Total model calls: {metrics.get('total_model_calls', 0)}")
    print(f"  Average calls per evaluation: {metrics.get('avg_calls_per_evaluation', 0):.1f}")

    # Analyze disagreement patterns
    agreements = sum(1 for r in successful_results if r['verifier_a'] == r['verifier_b'])
    disagreements = len(successful_results) - agreements

    print(f"\nDISAGREEMENT ANALYSIS:")
    print(f"Agreements: {agreements}")
    print(f"Disagreements: {disagreements}")
    print(f"Disagreement rate: {disagreements/len(successful_results):.1%}" if successful_results else "N/A")

    # Confidence analysis
    if successful_results:
        avg_a_conf = sum(r['a_confidence'] for r in successful_results) / len(successful_results)
        avg_b_conf = sum(r['b_confidence'] for r in successful_results) / len(successful_results)

        print(f"\nCONFIDENCE ANALYSIS:")
        print(f"Average Verifier A confidence: {avg_a_conf:.2f}")
        print(f"Average Verifier B confidence: {avg_b_conf:.2f}")
        print(f"Overall average confidence: {(avg_a_conf + avg_b_conf)/2:.2f}")

    print(f"\nKEY FINDINGS:")
    if escalation_rate > 0.6:
        print("✅ High escalation rate - system is aggressive on controversial content")
    if disagreements > agreements:
        print("✅ Models disagree frequently on controversial claims")
    if avg_time < 30:
        print("✅ Reasonable response times with real models")

    # Show reasoning examples from stored results (no double-compute)
    print(f"\nREASONING EXAMPLES:")
    for i, result in enumerate(successful_results[:2]):  # Show first 2
        if 'error' not in result and 'a_reasoning' in result and 'b_reasoning' in result:
            print(f"\nClaim: '{result['claim']}'")
            print(f"  Verifier A reasoning: {result['a_reasoning']}")
            print(f"  Verifier B reasoning: {result['b_reasoning']}")

    # Save results to JSONL for analysis and regression testing
    timestamp = int(time.time())
    os.makedirs("results", exist_ok=True)
    results_file = f"results/dte_test_{timestamp}_seed{seed}.jsonl"

    # Combine all results with metadata
    all_results = {
        'metadata': {
            'timestamp': timestamp,
            'seed': seed,
            'models': {
                'verifier_a': config.verifier_a_model,
                'verifier_b': config.verifier_b_model,
                'referee': config.judge_model
            },
            'gamma': config.gamma,
            'temperatures': {
                'verifier_a': config.temperature_a,
                'verifier_b': config.temperature_b,
                'referee': config.temperature_judge
            }
        },
        'labeled_claims': {
            'results': labeled_results,
            'accuracy': labeled_accuracy,
            'escalation_rate': labeled_escalation_rate
        },
        'contested_factual_claims': {
            'results': results,
            'escalation_rate': escalation_rate,
            'disagreement_rate': disagreements/len(successful_results) if successful_results else 0
        },
        'system_metrics': metrics
    }

    # Save as JSONL (one line per claim + metadata)
    with open(results_file, 'w') as f:
        # Write metadata first
        f.write(json.dumps(all_results['metadata']) + '\n')

        # Write labeled results
        for result in labeled_results:
            result['type'] = 'labeled'
            f.write(json.dumps(result) + '\n')

        # Write contested factual results
        for result in results:
            result['type'] = 'contested_factual'
            f.write(json.dumps(result) + '\n')

        # Write summary metrics
        summary = {
            'type': 'summary',
            'labeled_accuracy': labeled_accuracy,
            'contested_factual_escalation_rate': escalation_rate,
            'total_model_calls': metrics.get('total_model_calls', 0)
        }
        f.write(json.dumps(summary) + '\n')

    print(f"\n💾 Results saved to: {results_file}")
    print(f"   Use for regression testing and analysis")

    return True

def main():
    """Run fast real model test."""
    try:
        success = test_real_models()
        if success:
            print(f"\n🎉 Real model test completed successfully!")
        return success
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)