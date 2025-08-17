#!/usr/bin/env python3
"""
Example Usage of DTE System

This script demonstrates various ways to use the Disagreement-Triggered Escalation
system for fact verification.
"""

import sys
import json
from typing import List

from scripts.dte_core import DTESystem, create_simple_test_models
from scripts.evaluation_harness import DTEEvaluator, create_example_dataset
from scripts.ollama_integration import create_ollama_dte_system, get_available_ollama_models


def example_basic_usage():
    """Basic DTE system usage example."""
    print("=== Basic DTE Usage Example ===")
    
    # Create DTE system with simple test models
    print("Creating DTE system...")
    model_a, model_b, referee = create_simple_test_models()
    dte = DTESystem(verifier_a_lm=model_a, verifier_b_lm=model_b, referee_lm=referee, default_gamma=0.7)
    
    # Test claims
    test_claims = [
        "The capital of France is Paris.",
        "The Earth is flat.", 
        "Python was created by Guido van Rossum.",
        "The moon is made of cheese.",
        "Water boils at 100°C at sea level."
    ]
    
    print("\nEvaluating test claims:")
    for claim in test_claims:
        result = dte.evaluate_claim(claim)
        print(f"Claim: {claim}")
        print(f"  → Prediction: {'TRUE' if result.final_prediction else 'FALSE'}")
        print(f"  → Escalated: {'YES' if result.escalated else 'NO'}")
        print(f"  → Verifier A: pred={result.verifier_a_result.prediction}, conf={result.verifier_a_result.confidence:.2f}")
        print(f"  → Verifier B: pred={result.verifier_b_result.prediction}, conf={result.verifier_b_result.confidence:.2f}")
        if result.escalated:
            print(f"  → Referee: pred={result.referee_result.prediction}, conf={result.referee_result.confidence:.2f}")
        print()
    
    # Show final metrics
    metrics = dte.get_metrics()
    print("Final Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")


def example_gamma_sweep():
    """Gamma threshold sweep analysis example."""
    print("\n=== Gamma Sweep Analysis Example ===")
    
    # Create DTE system
    model_a, model_b, referee = create_simple_test_models()
    dte = DTESystem(verifier_a_lm=model_a, verifier_b_lm=model_b, referee_lm=referee)
    
    # Create evaluator
    evaluator = DTEEvaluator(dte)
    
    # Create test dataset
    print("Creating synthetic dataset...")
    dataset = evaluator.create_synthetic_dataset(30)  # Smaller for demo
    
    # Run gamma sweep
    print("Running gamma sweep analysis...")
    gamma_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    results = evaluator.gamma_sweep_evaluation(dataset, gamma_values, verbose=True)
    
    # Display results
    print("\nGamma Sweep Results:")
    print(f"{'Gamma':<8} {'Accuracy':<10} {'Esc.Rate':<10} {'Avg.Calls':<10} {'FCR':<8}")
    print("-" * 50)
    for _, row in results.iterrows():
        print(f"{row['gamma']:<8.1f} {row['accuracy']:<10.3f} {row['escalation_rate']:<10.3f} "
              f"{row['avg_calls_per_evaluation']:<10.2f} {row['false_consensus_rate']:<8.3f}")
    
    # Generate and display report
    report = evaluator.generate_report(results)
    print(f"\n{report}")
    
    # Save results
    evaluator.save_results(results, "example_gamma_sweep.csv")
    print("\nResults saved to example_gamma_sweep.csv")


def example_ollama_integration():
    """Ollama integration example."""
    print("\n=== Ollama Integration Example ===")
    
    # Check available models
    available_models = get_available_ollama_models()
    print(f"Available Ollama models: {available_models}")
    
    if not available_models:
        print("No Ollama models found. Please install Ollama and pull some models:")
        print("  ollama pull llama2:7b")
        print("  ollama pull mistral:7b")
        return
    
    # Try to create DTE system with available models
    try:
        # Use available models or fallback to smaller ones
        available_set = set(available_models)
        
        if "llama2:7b" in available_set:
            verifier_a = "llama2:7b"
        elif any("llama2" in m for m in available_models):
            verifier_a = next(m for m in available_models if "llama2" in m)
        else:
            verifier_a = available_models[0]
            
        if "mistral:7b" in available_set:
            verifier_b = "mistral:7b"
        elif any("mistral" in m for m in available_models):
            verifier_b = next(m for m in available_models if "mistral" in m)
        else:
            verifier_b = available_models[-1] if len(available_models) > 1 else available_models[0]
            
        referee = available_models[0]  # Use any available model as referee
        
        print(f"Creating DTE system with:")
        print(f"  Verifier A: {verifier_a}")
        print(f"  Verifier B: {verifier_b}")
        print(f"  Referee: {referee}")
        
        dte = create_ollama_dte_system(
            verifier_a_model=verifier_a,
            verifier_b_model=verifier_b,
            judge_model=referee,
            ensure_models=False  # Don't auto-pull for demo
        )
        
        # Test with simple claims
        test_claims = [
            "The capital of France is Paris.",
            "The Earth is flat."
        ]
        
        print("\nTesting with Ollama models:")
        for claim in test_claims:
            print(f"Evaluating: {claim}")
            try:
                result = dte.evaluate_claim(claim)
                print(f"  Result: {'TRUE' if result.final_prediction else 'FALSE'}")
                print(f"  Escalated: {result.escalated}")
            except Exception as e:
                print(f"  Error: {e}")
            print()
            
    except Exception as e:
        print(f"Error creating Ollama DTE system: {e}")
        print("This is normal if Ollama is not running or models are not available.")


def example_custom_dataset():
    """Custom dataset evaluation example."""
    print("\n=== Custom Dataset Example ===")
    
    # Create DTE system
    model_a, model_b, referee = create_simple_test_models()
    dte = DTESystem(verifier_a_lm=model_a, verifier_b_lm=model_b, referee_lm=referee)
    evaluator = DTEEvaluator(dte)
    
    # Custom claims about programming
    programming_claims = [
        "Python is a programming language.",
        "JavaScript runs only in browsers.",
        "Git is a version control system.", 
        "HTML is a programming language.",
        "Linux is an operating system.",
        "CSS is used for database queries.",
        "React is a JavaScript library.",
        "SQL is used for styling web pages.",
        "Docker is a containerization platform.",
        "JSON stands for JavaScript Only Notation."
    ]
    
    programming_labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    
    # Load custom dataset
    dataset = evaluator.load_dataset(
        programming_claims, 
        programming_labels, 
        "programming_facts"
    )
    
    print(f"Created custom dataset with {len(dataset)} programming-related claims")
    
    # Evaluate with different gamma values
    results = evaluator.gamma_sweep_evaluation(
        dataset, 
        gamma_values=[0.5, 0.7, 0.9]
    )
    
    print("\nCustom Dataset Results:")
    for _, row in results.iterrows():
        print(f"γ={row['gamma']:.1f}: Accuracy={row['accuracy']:.3f}, "
              f"Escalation Rate={row['escalation_rate']:.3f}")


def example_interactive_mode():
    """Interactive claim evaluation mode."""
    print("\n=== Interactive Mode Example ===")
    
    # Create DTE system
    model_a, model_b, referee = create_simple_test_models()
    dte = DTESystem(verifier_a_lm=model_a, verifier_b_lm=model_b, referee_lm=referee, default_gamma=0.7)
    
    print("Interactive DTE System")
    print("Enter claims to evaluate (or 'quit' to exit):")
    print("Example: 'The capital of Japan is Tokyo.'")
    
    while True:
        try:
            claim = input("\nClaim: ").strip()
            
            if claim.lower() in ['quit', 'exit', 'q']:
                break
                
            if not claim:
                continue
                
            # Evaluate claim
            result = dte.evaluate_claim(claim)
            
            print(f"Prediction: {'TRUE' if result.final_prediction else 'FALSE'}")
            print(f"Escalated: {result.escalated}")
            print(f"Confidence: A={result.verifier_a_result.confidence:.2f}, "
                  f"B={result.verifier_b_result.confidence:.2f}")
            
            if result.escalated:
                print(f"Referee confidence: {result.referee_result.confidence:.2f}")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nSession complete!")
    
    # Show session metrics
    metrics = dte.get_metrics()
    print(f"Session summary: {metrics['total_evaluations']} evaluations, "
          f"{metrics['escalations']} escalations")


def main():
    """Run all examples."""
    print("DTE System Examples")
    print("=" * 50)
    
    examples = [
        ("basic", "Basic usage", example_basic_usage),
        ("gamma", "Gamma sweep analysis", example_gamma_sweep),
        ("ollama", "Ollama integration", example_ollama_integration),
        ("custom", "Custom dataset", example_custom_dataset),
        ("interactive", "Interactive mode", example_interactive_mode),
    ]
    
    if len(sys.argv) > 1:
        # Run specific example
        example_name = sys.argv[1].lower()
        for name, desc, func in examples:
            if name.startswith(example_name):
                print(f"Running: {desc}")
                func()
                return
        print(f"Unknown example: {example_name}")
        print("Available examples:", [name for name, _, _ in examples])
    else:
        # Run all examples except interactive
        for name, desc, func in examples:
            if name != "interactive":  # Skip interactive for batch run
                print(f"\nRunning: {desc}")
                try:
                    func()
                except Exception as e:
                    print(f"Error in {desc}: {e}")
                    
        print("\n" + "=" * 50)
        print("All examples completed!")
        print("Run 'python example_usage.py interactive' for interactive mode")


if __name__ == "__main__":
    main()