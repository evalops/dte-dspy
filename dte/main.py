"""
Main entry point for the DTE framework.

This module provides the command-line interface for running
disagreement-triggered escalation experiments.
"""

import argparse
import sys
import json
import time
from pathlib import Path
from typing import Optional, List

from .config import DTEConfig, setup_logging, validate_ollama_connection
from .core import DTESystem
from .evaluation import DTEEvaluator
from .data import load_dataset


def main(args: Optional[List[str]] = None) -> int:
    """
    Main entry point for the CLI.
    
    Args:
        args: Optional command line arguments (for testing)
        
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    parser = argparse.ArgumentParser(
        description="DTE: Disagreement-Triggered Escalation Framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Experiment parameters
    parser.add_argument(
        "--gamma", type=float, default=None,
        help="Confidence threshold for escalation (0-1)"
    )
    parser.add_argument(
        "--dataset", type=str, default="test",
        help="Dataset to evaluate (test, edge_cases, controversial, or file path)"
    )
    parser.add_argument(
        "--dataset-size", type=int, default=None,
        help="Limit dataset size"
    )
    
    # Model configuration
    parser.add_argument(
        "--verifier-a", type=str, default=None,
        help="Model for verifier A"
    )
    parser.add_argument(
        "--verifier-b", type=str, default=None,
        help="Model for verifier B"
    )
    parser.add_argument(
        "--referee", type=str, default=None,
        help="Model for referee"
    )
    parser.add_argument(
        "--api-base", type=str, default=None,
        help="Ollama API base URL"
    )
    
    # Experiment modes
    parser.add_argument(
        "--gamma-sweep", action="store_true",
        help="Run gamma threshold sweep"
    )
    parser.add_argument(
        "--gamma-values", type=str, default=None,
        help="Comma-separated gamma values for sweep (e.g., '0.1,0.5,0.9')"
    )
    parser.add_argument(
        "--ablation", action="store_true",
        help="Run full ablation study"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Run quick evaluation (10 claims)"
    )
    
    # Output configuration
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output file for results (JSON)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Output directory"
    )
    parser.add_argument(
        "--report", action="store_true",
        help="Generate detailed report"
    )
    
    # Logging and debugging
    parser.add_argument(
        "--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO", help="Logging level"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output"
    )
    
    # Parse arguments
    parsed_args = parser.parse_args(args)
    
    # Setup logging
    setup_logging(parsed_args.log_level)
    
    try:
        # Load configuration
        config = DTEConfig.from_env()
        
        # Apply command line overrides
        if parsed_args.gamma is not None:
            config.gamma = parsed_args.gamma
        if parsed_args.verifier_a:
            config.verifier_a_model = parsed_args.verifier_a
        if parsed_args.verifier_b:
            config.verifier_b_model = parsed_args.verifier_b
        if parsed_args.referee:
            config.referee_model = parsed_args.referee
        if parsed_args.api_base:
            config.api_base = parsed_args.api_base
        if parsed_args.dataset_size:
            config.dataset_size = parsed_args.dataset_size
        if parsed_args.output_dir:
            config.output_dir = parsed_args.output_dir
        if parsed_args.verbose:
            config.verbose = True
        
        # Validate Ollama connection
        if not validate_ollama_connection(config.api_base):
            print(f"❌ Cannot connect to Ollama at {config.api_base}")
            print("Please ensure Ollama is running: ollama serve")
            return 1
        
        if not parsed_args.quiet:
            print("✅ Connected to Ollama")
        
        # Create DTE system
        verifier_a_lm, verifier_b_lm, referee_lm = config.create_language_models()
        dte_system = DTESystem(
            verifier_a_lm=verifier_a_lm,
            verifier_b_lm=verifier_b_lm,
            referee_lm=referee_lm,
            gamma=config.gamma,
            use_cot=config.use_cot
        )
        
        if not parsed_args.quiet:
            print(f"🎯 DTE System initialized:")
            print(f"   Verifier A: {config.verifier_a_model}")
            print(f"   Verifier B: {config.verifier_b_model}")
            print(f"   Referee: {config.referee_model}")
            print(f"   Gamma: {config.gamma}")
        
        # Create evaluator
        evaluator = DTEEvaluator(dte_system)
        
        # Run experiment based on mode
        start_time = time.time()
        
        if parsed_args.quick:
            # Quick evaluation
            if not parsed_args.quiet:
                print("\n🚀 Running quick evaluation...")
            
            from .evaluation import run_quick_evaluation
            results = run_quick_evaluation(dte_system, 
                                         num_claims=10,
                                         dataset_name=parsed_args.dataset)
            
        elif parsed_args.gamma_sweep:
            # Gamma sweep
            if not parsed_args.quiet:
                print(f"\n📊 Running gamma sweep on {parsed_args.dataset}...")
            
            gamma_values = None
            if parsed_args.gamma_values:
                gamma_values = [float(x.strip()) for x in parsed_args.gamma_values.split(',')]
            
            claims, ground_truth = load_dataset(parsed_args.dataset, config.dataset_size)
            results_df = evaluator.gamma_sweep(claims, ground_truth, gamma_values, 
                                             verbose=not parsed_args.quiet)
            results = results_df.to_dict('records')
            
        elif parsed_args.ablation:
            # Full ablation study
            if not parsed_args.quiet:
                print(f"\n🔬 Running ablation study on {parsed_args.dataset}...")
            
            results = evaluator.run_ablation_study(
                dataset_name=parsed_args.dataset,
                dataset_size=config.dataset_size
            )
            
        else:
            # Single evaluation
            if not parsed_args.quiet:
                print(f"\n🎯 Evaluating {parsed_args.dataset}...")
            
            claims, ground_truth = load_dataset(parsed_args.dataset, config.dataset_size)
            dte_results = evaluator.evaluate_dataset(claims, ground_truth,
                                                    parsed_args.dataset,
                                                    verbose=not parsed_args.quiet)
            
            # Get metrics
            metrics = dte_system.get_metrics()
            results = {
                'dataset_info': {
                    'name': parsed_args.dataset,
                    'size': len(claims)
                },
                'metrics': metrics,
                'detailed_results': [r.__dict__ for r in dte_results] if config.save_detailed_results else None
            }
        
        end_time = time.time()
        
        # Generate report if requested
        if parsed_args.report and isinstance(results, dict) and 'sweep_results' in results:
            import pandas as pd
            sweep_df = pd.DataFrame(results['sweep_results'])
            report = evaluator.generate_report(sweep_df, results.get('dataset_info'))
            print(f"\n{report}")
        
        # Save results
        if parsed_args.output:
            output_path = Path(parsed_args.output)
        else:
            timestamp = int(time.time())
            output_path = Path(config.output_dir) / f"dte_results_{timestamp}.json"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add metadata
        output_data = {
            'results': results,
            'config': config.to_dict(),
            'metadata': {
                'timestamp': time.time(),
                'duration': end_time - start_time,
                'command_line_args': vars(parsed_args)
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        if not parsed_args.quiet:
            print(f"\n💾 Results saved to {output_path}")
            print(f"⏱️  Completed in {end_time - start_time:.1f}s")
        
        # Print summary
        if isinstance(results, dict) and 'metrics' in results:
            metrics = results['metrics']
            print(f"\n📈 Summary:")
            print(f"   Accuracy: {metrics.get('overall_accuracy', 0):.1%}")
            print(f"   Escalation rate: {metrics.get('escalation_rate', 0):.1%}")
            print(f"   Average cost: {metrics.get('avg_calls_per_evaluation', 0):.1f} calls/claim")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n❌ Experiment interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        if parsed_args.log_level == "DEBUG":
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())