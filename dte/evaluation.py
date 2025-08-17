"""
Evaluation harness for DTE experiments.

This module provides comprehensive evaluation capabilities including
gamma threshold sweeping, accuracy vs cost analysis, and statistical reporting.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging

from .core import DTESystem, DTEResult
from .data import load_dataset, validate_dataset, normalize_labels
from .metrics import calculate_metrics, MetricsResult

logger = logging.getLogger(__name__)


class DTEEvaluator:
    """Comprehensive evaluator for DTE systems."""
    
    def __init__(self, dte_system: DTESystem):
        self.dte_system = dte_system
        self.results_history: List[Dict[str, Any]] = []
        
    def evaluate_single(self, 
                       claim: str, 
                       ground_truth: Optional[str] = None,
                       verbose: bool = False) -> DTEResult:
        """Evaluate a single claim."""
        if verbose:
            print(f"Evaluating: {claim[:60]}...")
            
        result = self.dte_system.evaluate_claim(claim, ground_truth)
        
        if verbose:
            status = "🔥 ESCALATED" if result.escalated else "🤝 CONSENSUS"
            print(f"  Result: {result.final_verdict} [{status}]")
            
        return result
    
    def evaluate_dataset(self,
                        claims: List[str],
                        ground_truth: List[str],
                        dataset_name: str = "custom",
                        verbose: bool = True) -> List[DTEResult]:
        """
        Evaluate a complete dataset.
        
        Args:
            claims: List of claims to evaluate
            ground_truth: List of ground truth labels
            dataset_name: Name for the dataset
            verbose: Whether to show progress
            
        Returns:
            List of DTEResult objects
        """
        validate_dataset(claims, ground_truth)
        ground_truth = normalize_labels(ground_truth)
        
        results = []
        start_time = time.time()
        
        iterator = enumerate(zip(claims, ground_truth))
        if verbose:
            iterator = tqdm(iterator, total=len(claims), 
                          desc=f"Evaluating {dataset_name}")
        
        for i, (claim, gt) in iterator:
            result = self.dte_system.evaluate_claim(claim, gt)
            results.append(result)
            
            if verbose and i > 0 and i % 10 == 0:
                # Show intermediate stats
                metrics = self.dte_system.get_metrics()
                escalation_rate = metrics.get('escalation_rate', 0)
                accuracy = metrics.get('overall_accuracy', 0)
                print(f"  Progress: {escalation_rate:.1%} escalation, {accuracy:.1%} accuracy")
        
        end_time = time.time()
        
        # Store evaluation metadata
        evaluation_info = {
            'dataset_name': dataset_name,
            'dataset_size': len(claims),
            'evaluation_time': end_time - start_time,
            'timestamp': time.time(),
            'final_metrics': self.dte_system.get_metrics()
        }
        self.results_history.append(evaluation_info)
        
        if verbose:
            print(f"Completed in {end_time - start_time:.1f}s")
        
        return results
    
    def gamma_sweep(self,
                   claims: List[str],
                   ground_truth: List[str],
                   gamma_values: Optional[List[float]] = None,
                   verbose: bool = True) -> pd.DataFrame:
        """
        Evaluate across different gamma thresholds.
        
        Args:
            claims: Claims to evaluate
            ground_truth: Ground truth labels
            gamma_values: List of gamma values to test
            verbose: Whether to show progress
            
        Returns:
            DataFrame with results for each gamma
        """
        if gamma_values is None:
            gamma_values = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
        
        validate_dataset(claims, ground_truth)
        ground_truth = normalize_labels(ground_truth)
        
        results = []
        
        for gamma in tqdm(gamma_values, desc="Gamma sweep", disable=not verbose):
            if verbose:
                print(f"\nEvaluating γ={gamma:.2f}")
            
            # Reset system with new gamma
            self.dte_system.gamma = gamma
            self.dte_system.reset_metrics()
            
            # Evaluate dataset
            dte_results = []
            for claim, gt in zip(claims, ground_truth):
                result = self.dte_system.evaluate_claim(claim, gt)
                dte_results.append(result)
            
            # Calculate metrics
            metrics = calculate_metrics(dte_results, ground_truth)
            gamma_result = {
                'gamma': gamma,
                **metrics.to_dict()
            }
            results.append(gamma_result)
            
            if verbose:
                print(f"  Accuracy: {metrics.overall_accuracy:.1%}, "
                      f"Escalation: {metrics.escalation_rate:.1%}, "
                      f"Cost: {metrics.avg_calls_per_evaluation:.1f}")
        
        return pd.DataFrame(results)
    
    def run_ablation_study(self,
                          dataset_name: str = "test",
                          dataset_size: Optional[int] = None,
                          gamma_values: Optional[List[float]] = None) -> Dict[str, Any]:
        """Run comprehensive ablation study."""
        logger.info(f"Running ablation study on {dataset_name}")
        
        # Load dataset
        claims, ground_truth = load_dataset(dataset_name, dataset_size)
        
        # Run gamma sweep
        sweep_results = self.gamma_sweep(claims, ground_truth, gamma_values)
        
        # Find optimal configurations
        best_accuracy = sweep_results.loc[sweep_results['overall_accuracy'].idxmax()]
        best_efficiency = sweep_results.loc[
            (sweep_results['overall_accuracy'] / sweep_results['avg_calls_per_evaluation']).idxmax()
        ]
        lowest_cost = sweep_results.loc[sweep_results['avg_calls_per_evaluation'].idxmin()]
        
        summary = {
            'dataset_info': {
                'name': dataset_name,
                'size': len(claims),
                'balance': {
                    'positive': sum(1 for gt in ground_truth if gt == "yes"),
                    'negative': sum(1 for gt in ground_truth if gt == "no")
                }
            },
            'optimal_configs': {
                'best_accuracy': {
                    'gamma': best_accuracy['gamma'],
                    'accuracy': best_accuracy['overall_accuracy'],
                    'escalation_rate': best_accuracy['escalation_rate']
                },
                'best_efficiency': {
                    'gamma': best_efficiency['gamma'], 
                    'accuracy': best_efficiency['overall_accuracy'],
                    'cost': best_efficiency['avg_calls_per_evaluation'],
                    'efficiency': best_efficiency['overall_accuracy'] / best_efficiency['avg_calls_per_evaluation']
                },
                'lowest_cost': {
                    'gamma': lowest_cost['gamma'],
                    'cost': lowest_cost['avg_calls_per_evaluation'],
                    'accuracy': lowest_cost['overall_accuracy']
                }
            },
            'sweep_results': sweep_results.to_dict('records')
        }
        
        return summary
    
    def generate_report(self, 
                       results_df: pd.DataFrame,
                       dataset_info: Optional[Dict[str, Any]] = None) -> str:
        """Generate comprehensive evaluation report."""
        report = []
        report.append("=" * 80)
        report.append("DTE SYSTEM EVALUATION REPORT")
        report.append("=" * 80)
        
        if dataset_info:
            report.append(f"\nDataset: {dataset_info.get('name', 'Unknown')}")
            report.append(f"Size: {dataset_info.get('size', 'Unknown')} claims")
            if 'balance' in dataset_info:
                balance = dataset_info['balance']
                report.append(f"Balance: {balance['positive']} positive, {balance['negative']} negative")
        
        report.append(f"\nGamma values tested: {len(results_df)}")
        report.append(f"Range: {results_df['gamma'].min():.2f} - {results_df['gamma'].max():.2f}")
        
        # Performance summary
        report.append(f"\n📊 PERFORMANCE SUMMARY")
        report.append(f"Accuracy range: {results_df['overall_accuracy'].min():.1%} - {results_df['overall_accuracy'].max():.1%}")
        report.append(f"Escalation range: {results_df['escalation_rate'].min():.1%} - {results_df['escalation_rate'].max():.1%}")
        report.append(f"Cost range: {results_df['avg_calls_per_evaluation'].min():.1f} - {results_df['avg_calls_per_evaluation'].max():.1f} calls/claim")
        
        # Optimal configurations
        best_accuracy = results_df.loc[results_df['overall_accuracy'].idxmax()]
        best_efficiency = results_df.loc[
            (results_df['overall_accuracy'] / results_df['avg_calls_per_evaluation']).idxmax()
        ]
        
        report.append(f"\n🎯 OPTIMAL CONFIGURATIONS")
        report.append(f"Best Accuracy: γ={best_accuracy['gamma']:.2f} → {best_accuracy['overall_accuracy']:.1%} accuracy")
        report.append(f"Best Efficiency: γ={best_efficiency['gamma']:.2f} → {best_efficiency['overall_accuracy']:.1%} accuracy, {best_efficiency['avg_calls_per_evaluation']:.1f} calls/claim")
        
        # Insights
        report.append(f"\n💡 KEY INSIGHTS")
        
        # Escalation patterns
        high_escalation = results_df[results_df['escalation_rate'] > 0.3]
        if len(high_escalation) > 0:
            report.append(f"• High escalation (>30%) occurs at γ ≤ {high_escalation['gamma'].max():.2f}")
        
        # Accuracy-cost trade-offs
        efficient_configs = results_df[results_df['overall_accuracy'] > results_df['overall_accuracy'].quantile(0.8)]
        if len(efficient_configs) > 0:
            min_cost_efficient = efficient_configs['avg_calls_per_evaluation'].min()
            report.append(f"• Top 20% accuracy achieved with {min_cost_efficient:.1f}+ calls/claim")
        
        # Diminishing returns
        if len(results_df) > 3:
            accuracy_gains = results_df.sort_values('gamma')['overall_accuracy'].diff()
            if accuracy_gains.iloc[-1] < 0.01:  # Less than 1% gain
                report.append(f"• Diminishing returns beyond γ={results_df.sort_values('gamma')['gamma'].iloc[-2]:.2f}")
        
        report.append(f"\n" + "=" * 80)
        
        return "\n".join(report)
    
    def save_results(self,
                    results: Any,
                    file_path: Path,
                    include_metadata: bool = True) -> None:
        """Save evaluation results to file."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(results, pd.DataFrame):
            data = results.to_dict('records')
        elif isinstance(results, list):
            data = [r.__dict__ if hasattr(r, '__dict__') else r for r in results]
        else:
            data = results
        
        output = {
            'results': data,
            'metadata': {
                'timestamp': time.time(),
                'system_config': self.dte_system.gamma,
                'evaluation_history': self.results_history
            } if include_metadata else {}
        }
        
        if file_path.suffix == '.json':
            with open(file_path, 'w') as f:
                json.dump(output, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        logger.info(f"Results saved to {file_path}")


def run_quick_evaluation(dte_system: DTESystem,
                        num_claims: int = 10,
                        dataset_name: str = "test") -> Dict[str, Any]:
    """Run a quick evaluation for testing/debugging."""
    evaluator = DTEEvaluator(dte_system)
    
    # Load small dataset
    claims, ground_truth = load_dataset(dataset_name, num_claims)
    
    # Evaluate
    results = evaluator.evaluate_dataset(claims, ground_truth, 
                                       dataset_name=f"{dataset_name}_quick")
    
    # Get summary metrics
    metrics = dte_system.get_metrics()
    
    return {
        'num_claims': len(claims),
        'num_escalations': metrics['escalations'],
        'escalation_rate': metrics.get('escalation_rate', 0),
        'accuracy': metrics.get('overall_accuracy', 0),
        'avg_cost': metrics.get('avg_calls_per_evaluation', 0),
        'results': [r.__dict__ for r in results]
    }