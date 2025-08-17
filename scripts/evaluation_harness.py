"""
Evaluation Harness for Disagreement-Triggered Escalation (DTE)

This module provides comprehensive evaluation capabilities including:
- Gamma threshold sweeping
- Accuracy vs cost analysis  
- False consensus rate measurement
- Dataset loading and processing
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
from dataclasses import asdict
import logging

from scripts.dte_core import DTESystem, DTEResult, VerificationResult

logger = logging.getLogger(__name__)


class DTEEvaluator:
    """Comprehensive evaluator for DTE systems."""
    
    def __init__(self, dte_system: DTESystem):
        self.dte_system = dte_system
        self.evaluation_results = []
        
    def load_dataset(self, 
                    claims: List[str], 
                    ground_truth: List[int],
                    dataset_name: str = "custom") -> pd.DataFrame:
        """
        Load a dataset of claims with ground truth labels.
        
        Args:
            claims: List of factual claims to evaluate
            ground_truth: List of ground truth labels (0 or 1)
            dataset_name: Name identifier for the dataset
            
        Returns:
            DataFrame with claims and labels
        """
        assert len(claims) == len(ground_truth), "Claims and labels must have same length"
        
        return pd.DataFrame({
            'claim': claims,
            'ground_truth': ground_truth,
            'dataset': dataset_name
        })
    
    def create_synthetic_dataset(self, size: int = 100) -> pd.DataFrame:
        """Create a synthetic dataset for testing."""
        claims = []
        labels = []
        
        # True claims
        true_claims = [
            "The capital of France is Paris.",
            "Water boils at 100 degrees Celsius at sea level.",
            "Shakespeare wrote Romeo and Juliet.",
            "The Earth orbits around the Sun.",
            "Python is a programming language.",
            "The Pacific Ocean is the largest ocean.",
            "DNA stands for Deoxyribonucleic acid.",
            "The Great Wall of China is in China.",
            "Gravity causes objects to fall.",
            "The human heart has four chambers."
        ]
        
        # False claims  
        false_claims = [
            "The capital of France is London.",
            "Water boils at 50 degrees Celsius at sea level.",
            "Shakespeare wrote Harry Potter.",
            "The Sun orbits around the Earth.",
            "Python is a type of bird only.",
            "The Atlantic Ocean is the largest ocean.",
            "DNA stands for Digital Network Access.",
            "The Great Wall of China is in Japan.",
            "Gravity causes objects to float.",
            "The human heart has two chambers."
        ]
        
        for i in range(size):
            if i % 2 == 0:
                claim = np.random.choice(true_claims)
                label = 1
            else:
                claim = np.random.choice(false_claims) 
                label = 0
            claims.append(claim)
            labels.append(label)
            
        return self.load_dataset(claims, labels, "synthetic")
    
    def gamma_sweep_evaluation(self, 
                             dataset: pd.DataFrame,
                             gamma_values: Optional[List[float]] = None,
                             verbose: bool = True) -> pd.DataFrame:
        """
        Evaluate DTE system across different gamma thresholds.
        
        Args:
            dataset: DataFrame with 'claim' and 'ground_truth' columns
            gamma_values: List of gamma thresholds to test
            verbose: Whether to show progress bars
            
        Returns:
            DataFrame with results for each gamma value
        """
        if gamma_values is None:
            gamma_values = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
            
        results = []
        
        for gamma in tqdm(gamma_values, desc="Gamma sweep", disable=not verbose):
            gamma_results = self._evaluate_single_gamma(dataset, gamma, verbose)
            gamma_results['gamma'] = gamma
            results.append(gamma_results)
            
        return pd.DataFrame(results)
    
    def _evaluate_single_gamma(self, 
                              dataset: pd.DataFrame, 
                              gamma: float,
                              verbose: bool = False) -> Dict[str, Any]:
        """Evaluate DTE system for a single gamma value."""
        self.dte_system.reset_metrics()
        predictions = []
        escalated_predictions = []
        non_escalated_predictions = []
        escalation_flags: list[bool] = []
        
        iterator = tqdm(dataset.iterrows(), total=len(dataset), 
                       desc=f"Evaluating γ={gamma:.2f}", 
                       disable=not verbose, leave=False)
        
        for _, row in iterator:
            result = self.dte_system.evaluate_claim(str(row['claim']), gamma)
            predictions.append(result.final_prediction)
            escalation_flags.append(result.escalated)
            
            if result.escalated:
                escalated_predictions.append(result.final_prediction)
            else:
                non_escalated_predictions.append(result.final_prediction)
        
        # Calculate metrics
        ground_truth = dataset['ground_truth'].values
        accuracy = np.mean(predictions == ground_truth)
        
        # Escalation-specific metrics
        escalated_mask = np.array(escalation_flags, dtype=bool)
        escalation_rate = sum(escalation_flags) / len(escalation_flags)
        
        escalated_accuracy = None
        non_escalated_accuracy = None
        
        if np.any(escalated_mask):
            escalated_gt = ground_truth[escalated_mask]
            escalated_preds = np.array(predictions)[escalated_mask]
            escalated_accuracy = np.mean(escalated_preds == escalated_gt)
            
        if np.any(~escalated_mask):
            non_escalated_gt = ground_truth[~escalated_mask]
            non_escalated_preds = np.array(predictions)[~escalated_mask]
            non_escalated_accuracy = np.mean(non_escalated_preds == non_escalated_gt)
        
        # False consensus rate: cases where verifiers agree but are wrong
        false_consensus_count = 0
        total_consensus_count = 0
        
        for i, row in dataset.iterrows():
            result = self.dte_system.evaluate_claim(str(row['claim']), gamma)
            verifier_agreement = (result.verifier_a_result.prediction == 
                                result.verifier_b_result.prediction)
            
            if verifier_agreement and not result.escalated:
                total_consensus_count += 1
                if result.final_prediction != row['ground_truth']:
                    false_consensus_count += 1
        
        false_consensus_rate = (false_consensus_count / total_consensus_count 
                              if total_consensus_count > 0 else 0)
        
        # Get system metrics
        sys_metrics = self.dte_system.get_metrics()
        
        return {
            'accuracy': accuracy,
            'escalation_rate': escalation_rate,
            'escalated_accuracy': escalated_accuracy,
            'non_escalated_accuracy': non_escalated_accuracy,
            'false_consensus_rate': false_consensus_rate,
            'disagreement_rate': sys_metrics.get('disagreement_rate', 0),
            'avg_calls_per_evaluation': sys_metrics.get('avg_calls_per_evaluation', 0),
            'total_evaluations': len(dataset),
            'total_escalations': sys_metrics.get('escalations', 0),
            'total_agreements': sys_metrics.get('agreements', 0),
            'total_disagreements': sys_metrics.get('disagreements', 0)
        }
    
    def plot_gamma_analysis(self, results_df: pd.DataFrame, save_path: Optional[str] = None):
        """Create comprehensive plots of gamma sweep results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('DTE System: Gamma Threshold Analysis', fontsize=16)
        
        # Accuracy vs Escalation Rate
        axes[0, 0].plot(results_df['escalation_rate'], results_df['accuracy'], 'bo-')
        axes[0, 0].set_xlabel('Escalation Rate')
        axes[0, 0].set_ylabel('Overall Accuracy')
        axes[0, 0].set_title('Accuracy vs Escalation Rate')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add gamma labels
        for i, row in results_df.iterrows():
            axes[0, 0].annotate(f'γ={row["gamma"]:.2f}', 
                              (row['escalation_rate'], row['accuracy']),
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Cost vs Accuracy
        axes[0, 1].plot(results_df['avg_calls_per_evaluation'], results_df['accuracy'], 'ro-')
        axes[0, 1].set_xlabel('Average Calls per Evaluation')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Cost vs Accuracy Trade-off')
        axes[0, 1].grid(True, alpha=0.3)
        
        # False Consensus Rate
        axes[1, 0].plot(results_df['gamma'], results_df['false_consensus_rate'], 'go-')
        axes[1, 0].set_xlabel('Gamma Threshold')
        axes[1, 0].set_ylabel('False Consensus Rate')
        axes[1, 0].set_title('False Consensus Rate vs Gamma')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Escalation Components
        axes[1, 1].plot(results_df['gamma'], results_df['escalation_rate'], 'mo-', label='Total Escalation')
        axes[1, 1].plot(results_df['gamma'], results_df['disagreement_rate'], 'co-', label='Disagreement')
        axes[1, 1].set_xlabel('Gamma Threshold')
        axes[1, 1].set_ylabel('Rate')
        axes[1, 1].set_title('Escalation Components')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def save_results(self, results_df: pd.DataFrame, filepath: str):
        """Save evaluation results to file."""
        if filepath.endswith('.csv'):
            results_df.to_csv(filepath, index=False)
        elif filepath.endswith('.json'):
            results_df.to_json(filepath, orient='records', indent=2)
        else:
            raise ValueError("Filepath must end with .csv or .json")
        
        logger.info(f"Results saved to {filepath}")
    
    def generate_report(self, results_df: pd.DataFrame) -> str:
        """Generate a comprehensive text report."""
        report = []
        report.append("=== DTE System Evaluation Report ===\n")
        
        # Best performing configurations
        best_accuracy = results_df.loc[results_df['accuracy'].idxmax()]
        lowest_cost = results_df.loc[results_df['avg_calls_per_evaluation'].idxmin()]
        best_fcr = results_df.loc[results_df['false_consensus_rate'].idxmin()]
        
        report.append("Best Configurations:")
        report.append(f"  Highest Accuracy: γ={best_accuracy['gamma']:.2f} → {best_accuracy['accuracy']:.3f} accuracy")
        report.append(f"  Lowest Cost: γ={lowest_cost['gamma']:.2f} → {lowest_cost['avg_calls_per_evaluation']:.2f} calls/eval")
        report.append(f"  Lowest FCR: γ={best_fcr['gamma']:.2f} → {best_fcr['false_consensus_rate']:.3f} FCR\n")
        
        # Summary statistics
        report.append("Summary Statistics:")
        report.append(f"  Accuracy range: {results_df['accuracy'].min():.3f} - {results_df['accuracy'].max():.3f}")
        report.append(f"  Escalation rate range: {results_df['escalation_rate'].min():.3f} - {results_df['escalation_rate'].max():.3f}")
        report.append(f"  Cost range: {results_df['avg_calls_per_evaluation'].min():.2f} - {results_df['avg_calls_per_evaluation'].max():.2f} calls/eval")
        report.append(f"  FCR range: {results_df['false_consensus_rate'].min():.3f} - {results_df['false_consensus_rate'].max():.3f}")
        
        # Recommendations
        report.append("\nRecommendations:")
        
        # Find good balance point (high accuracy, reasonable cost)
        results_df['efficiency'] = results_df['accuracy'] / results_df['avg_calls_per_evaluation']
        best_efficiency = results_df.loc[results_df['efficiency'].idxmax()]
        
        report.append(f"  Recommended γ for efficiency: {best_efficiency['gamma']:.2f}")
        report.append(f"    - Accuracy: {best_efficiency['accuracy']:.3f}")
        report.append(f"    - Cost: {best_efficiency['avg_calls_per_evaluation']:.2f} calls/eval")
        report.append(f"    - Escalation rate: {best_efficiency['escalation_rate']:.3f}")
        
        return "\n".join(report)


def create_example_dataset() -> pd.DataFrame:
    """Create an example dataset for demonstration."""
    claims = [
        "The capital of France is Paris.",
        "The capital of France is London.", 
        "Water boils at 100°C at sea level.",
        "Water boils at 50°C at sea level.",
        "Shakespeare wrote Romeo and Juliet.",
        "Shakespeare wrote Harry Potter.",
        "The Earth is round.",
        "The Earth is flat.",
        "Python is a programming language.",
        "Python is only a type of snake."
    ]
    
    ground_truth = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    
    evaluator = DTEEvaluator(None)  # type: ignore # Will be replaced in actual usage
    return evaluator.load_dataset(claims, ground_truth, "example")


if __name__ == "__main__":
    # Example usage
    from scripts.dte_core import create_simple_test_models
    
    # Create DTE system
    model_a, model_b, referee = create_simple_test_models()
    dte_system = DTESystem(model_a, model_b, referee)
    
    # Create evaluator
    evaluator = DTEEvaluator(dte_system)
    
    # Create test dataset
    dataset = evaluator.create_synthetic_dataset(50)
    
    # Run gamma sweep
    results = evaluator.gamma_sweep_evaluation(dataset)
    
    # Generate report
    report = evaluator.generate_report(results)
    print(report)
    
    # Save results
    evaluator.save_results(results, "dte_evaluation_results.csv")
    
    # Create plots
    evaluator.plot_gamma_analysis(results, "dte_analysis.png")