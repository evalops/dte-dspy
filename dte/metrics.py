"""
Metrics calculation and analysis for DTE experiments.

This module provides utilities for calculating performance metrics,
cost analysis, and statistical comparisons for DTE systems.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
from .core import DTEResult


@dataclass
class MetricsResult:
    """Comprehensive metrics for DTE evaluation."""
    
    # Basic counts
    total_evaluations: int
    escalations: int
    agreements: int
    disagreements: int
    
    # Accuracy metrics
    correct_predictions: int
    overall_accuracy: float
    escalated_accuracy: Optional[float]
    non_escalated_accuracy: Optional[float]
    
    # Rate metrics  
    escalation_rate: float
    disagreement_rate: float
    agreement_rate: float
    
    # Cost metrics
    total_model_calls: int
    avg_calls_per_evaluation: float
    cost_per_correct: float
    
    # Advanced metrics
    false_consensus_rate: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {k: v for k, v in self.__dict__.items() if v is not None}
    
    def efficiency_score(self) -> float:
        """Calculate efficiency as accuracy per call."""
        return self.overall_accuracy / self.avg_calls_per_evaluation if self.avg_calls_per_evaluation > 0 else 0


def calculate_metrics(results: List[DTEResult], 
                     ground_truth: List[str]) -> MetricsResult:
    """
    Calculate comprehensive metrics from DTE results.
    
    Args:
        results: List of DTEResult objects
        ground_truth: List of ground truth labels ("yes"/"no")
        
    Returns:
        MetricsResult with all calculated metrics
    """
    if len(results) != len(ground_truth):
        raise ValueError("Results and ground truth must have same length")
    
    total_evaluations = len(results)
    if total_evaluations == 0:
        raise ValueError("Cannot calculate metrics for empty results")
    
    # Convert ground truth to binary
    gt_binary = [1 if gt == "yes" else 0 for gt in ground_truth]
    
    # Basic counts
    escalations = sum(1 for r in results if r.escalated)
    agreements = sum(1 for r in results if r.verifier_a_result.prediction == r.verifier_b_result.prediction)
    disagreements = total_evaluations - agreements
    
    # Accuracy calculations
    correct_predictions = sum(1 for r, gt in zip(results, gt_binary) 
                            if r.final_prediction == gt)
    overall_accuracy = correct_predictions / total_evaluations
    
    # Escalated vs non-escalated accuracy
    escalated_results = [(r, gt) for r, gt in zip(results, gt_binary) if r.escalated]
    non_escalated_results = [(r, gt) for r, gt in zip(results, gt_binary) if not r.escalated]
    
    escalated_accuracy = None
    if escalated_results:
        escalated_correct = sum(1 for r, gt in escalated_results if r.final_prediction == gt)
        escalated_accuracy = escalated_correct / len(escalated_results)
    
    non_escalated_accuracy = None
    if non_escalated_results:
        non_escalated_correct = sum(1 for r, gt in non_escalated_results if r.final_prediction == gt)
        non_escalated_accuracy = non_escalated_correct / len(non_escalated_results)
    
    # Rate calculations
    escalation_rate = escalations / total_evaluations
    disagreement_rate = disagreements / total_evaluations
    agreement_rate = agreements / total_evaluations
    
    # Cost calculations
    total_model_calls = sum(3 if r.escalated else 2 for r in results)
    avg_calls_per_evaluation = total_model_calls / total_evaluations
    cost_per_correct = total_model_calls / correct_predictions if correct_predictions > 0 else float('inf')
    
    # False consensus rate (agreed but wrong)
    false_consensus_rate = None
    consensus_results = [r for r in results if not r.escalated]
    if consensus_results:
        false_consensus = sum(1 for r, gt in zip(consensus_results, 
                                               [gt_binary[i] for i, orig_r in enumerate(results) if not orig_r.escalated])
                            if r.final_prediction != gt)
        false_consensus_rate = false_consensus / len(consensus_results)
    
    # Precision, recall, F1 for positive class
    predictions = [r.final_prediction for r in results]
    
    true_positives = sum(1 for pred, gt in zip(predictions, gt_binary) if pred == 1 and gt == 1)
    false_positives = sum(1 for pred, gt in zip(predictions, gt_binary) if pred == 1 and gt == 0)
    false_negatives = sum(1 for pred, gt in zip(predictions, gt_binary) if pred == 0 and gt == 1)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else None
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else None
    
    f1_score = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    
    return MetricsResult(
        total_evaluations=total_evaluations,
        escalations=escalations,
        agreements=agreements,
        disagreements=disagreements,
        correct_predictions=correct_predictions,
        overall_accuracy=overall_accuracy,
        escalated_accuracy=escalated_accuracy,
        non_escalated_accuracy=non_escalated_accuracy,
        escalation_rate=escalation_rate,
        disagreement_rate=disagreement_rate,
        agreement_rate=agreement_rate,
        total_model_calls=total_model_calls,
        avg_calls_per_evaluation=avg_calls_per_evaluation,
        cost_per_correct=cost_per_correct,
        false_consensus_rate=false_consensus_rate,
        precision=precision,
        recall=recall,
        f1_score=f1_score
    )


def compare_configurations(metrics_list: List[MetricsResult],
                         config_names: List[str]) -> Dict[str, Any]:
    """Compare multiple DTE configurations."""
    if len(metrics_list) != len(config_names):
        raise ValueError("Metrics and config names must have same length")
    
    comparison: Dict[str, Any] = {
        'configurations': config_names,
        'accuracy_comparison': [m.overall_accuracy for m in metrics_list],
        'cost_comparison': [m.avg_calls_per_evaluation for m in metrics_list],
        'efficiency_comparison': [m.efficiency_score() for m in metrics_list],
        'escalation_comparison': [m.escalation_rate for m in metrics_list]
    }
    
    # Find best in each category
    best_accuracy_idx = np.argmax(comparison['accuracy_comparison'])
    best_efficiency_idx = np.argmax(comparison['efficiency_comparison'])
    lowest_cost_idx = np.argmin(comparison['cost_comparison'])
    
    comparison['best_configs'] = {
        'accuracy': config_names[best_accuracy_idx],
        'efficiency': config_names[best_efficiency_idx], 
        'cost': config_names[lowest_cost_idx]
    }
    
    return comparison


def calculate_statistical_significance(results_a: List[DTEResult],
                                     results_b: List[DTEResult],
                                     ground_truth: List[str]) -> Dict[str, Any]:
    """Calculate statistical significance between two configurations."""
    from scipy import stats
    
    # Get accuracy for each configuration
    gt_binary = [1 if gt == "yes" else 0 for gt in ground_truth]
    
    accuracy_a = [1 if r.final_prediction == gt else 0 
                  for r, gt in zip(results_a, gt_binary)]
    accuracy_b = [1 if r.final_prediction == gt else 0
                  for r, gt in zip(results_b, gt_binary)]
    
    # Paired t-test for accuracy differences
    t_stat, p_value = stats.ttest_rel(accuracy_a, accuracy_b)
    
    # Effect size (Cohen's d)
    diff = np.array(accuracy_a) - np.array(accuracy_b)
    cohens_d = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'significant': p_value < 0.05,
        'mean_accuracy_a': np.mean(accuracy_a),
        'mean_accuracy_b': np.mean(accuracy_b),
        'accuracy_difference': np.mean(accuracy_a) - np.mean(accuracy_b)
    }


def generate_summary_stats(metrics_list: List[MetricsResult]) -> Dict[str, Any]:
    """Generate summary statistics across multiple evaluations."""
    if not metrics_list:
        return {}
    
    accuracy_values = [m.overall_accuracy for m in metrics_list]
    cost_values = [m.avg_calls_per_evaluation for m in metrics_list]
    escalation_values = [m.escalation_rate for m in metrics_list]
    
    return {
        'accuracy_stats': {
            'mean': np.mean(accuracy_values),
            'std': np.std(accuracy_values),
            'min': np.min(accuracy_values),
            'max': np.max(accuracy_values),
            'median': np.median(accuracy_values)
        },
        'cost_stats': {
            'mean': np.mean(cost_values),
            'std': np.std(cost_values),
            'min': np.min(cost_values),
            'max': np.max(cost_values),
            'median': np.median(cost_values)
        },
        'escalation_stats': {
            'mean': np.mean(escalation_values),
            'std': np.std(escalation_values),
            'min': np.min(escalation_values),
            'max': np.max(escalation_values),
            'median': np.median(escalation_values)
        },
        'num_evaluations': len(metrics_list)
    }