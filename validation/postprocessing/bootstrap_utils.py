"""
Bootstrap utilities and metric computation functions.

Provides functions to compute metrics with bootstrap confidence intervals
for organ, tumor, and diagnosis questions.
"""

import random
from typing import Dict, List
import numpy as np


def compute_organ_metrics(results: List[Dict], n_bootstrap: int = 10000) -> Dict:
    """
    Compute accuracy with bootstrap 95% CI for organ questions.

    Args:
        results: List of result dictionaries with 'is_correct' field
        n_bootstrap: Number of bootstrap iterations

    Returns:
        Dictionary with n_total, n_correct, accuracy, ci_lower, ci_upper
    """
    n = len(results)
    if n == 0:
        return {
            'n_total': 0,
            'n_correct': 0,
            'accuracy': 0.0,
            'ci_lower': 0.0,
            'ci_upper': 0.0,
            'avg_organ_score': 0.0,
            'organ_score_ci_lower': 0.0,
            'organ_score_ci_upper': 0.0
        }

    n_correct = sum(1 for r in results if r['is_correct'])
    accuracy = n_correct / n

    # Calculate average organ score
    organ_scores = [r.get('organ_score', 0.0) for r in results]
    avg_organ_score = sum(organ_scores) / n

    # Bootstrap
    bootstrap_accuracies = []
    bootstrap_organ_scores = []
    for _ in range(n_bootstrap):
        sample = random.choices(results, k=n)
        
        # Accuracy bootstrap
        bootstrap_acc = sum(1 for r in sample if r['is_correct']) / n
        bootstrap_accuracies.append(bootstrap_acc)
        
        # Organ score bootstrap
        sample_scores = [r.get('organ_score', 0.0) for r in sample]
        bootstrap_organ_scores.append(sum(sample_scores) / n)

    ci_lower = np.percentile(bootstrap_accuracies, 2.5)
    ci_upper = np.percentile(bootstrap_accuracies, 97.5)
    
    organ_score_ci_lower = np.percentile(bootstrap_organ_scores, 2.5)
    organ_score_ci_upper = np.percentile(bootstrap_organ_scores, 97.5)

    return {
        'n_total': n,
        'n_correct': n_correct,
        'accuracy': accuracy,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'avg_organ_score': avg_organ_score,
        'organ_score_ci_lower': organ_score_ci_lower,
        'organ_score_ci_upper': organ_score_ci_upper
    }


def compute_tumor_metrics(results: List[Dict], n_bootstrap: int = 10000) -> Dict:
    """
    Compute precision, recall, F1 with bootstrap 95% CI for tumor questions.

    Handles binary (yes/no) and non-binary responses separately.
    Non-binary responses are excluded from metrics but logged.

    Args:
        results: List of result dictionaries
        n_bootstrap: Number of bootstrap iterations

    Returns:
        Dictionary with metrics, confusion matrix, and non-binary cases
    """
    # Separate binary and non-binary responses
    binary_results = []
    non_binary_cases = []

    for r in results:
        extracted = r['extracted_answer'].lower().strip()
        ground_truth = r['ground_truth'].lower().strip()

        if extracted in ['yes', 'no']:
            binary_results.append(r)
        else:
            non_binary_cases.append({
                'question_id': r['question_id'],
                'ground_truth': r['ground_truth'],
                'extracted_answer': r['extracted_answer'],
                'model_response_preview': r.get('model_response', '')[:200]
            })

    if len(binary_results) == 0:
        # No binary results - return zeros
        return {
            'n_total': len(results),
            'n_binary': 0,
            'n_non_binary': len(non_binary_cases),
            'non_binary_cases': non_binary_cases,
            'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0,
            'precision': 0.0,
            'precision_ci_lower': 0.0,
            'precision_ci_upper': 0.0,
            'recall': 0.0,
            'recall_ci_lower': 0.0,
            'recall_ci_upper': 0.0,
            'f1': 0.0,
            'f1_ci_lower': 0.0,
            'f1_ci_upper': 0.0
        }

    # Compute confusion matrix
    # Ground truth: Yes = Positive, No = Negative
    # Prediction: yes = Positive, no = Negative
    tp = sum(1 for r in binary_results
             if r['ground_truth'].lower().strip() == 'yes' and
                r['extracted_answer'].lower().strip() == 'yes')
    fp = sum(1 for r in binary_results
             if r['ground_truth'].lower().strip() == 'no' and
                r['extracted_answer'].lower().strip() == 'yes')
    tn = sum(1 for r in binary_results
             if r['ground_truth'].lower().strip() == 'no' and
                r['extracted_answer'].lower().strip() == 'no')
    fn = sum(1 for r in binary_results
             if r['ground_truth'].lower().strip() == 'yes' and
                r['extracted_answer'].lower().strip() == 'no')

    # Compute metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Bootstrap for confidence intervals
    def compute_metrics_for_sample(sample):
        """Compute precision, recall, F1 for a bootstrap sample."""
        s_tp = sum(1 for r in sample
                   if r['ground_truth'].lower().strip() == 'yes' and
                      r['extracted_answer'].lower().strip() == 'yes')
        s_fp = sum(1 for r in sample
                   if r['ground_truth'].lower().strip() == 'no' and
                      r['extracted_answer'].lower().strip() == 'yes')
        s_tn = sum(1 for r in sample
                   if r['ground_truth'].lower().strip() == 'no' and
                      r['extracted_answer'].lower().strip() == 'no')
        s_fn = sum(1 for r in sample
                   if r['ground_truth'].lower().strip() == 'yes' and
                      r['extracted_answer'].lower().strip() == 'no')

        s_precision = s_tp / (s_tp + s_fp) if (s_tp + s_fp) > 0 else 0.0
        s_recall = s_tp / (s_tp + s_fn) if (s_tp + s_fn) > 0 else 0.0
        s_f1 = 2 * s_precision * s_recall / (s_precision + s_recall) if (s_precision + s_recall) > 0 else 0.0

        return s_precision, s_recall, s_f1

    bootstrap_precision = []
    bootstrap_recall = []
    bootstrap_f1 = []

    n = len(binary_results)
    for _ in range(n_bootstrap):
        sample = random.choices(binary_results, k=n)
        p, r, f = compute_metrics_for_sample(sample)
        bootstrap_precision.append(p)
        bootstrap_recall.append(r)
        bootstrap_f1.append(f)

    return {
        'n_total': len(results),
        'n_binary': len(binary_results),
        'n_non_binary': len(non_binary_cases),
        'non_binary_cases': non_binary_cases,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        'precision': precision,
        'precision_ci_lower': float(np.percentile(bootstrap_precision, 2.5)),
        'precision_ci_upper': float(np.percentile(bootstrap_precision, 97.5)),
        'recall': recall,
        'recall_ci_lower': float(np.percentile(bootstrap_recall, 2.5)),
        'recall_ci_upper': float(np.percentile(bootstrap_recall, 97.5)),
        'f1': f1,
        'f1_ci_lower': float(np.percentile(bootstrap_f1, 2.5)),
        'f1_ci_upper': float(np.percentile(bootstrap_f1, 97.5)),
        'coverage': len(binary_results) / len(results) if len(results) > 0 else 0.0
    }


def compute_diagnosis_metrics(results: List[Dict], n_bootstrap: int = 10000) -> Dict:
    """
    Compute accuracy with bootstrap 95% CI for diagnosis questions.

    Uses LLM-judged extracted_answer and is_correct fields.

    Args:
        results: List of result dictionaries with 'is_correct' field
        n_bootstrap: Number of bootstrap iterations

    Returns:
        Dictionary with n_total, n_correct, accuracy, ci_lower, ci_upper
    """
    # Same as organ metrics
    return compute_organ_metrics(results, n_bootstrap)
