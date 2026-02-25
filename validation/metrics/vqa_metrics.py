"""
VQA Metrics Computation

Computes accuracy, precision, recall, and F1 scores for VQA validation.
"""

from typing import Dict, List, Any
from collections import defaultdict


class VQAMetrics:
    """
    Metrics computer for VQA validation.

    Handles three question types:
    - Q1 (organ): Simple accuracy
    - Q2 (tumor): Binary classification metrics (precision/recall/F1)
    - Q3 (diagnosis): Simple accuracy
    """

    @staticmethod
    def compute(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute metrics from evaluation results.

        Args:
            results: List of result dictionaries from evaluator

        Returns:
            Dictionary with metrics organized by question type
        """
        # Group results by question type
        results_by_type = defaultdict(list)
        for r in results:
            q_type = r.get('metadata', {}).get('question_type', 'unknown')
            results_by_type[q_type].append(r)

        metrics = {}

        # Q1: Organ (simple accuracy)
        if 'organ' in results_by_type:
            metrics['q1_organ'] = VQAMetrics._compute_accuracy_metrics(
                results_by_type['organ']
            )

        # Q2: Tumor (binary classification)
        if 'tumor' in results_by_type:
            metrics['q2_tumor'] = VQAMetrics._compute_binary_metrics(
                results_by_type['tumor']
            )

        # Q3: Diagnosis (simple accuracy)
        if 'diagnosis' in results_by_type:
            metrics['q3_diagnosis'] = VQAMetrics._compute_accuracy_metrics(
                results_by_type['diagnosis']
            )

        return metrics

    @staticmethod
    def _compute_accuracy_metrics(results: List[Dict]) -> Dict[str, Any]:
        """
        Compute accuracy metrics.

        For organ questions: Computes hierarchical scores (0.0, 0.5, 0.75, 1.0)
        For other questions: Simple binary accuracy
        """
        total_correct = 0
        total_predictions = 0
        unknown_preds = 0

        # Track hierarchical scores for organ questions
        organ_scores = []
        score_counts = {1.0: 0, 0.75: 0, 0.5: 0, 0.0: 0}

        for r in results:
            extracted = r.get('extracted_answer', 'unknown')
            # Handle None or 'unknown' extracted answers
            if extracted is None or extracted == 'unknown':
                unknown_preds += 1
                continue

            total_predictions += 1

            # Check if this has an organ_score (hierarchical scoring)
            if 'organ_score' in r:
                score = r['organ_score']
                organ_scores.append(score)
                score_counts[score] = score_counts.get(score, 0) + 1
                if score == 1.0:
                    total_correct += 1
            else:
                # Standard binary accuracy
                if r.get('is_correct', False):
                    total_correct += 1

        accuracy = total_correct / total_predictions if total_predictions > 0 else 0

        metrics = {
            'accuracy': accuracy,
            'correct': total_correct,
            'incorrect': total_predictions - total_correct,
            'unknown_preds': unknown_preds,
            'total': total_predictions
        }

        # Add hierarchical metrics if organ scores are present
        if organ_scores:
            avg_hierarchical_score = sum(organ_scores) / len(organ_scores)
            metrics['avg_hierarchical_score'] = avg_hierarchical_score
            metrics['score_breakdown'] = score_counts

        return metrics

    @staticmethod
    def _compute_binary_metrics(results: List[Dict]) -> Dict[str, Any]:
        """
        Compute binary classification metrics (precision, recall, F1).

        Used for Q2 (tumor yes/no) where we can compute TP/TN/FP/FN.
        """
        tp = fp = tn = fn = 0
        unknown_preds = 0

        for r in results:
            extracted = r.get('extracted_answer', 'unknown')
            ground_truth = r.get('ground_truth', '').lower()

            # Handle None or 'unknown' extracted answers
            if extracted is None or extracted == 'unknown':
                unknown_preds += 1
                continue

            # Normalize to yes/no
            pred = extracted.lower()
            gt = ground_truth.lower()

            if pred == 'yes' and gt == 'yes':
                tp += 1
            elif pred == 'yes' and gt == 'no':
                fp += 1
            elif pred == 'no' and gt == 'no':
                tn += 1
            elif pred == 'no' and gt == 'yes':
                fn += 1

        # Compute metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'unknown_preds': unknown_preds,
            'total': tp + tn + fp + fn
        }

    @staticmethod
    def generate_report(metrics: Dict[str, Any], model_name: str) -> str:
        """
        Generate human-readable report from metrics.

        Args:
            metrics: Metrics dictionary from compute()
            model_name: Name of the model

        Returns:
            Formatted report string
        """
        lines = []
        lines.append(f"{'='*80}")
        lines.append(f"Model: {model_name}")
        lines.append(f"{'='*80}")

        # Q1: Organ
        if 'q1_organ' in metrics:
            q1 = metrics['q1_organ']
            lines.append(f"\nQ1 - Organ Identification:")

            # Show hierarchical scoring if available
            if 'avg_hierarchical_score' in q1:
                lines.append(f"  Avg Hierarchical Score: {q1['avg_hierarchical_score']:.3f}")
                lines.append(f"  Accuracy (Exact Match): {q1['accuracy']:.3f}")

                # Show score breakdown
                breakdown = q1.get('score_breakdown', {})
                total = q1['total']
                lines.append(f"  Score Breakdown:")
                lines.append(f"    Exact Match (1.0):     {breakdown.get(1.0, 0):3d} ({100*breakdown.get(1.0, 0)/total:.1f}%)")
                lines.append(f"    One-Hop Away (0.75):   {breakdown.get(0.75, 0):3d} ({100*breakdown.get(0.75, 0)/total:.1f}%)")
                lines.append(f"    Two-Hop Away (0.5):    {breakdown.get(0.5, 0):3d} ({100*breakdown.get(0.5, 0)/total:.1f}%)")
                lines.append(f"    No Match (0.0):        {breakdown.get(0.0, 0):3d} ({100*breakdown.get(0.0, 0)/total:.1f}%)")
                lines.append(f"  Total:      {total}")
            else:
                # Fallback to simple accuracy display
                lines.append(f"  Accuracy:   {q1['accuracy']:.3f}")
                lines.append(f"  Correct:    {q1['correct']}")
                lines.append(f"  Incorrect:  {q1['incorrect']}")
                lines.append(f"  Total:      {q1['total']}")

            if q1['unknown_preds'] > 0:
                lines.append(f"  Unknown:    {q1['unknown_preds']}")

        # Q2: Tumor
        if 'q2_tumor' in metrics:
            q2 = metrics['q2_tumor']
            lines.append(f"\nQ2 - Tumor Presence (Binary Yes/No):")
            lines.append(f"  Accuracy:   {q2['accuracy']:.3f}")
            lines.append(f"  Precision:  {q2['precision']:.3f}")
            lines.append(f"  Recall:     {q2['recall']:.3f}")
            lines.append(f"  F1-Score:   {q2['f1']:.3f}")
            lines.append(f"  TP/TN/FP/FN: {q2['tp']}/{q2['tn']}/{q2['fp']}/{q2['fn']}")
            lines.append(f"  Total:      {q2['total']}")
            if q2['unknown_preds'] > 0:
                lines.append(f"  Unknown:    {q2['unknown_preds']}")

        # Q3: Diagnosis
        if 'q3_diagnosis' in metrics:
            q3 = metrics['q3_diagnosis']
            lines.append(f"\nQ3 - Diagnosis (Multiple Choice):")
            lines.append(f"  Accuracy:   {q3['accuracy']:.3f}")
            lines.append(f"  Correct:    {q3['correct']}")
            lines.append(f"  Incorrect:  {q3['incorrect']}")
            lines.append(f"  Total:      {q3['total']}")
            if q3['unknown_preds'] > 0:
                lines.append(f"  Unknown:    {q3['unknown_preds']}")

        lines.append("")
        return "\n".join(lines)
