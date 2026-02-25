#!/usr/bin/env python3
"""
Compute comprehensive validation metrics with bootstrap confidence intervals.

This script computes metrics for organ, tumor, and diagnosis questions:
- Organ: Accuracy with 95% CI
- Tumor: Precision, Recall, F1 with 95% CI (excludes non-binary responses)
- Diagnosis: Accuracy with 95% CI (using LLM-judged extractions)

Usage:
    python compute_metrics.py
    python compute_metrics.py --input results_llm_parsed.json --n-bootstrap 10000
"""

import argparse
import json
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import numpy as np

from bootstrap_utils import (
    compute_organ_metrics,
    compute_tumor_metrics,
    compute_diagnosis_metrics
)

# Configuration
DEFAULT_INPUT = "results_llm_parsed.json"
DEFAULT_OUTPUT_JSON = "metrics_output.json"
DEFAULT_OUTPUT_TXT = "metrics_report.txt"
DEFAULT_NON_BINARY_FILE = "non_binary_tumor_cases.json"
RANDOM_SEED = 42
N_BOOTSTRAP = 10000


def load_results(file_path: Path) -> List[Dict]:
    """Load and return results from JSON file."""
    print(f"Loading results from {file_path}...")
    with open(file_path, 'r') as f:
        results = json.load(f)
    print(f"✓ Loaded {len(results)} entries")
    return results


def validate_data(results: List[Dict]) -> Dict:
    """
    Validate data structure and return summary.

    Returns:
        Dictionary with validation summary and any warnings
    """
    print("\nValidating data...")

    required_fields = ['model_name', 'question_id', 'ground_truth',
                      'extracted_answer', 'is_correct', 'metadata']

    warnings = []
    models = set()
    question_types = defaultdict(int)
    seen_ids = set()

    for i, entry in enumerate(results):
        # Check required fields
        missing = [f for f in required_fields if f not in entry]
        if missing:
            warnings.append(f"Entry {i} (ID: {entry.get('question_id', 'unknown')}) "
                          f"missing fields: {missing}")
            continue

        # Track models and question types
        models.add(entry['model_name'])
        qtype = entry.get('metadata', {}).get('question_type', 'unknown')
        question_types[qtype] += 1

        # Check for duplicates
        entry_key = (entry['model_name'], entry['question_id'])
        if entry_key in seen_ids:
            warnings.append(f"Duplicate entry: {entry_key}")
        seen_ids.add(entry_key)

    summary = {
        'total_entries': len(results),
        'n_models': len(models),
        'models': sorted(models),
        'question_types': dict(question_types),
        'n_warnings': len(warnings),
        'warnings': warnings[:10]  # Limit to first 10
    }

    print(f"✓ {summary['total_entries']} entries")
    print(f"✓ {summary['n_models']} models: {', '.join(summary['models'])}")
    print(f"✓ Question types: {dict(question_types)}")

    if warnings:
        print(f"⚠️  {len(warnings)} warnings (showing first 10):")
        for w in warnings[:10]:
            print(f"  - {w}")

    return summary


def group_by_model_and_type(results: List[Dict]) -> Dict:
    """
    Group results by model and question type.

    Returns:
        {
            'model_name': {
                'organ': [...],
                'tumor': [...],
                'diagnosis': [...]
            },
            ...
        }
    """
    grouped = defaultdict(lambda: {'organ': [], 'tumor': [], 'diagnosis': []})

    for entry in results:
        model = entry['model_name']
        qtype = entry.get('metadata', {}).get('question_type', 'unknown')

        if qtype in ['organ', 'tumor', 'diagnosis']:
            grouped[model][qtype].append(entry)

    return dict(grouped)


def compute_all_metrics(grouped_data: Dict, n_bootstrap: int) -> Dict:
    """
    Compute metrics for all models and question types.

    Args:
        grouped_data: Grouped results by model and question type
        n_bootstrap: Number of bootstrap iterations

    Returns:
        Dictionary of metrics by model and question type
    """
    print("\n" + "=" * 80)
    print("COMPUTING METRICS")
    print("=" * 80)

    metrics = {}

    for model in sorted(grouped_data.keys()):
        print(f"\n{model}:")
        data_by_type = grouped_data[model]

        metrics[model] = {}

        # Organ questions
        print("  - Organ questions...")
        metrics[model]['organ'] = compute_organ_metrics(
            data_by_type['organ'],
            n_bootstrap
        )

        # Tumor questions
        print("  - Tumor questions...")
        metrics[model]['tumor'] = compute_tumor_metrics(
            data_by_type['tumor'],
            n_bootstrap
        )

        # Diagnosis questions
        print("  - Diagnosis questions...")
        metrics[model]['diagnosis'] = compute_diagnosis_metrics(
            data_by_type['diagnosis'],
            n_bootstrap
        )

    return metrics


def save_json_output(metrics: Dict, output_path: Path, metadata: Dict):
    """Save structured JSON output."""
    output = {
        'metadata': metadata,
        'per_model_metrics': metrics
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"✓ Saved JSON output to {output_path}")


def generate_text_report(metrics: Dict, output_path: Path, metadata: Dict):
    """Generate human-readable text report."""
    lines = []

    # Header
    lines.append("=" * 80)
    lines.append("VALIDATION METRICS REPORT")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Generated: {metadata['timestamp']}")
    lines.append(f"Input file: {metadata['input_file']}")
    lines.append(f"Bootstrap iterations: {metadata['n_bootstrap_iterations']:,}")
    lines.append(f"Random seed: {metadata['random_seed']}")
    lines.append("")

    # Per-model sections
    for model in sorted(metrics.keys()):
        lines.append("=" * 80)
        lines.append(model)
        lines.append("─" * 80)
        lines.append("")

        # Organ questions
        organ = metrics[model]['organ']
        lines.append("ORGAN QUESTIONS (Accuracy & Score)")
        lines.append(f"  Total questions:    {organ['n_total']}")
        lines.append(f"  Correct:            {organ['n_correct']}")
        lines.append(f"  Accuracy:           {organ['accuracy']*100:.2f}% "
                    f"[95% CI: {organ['ci_lower']*100:.2f}% - {organ['ci_upper']*100:.2f}%]")
        lines.append(f"  Avg Organ Score:    {organ['avg_organ_score']:.4f} "
                    f"[95% CI: {organ['organ_score_ci_lower']:.4f} - {organ['organ_score_ci_upper']:.4f}]")
        lines.append("")

        # Tumor questions
        tumor = metrics[model]['tumor']
        lines.append("TUMOR QUESTIONS (Precision, Recall, F1)")
        lines.append(f"  Total questions:    {tumor['n_total']}")
        lines.append(f"  Binary answers:     {tumor['n_binary']}  "
                    f"({100*tumor['n_binary']/tumor['n_total']:.2f}%)")
        lines.append(f"  Non-binary:         {tumor['n_non_binary']}  "
                    f"({100*tumor['n_non_binary']/tumor['n_total']:.2f}%)")
        lines.append(f"  Coverage:           {tumor['coverage']*100:.2f}%")
        lines.append("")
        lines.append("  Confusion Matrix:")
        lines.append(f"    TP: {tumor['tp']:<4} |  FP: {tumor['fp']}")
        lines.append(f"    FN: {tumor['fn']:<4} |  TN: {tumor['tn']}")
        lines.append("")
        lines.append(f"  Precision:          {tumor['precision']*100:.2f}% "
                    f"[95% CI: {tumor['precision_ci_lower']*100:.2f}% - "
                    f"{tumor['precision_ci_upper']*100:.2f}%]")
        lines.append(f"  Recall:             {tumor['recall']*100:.2f}% "
                    f"[95% CI: {tumor['recall_ci_lower']*100:.2f}% - "
                    f"{tumor['recall_ci_upper']*100:.2f}%]")
        lines.append(f"  F1 Score:           {tumor['f1']*100:.2f}% "
                    f"[95% CI: {tumor['f1_ci_lower']*100:.2f}% - "
                    f"{tumor['f1_ci_upper']*100:.2f}%]")
        lines.append("")

        # Diagnosis questions
        diag = metrics[model]['diagnosis']
        lines.append("DIAGNOSIS QUESTIONS (Accuracy with LLM Judge)")
        lines.append(f"  Total questions:    {diag['n_total']}")
        lines.append(f"  Correct:            {diag['n_correct']}")
        lines.append(f"  Accuracy:           {diag['accuracy']*100:.2f}% "
                    f"[95% CI: {diag['ci_lower']*100:.2f}% - {diag['ci_upper']*100:.2f}%]")
        lines.append("")

    # Non-binary tumor cases summary
    lines.append("=" * 80)
    lines.append("NON-BINARY TUMOR RESPONSES SUMMARY")
    lines.append("=" * 80)
    lines.append("")

    total_non_binary = sum(metrics[m]['tumor']['n_non_binary'] for m in metrics)
    lines.append(f"Total non-binary responses: {total_non_binary} across all models")
    lines.append("")

    for model in sorted(metrics.keys()):
        n_non_binary = metrics[model]['tumor']['n_non_binary']
        if n_non_binary > 0:
            lines.append(f"{model}: {n_non_binary} non-binary responses")

    if total_non_binary > 0:
        lines.append("")
        lines.append(f"See {DEFAULT_NON_BINARY_FILE} for detailed list")

    # Overall statistics
    lines.append("")
    lines.append("=" * 80)
    lines.append("OVERALL STATISTICS")
    lines.append("=" * 80)
    lines.append("")

    total_organ = sum(metrics[m]['organ']['n_total'] for m in metrics)
    total_tumor = sum(metrics[m]['tumor']['n_total'] for m in metrics)
    total_diagnosis = sum(metrics[m]['diagnosis']['n_total'] for m in metrics)
    total_questions = total_organ + total_tumor + total_diagnosis

    lines.append(f"Total questions processed: {total_questions:,}")
    lines.append(f"  - Organ:     {total_organ:,}")
    lines.append(f"  - Tumor:     {total_tumor:,} ({total_tumor - total_non_binary:,} binary, {total_non_binary} non-binary)")
    lines.append(f"  - Diagnosis: {total_diagnosis:,}")
    lines.append("")
    lines.append(f"Models evaluated: {len(metrics)}")
    lines.append("")

    # Find best performing model (by average accuracy across question types)
    avg_accuracies = {}
    for model in metrics:
        organ_acc = metrics[model]['organ']['accuracy']
        tumor_f1 = metrics[model]['tumor']['f1']  # Use F1 for tumor
        diag_acc = metrics[model]['diagnosis']['accuracy']
        avg_acc = (organ_acc + tumor_f1 + diag_acc) / 3
        avg_accuracies[model] = avg_acc

    best_model = max(avg_accuracies, key=avg_accuracies.get)
    lines.append(f"Best performing model (by average score):")
    lines.append(f"  {best_model}: {avg_accuracies[best_model]*100:.2f}%")

    lines.append("")
    lines.append("=" * 80)

    # Write report
    report_text = "\n".join(lines)
    with open(output_path, 'w') as f:
        f.write(report_text)

    print(f"✓ Saved text report to {output_path}")

    # Also print to console
    print("\n" + report_text)


def save_non_binary_cases(metrics: Dict, output_path: Path):
    """Save all non-binary tumor cases to JSON file."""
    all_non_binary = []

    for model, model_metrics in metrics.items():
        for case in model_metrics['tumor']['non_binary_cases']:
            all_non_binary.append({
                'model_name': model,
                **case
            })

    if all_non_binary:
        with open(output_path, 'w') as f:
            json.dump(all_non_binary, f, indent=2)
        print(f"✓ Saved {len(all_non_binary)} non-binary tumor cases to {output_path}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Compute validation metrics with bootstrap confidence intervals"
    )
    parser.add_argument(
        '--input',
        type=str,
        default=DEFAULT_INPUT,
        help=f'Input results file (default: {DEFAULT_INPUT})'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=DEFAULT_OUTPUT_JSON,
        help=f'Output JSON file (default: {DEFAULT_OUTPUT_JSON})'
    )
    parser.add_argument(
        '--report',
        type=str,
        default=DEFAULT_OUTPUT_TXT,
        help=f'Output text report file (default: {DEFAULT_OUTPUT_TXT})'
    )
    parser.add_argument(
        '--n-bootstrap',
        type=int,
        default=N_BOOTSTRAP,
        help=f'Number of bootstrap iterations (default: {N_BOOTSTRAP})'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=RANDOM_SEED,
        help=f'Random seed for reproducibility (default: {RANDOM_SEED})'
    )

    args = parser.parse_args()

    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 80)
    print("VALIDATION METRICS COMPUTATION")
    print("=" * 80)

    # Load and validate data
    results = load_results(Path(args.input))
    validation_summary = validate_data(results)

    # Group data by model and question type
    print("\nGrouping data by model and question type...")
    grouped_data = group_by_model_and_type(results)
    print(f"✓ Grouped data for {len(grouped_data)} models")

    # Compute metrics
    metrics = compute_all_metrics(grouped_data, args.n_bootstrap)

    # Prepare metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'input_file': args.input,
        'n_bootstrap_iterations': args.n_bootstrap,
        'random_seed': args.seed,
        'validation_summary': validation_summary
    }

    # Save outputs
    print("\n" + "=" * 80)
    print("SAVING OUTPUTS")
    print("=" * 80)

    save_json_output(metrics, Path(args.output), metadata)
    generate_text_report(metrics, Path(args.report), metadata)
    save_non_binary_cases(metrics, Path(DEFAULT_NON_BINARY_FILE))

    print("\n" + "=" * 80)
    print("✓ COMPLETE!")
    print("=" * 80)
    print(f"\nOutputs:")
    print(f"  - JSON metrics: {args.output}")
    print(f"  - Text report:  {args.report}")
    print(f"  - Non-binary cases: {DEFAULT_NON_BINARY_FILE}")


if __name__ == '__main__':
    main()
