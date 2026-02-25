#!/usr/bin/env python3
"""
Unified Benchmark Runner

Single entry point for running VQA validation benchmark evaluations.
Supports both single-GPU and DDP (multi-GPU) execution.
"""

import argparse
import torch
import os
import json
from pathlib import Path
from dotenv import load_dotenv

from validation.base_models import create_model
from validation.datasets import VQADataset
from validation.evaluator import Evaluator
from validation.evaluator_ddp import run_ddp_evaluation
from validation.parsers import VQAParser
from validation.metrics import VQAMetrics
from validation.common import load_config


def run_single_gpu(
    benchmark: str,
    config_path: str,
    max_questions: int = None,
    test_slides: int = None,
    models_to_run: list = None,
):
    """Run evaluation on single GPU."""
    print(f"\n=== Single GPU {benchmark.upper()} Evaluation ===")
    if max_questions:
        print(f"TEST MODE: Limited to {max_questions} questions")

    # Load config
    config = load_config(config_path)

    # Create dataset for VQA benchmark
    print("\nLoading VQA dataset...")
    dataset = VQADataset(config)
    parser_fn = VQAParser.parse
    metrics_cls = VQAMetrics
    output_dir = Path(config["validation"]["output_dir"])
    eval_method = config.get("pipeline", {}).get("evaluation_method", "generation")
    checkpoint_interval = 10

    # Get models
    all_models = []
    for model_type in ["antoni_alpha", "medgemma"]:
        if model_type in config["models"]:
            for model_cfg in config["models"][model_type]:
                model_cfg["type"] = model_type
                all_models.append(model_cfg)

    # Filter models if specified
    if models_to_run:
        all_models = [m for m in all_models if m["name"] in models_to_run]

    # Print dataset statistics
    stats = dataset.get_statistics()
    print(f"\nDataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print(f"\nEvaluation method: {eval_method}")
    print(f"Evaluating {len(all_models)} models")

    # Create evaluator
    evaluator = Evaluator(
        output_dir=output_dir,
        checkpoint_interval=checkpoint_interval,
        evaluation_method=eval_method,
    )

    # Load checkpoint if exists
    evaluator.load_checkpoint()

    # Get device
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Evaluate each model
    for model_idx, model_cfg in enumerate(all_models):
        model_name = model_cfg["name"]
        print(f"\n{'=' * 80}")
        print(f"Model {model_idx + 1}/{len(all_models)}: {model_name}")
        print(f"{'=' * 80}")

        try:
            # Create model
            model = create_model(model_cfg, device=device)

            # Run evaluation
            evaluator.evaluate_model(
                model=model,
                dataset=dataset,
                model_name=model_name,
                max_questions=max_questions,
                parser_fn=parser_fn,
            )

            # Cleanup
            model.cleanup()
            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Save results
    evaluator.save_results()

    # Compute and display metrics
    print("\n" + "=" * 80)
    print("COMPUTING METRICS")
    print("=" * 80)

    results_by_model = evaluator.get_results_by_model()
    all_metrics = {}

    for model_name, results in results_by_model.items():
        metrics = metrics_cls.compute(results)
        all_metrics[model_name] = metrics

        # Generate and print report
        report = metrics_cls.generate_report(metrics, model_name)
        print(report)

    # Save metrics
    metrics_file = output_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMetrics saved to: {metrics_file}")

    # Cleanup dataset
    if hasattr(dataset, "clear_feature_cache"):
        dataset.clear_feature_cache()
    if hasattr(dataset, "cleanup"):
        dataset.cleanup()

    print(f"\n✅ {benchmark.upper()} Evaluation complete!")


def run_ddp(
    benchmark: str,
    config_path: str,
    max_questions: int = None,
    test_slides: int = None,
    models_to_run: list = None,
):
    """Run evaluation with DDP (multi-GPU)."""
    print(f"\n=== DDP {benchmark.upper()} Evaluation ===")
    if max_questions:
        print(f"TEST MODE: Limited to {max_questions} questions")

    # Load config
    config = load_config(config_path)

    # Create dataset for VQA benchmark
    print("\nLoading VQA dataset...")
    dataset = VQADataset(config)
    parser_fn = VQAParser.parse
    metrics_cls = VQAMetrics
    output_dir = Path(config["validation"]["output_dir"])
    eval_method = config.get("pipeline", {}).get("evaluation_method", "generation")

    # Get models
    all_models = []
    for model_type in ["antoni_alpha", "medgemma"]:
        if model_type in config["models"]:
            for model_cfg in config["models"][model_type]:
                model_cfg["type"] = model_type
                all_models.append(model_cfg)

    # Filter models if specified
    if models_to_run:
        all_models = [m for m in all_models if m["name"] in models_to_run]

    # Print dataset statistics
    stats = dataset.get_statistics()
    print(f"\nDataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print(f"\nEvaluation method: {eval_method}")
    print(f"Evaluating {len(all_models)} models with DDP")

    # Setup output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run DDP evaluation
    run_ddp_evaluation(
        dataset=dataset,
        models_config=all_models,
        output_dir=output_dir,
        evaluation_method=eval_method,
        parser_fn=parser_fn,
        max_questions=max_questions,
    )

    # After DDP completes, compute metrics from merged results
    print("\n" + "=" * 80)
    print("COMPUTING METRICS")
    print("=" * 80)

    # Load merged results
    checkpoint_file = output_dir / "checkpoint.json"
    if not checkpoint_file.exists():
        print(f"Error: No checkpoint file found at {checkpoint_file}")
        return

    with open(checkpoint_file, "r") as f:
        all_results = json.load(f)

    # Group by model
    results_by_model = {}
    for result in all_results:
        model_name = result["model_name"]
        if model_name not in results_by_model:
            results_by_model[model_name] = []
        results_by_model[model_name].append(result)

    # Compute metrics for each model
    all_metrics = {}
    for model_name, results in results_by_model.items():
        metrics = metrics_cls.compute(results)
        all_metrics[model_name] = metrics

        # Generate and print report
        report = metrics_cls.generate_report(metrics, model_name)
        print(report)

    # Save metrics
    metrics_file = output_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMetrics saved to: {metrics_file}")

    # Save detailed results
    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to: {results_file}")

    # Cleanup dataset
    if hasattr(dataset, "clear_feature_cache"):
        dataset.clear_feature_cache()
    if hasattr(dataset, "cleanup"):
        dataset.cleanup()

    print(f"\n✅ DDP {benchmark.upper()} Evaluation complete!")


def main():
    """Main entry point."""
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

    parser = argparse.ArgumentParser(
        description="Unified Benchmark Runner for VQA evaluations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run VQA benchmark with DDP
  python -m validation.run_benchmark --benchmark vqa

  # Run VQA benchmark with single GPU
  python -m validation.run_benchmark --benchmark vqa --no-ddp

  # Test VQA with 8 questions
  python -m validation.run_benchmark --benchmark vqa --test --test-questions 8

  # Run specific models only
  python -m validation.run_benchmark --benchmark vqa --models antoni_alpha_18k_optimized
        """,
    )

    # Benchmark selection
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["vqa"],
        required=True,
        help="Which benchmark to run: vqa",
    )

    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config file (auto-detected if not specified)",
    )

    # GPU settings
    parser.add_argument(
        "--no-ddp", action="store_true", help="Disable DDP (multi-GPU) execution"
    )

    # Test mode
    parser.add_argument(
        "--test", action="store_true", help="Run in test mode with limited data"
    )
    parser.add_argument(
        "--test-questions",
        type=int,
        default=10,
        help="Number of questions in test mode (default: 10)",
    )

    # Model filtering
    parser.add_argument(
        "--models", nargs="+", help="Specific model names to evaluate (space-separated)"
    )

    args = parser.parse_args()

    # Auto-detect config if not specified
    if args.config is None:
        args.config = "validation/config.yaml"

    print("\n" + "=" * 80)
    print(f"{args.benchmark.upper()} BENCHMARK")
    print("=" * 80)
    print(f"Config: {args.config}")

    # Determine test parameters
    max_questions = args.test_questions if args.test else None

    # Check for DDP capability
    if torch.cuda.device_count() > 1 and not args.no_ddp:
        print(f"Detected {torch.cuda.device_count()} GPUs - using DDP")
        run_ddp(
            benchmark=args.benchmark,
            config_path=args.config,
            max_questions=max_questions,
            test_slides=None,
            models_to_run=args.models,
        )
    else:
        if args.no_ddp:
            print("DDP disabled by user - using single GPU")
        else:
            print("Single GPU detected")
        run_single_gpu(
            benchmark=args.benchmark,
            config_path=args.config,
            max_questions=max_questions,
            test_slides=None,
            models_to_run=args.models,
        )

    print(f"\n✅ {args.benchmark.upper()} Benchmark Complete!")


if __name__ == "__main__":
    main()
