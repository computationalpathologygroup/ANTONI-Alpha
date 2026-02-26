#!/usr/bin/env python3
"""
Create stratified subsets of the training data.

This script reads data/splits/train.txt and creates two stratified subsets:
- train_2k.txt: 2000 cases sampled proportionally from each subset
- train_9k.txt: 9000 cases sampled proportionally from each subset

The sampling is stratified by subset (HISTAI-breast, HISTAI-skin-b1, etc.)
to maintain the same proportions as the original training set.
"""

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List
import random


def load_and_group_cases(train_file: Path) -> Dict[str, List[str]]:
    """
    Load cases from train.txt and group by subset.

    Args:
        train_file: Path to train.txt file

    Returns:
        Dictionary mapping subset names to lists of case paths
    """
    subset_cases = defaultdict(list)

    with open(train_file, "r") as f:
        for line in f:
            case_path = line.strip()
            if not case_path:
                continue

            # Extract subset name from path (e.g., "histai/HISTAI-breast/case_1234" -> "HISTAI-breast")
            parts = case_path.split("/")
            if len(parts) >= 2:
                subset_name = parts[1]
                subset_cases[subset_name].append(case_path)

    return subset_cases


def create_stratified_sample(
    subset_cases: Dict[str, List[str]], target_size: int, seed: int = 42
) -> List[str]:
    """
    Create a stratified sample of cases proportional to subset sizes.

    Args:
        subset_cases: Dictionary mapping subset names to lists of case paths
        target_size: Target number of cases to sample
        seed: Random seed for reproducibility

    Returns:
        List of sampled case paths
    """
    random.seed(seed)

    # Calculate total cases
    total_cases = sum(len(cases) for cases in subset_cases.values())

    sampled_cases = []

    # Sample from each subset proportionally
    for subset_name, cases in sorted(subset_cases.items()):
        # Calculate proportion and target sample size for this subset
        proportion = len(cases) / total_cases
        subset_target = round(proportion * target_size)

        # Don't sample more than available
        subset_target = min(subset_target, len(cases))

        # Sample cases
        sampled = random.sample(cases, subset_target)
        sampled_cases.extend(sampled)

        print(
            f"{subset_name}: {len(cases)} total -> {subset_target} sampled ({proportion * 100:.1f}%)"
        )

    print(f"Total sampled: {len(sampled_cases)} (target: {target_size})")

    return sampled_cases


def save_cases(cases: List[str], output_file: Path) -> None:
    """
    Save cases to output file.

    Args:
        cases: List of case paths
        output_file: Path to output file
    """
    with open(output_file, "w") as f:
        for case in cases:
            f.write(f"{case}\n")
    print(f"Saved {len(cases)} cases to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Create stratified subsets of training data"
    )
    parser.add_argument(
        "--train-file",
        type=Path,
        default=Path("data/splits/train.txt"),
        help="Path to input train.txt file (default: data/splits/train.txt)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for subset files (default: same as train-file)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    # Determine output directory
    if args.output_dir is None:
        output_dir = args.train_file.parent
    else:
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

    # Load and group cases
    print(f"Loading cases from {args.train_file}...")
    subset_cases = load_and_group_cases(args.train_file)

    total_cases = sum(len(cases) for cases in subset_cases.values())
    print(f"\nFound {total_cases} total cases across {len(subset_cases)} subsets\n")

    # Print subset distribution
    print("Subset distribution:")
    for subset_name, cases in sorted(subset_cases.items()):
        proportion = len(cases) / total_cases
        print(f"  {subset_name}: {len(cases)} ({proportion * 100:.1f}%)")

    # Create 2k subset
    print("\n" + "=" * 60)
    print("Creating 2k subset...")
    print("=" * 60)
    cases_2k = create_stratified_sample(subset_cases, 2000, seed=args.seed)
    output_2k = output_dir / "train_2k.txt"
    save_cases(cases_2k, output_2k)

    # Create 9k subset
    print("\n" + "=" * 60)
    print("Creating 9k subset...")
    print("=" * 60)
    cases_9k = create_stratified_sample(subset_cases, 9000, seed=args.seed)
    output_9k = output_dir / "train_9k.txt"
    save_cases(cases_9k, output_9k)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
