#!/usr/bin/env python3
"""
Create train/validation/test splits using cluster sampling.

This script:
1. Loads cases from HISTAI-Instruct dataset (converted from JSONL to JSON)
2. Groups cases by subset (HISTAI-*)
3. For each subset:
   - Loads PRISM embeddings (first prototype)
   - Clusters with K-means (k=32)
   - Performs stratified 80-10-10 split by cluster
4. Outputs three text files with case_mappings
"""

import json
import h5py
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import glob
import re

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


class ClusterSamplingSplitter:
    """
    Creates train/val/test splits using cluster sampling approach.

    Args:
        prism_base_path (str): Base path to PRISM embeddings
        n_clusters (int): Number of K-means clusters per subset
        seed (int): Random seed for reproducibility
        exclude_files (List[str]): Text files with case_mappings to exclude
        min_samples_for_clustering (int): Minimum samples to use clustering (default: max(100, n_clusters * 3))
    """

    def __init__(
        self,
        prism_base_path: str,
        n_clusters: int = 32,
        seed: int = 42,
        exclude_files: Optional[List[str]] = None,
        min_samples_for_clustering: Optional[int] = None,
    ):
        self.prism_base_path = Path(prism_base_path)
        self.n_clusters = n_clusters
        self.seed = seed
        self.exclude_files = exclude_files or []
        # Use max(100, n_clusters * 3) as default threshold
        self.min_samples_for_clustering = min_samples_for_clustering or max(
            100, n_clusters * 3
        )

        # Set up logging
        self.logger = self._setup_logging()

        # State
        self.prism_index: Dict[str, str] = {}
        self.excluded_cases: Set[str] = set()

    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration"""
        logger = logging.getLogger(f"{__name__}.{id(self)}")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _load_excluded_cases(self) -> Set[str]:
        """
        Load case_mappings to exclude from text files.

        Returns:
            Set of case_mapping strings to exclude
        """
        excluded = set()

        for filepath in self.exclude_files:
            path = Path(filepath)
            if not path.exists():
                self.logger.warning(f"Exclude file not found: {filepath}")
                continue

            with open(path, "r") as f:
                for line in f:
                    case_mapping = line.strip()
                    if case_mapping:
                        excluded.add(case_mapping)

            self.logger.info(f"Loaded {len(excluded)} cases to exclude from {filepath}")

        if excluded:
            self.logger.info(f"Total cases to exclude: {len(excluded)}")

        return excluded

    def _extract_slide_id(self, case_mapping: str) -> Optional[str]:
        """
        Extract slide_id from case_mapping.

        Args:
            case_mapping: e.g., "histai/HISTAI-skin-b2/case_00009"

        Returns:
            Slide ID: e.g., "00009" or None if not found
        """
        parts = case_mapping.split("/")
        if parts:
            last_part = parts[-1]
            if last_part.startswith("case_"):
                return last_part[5:]  # Remove "case_" prefix
            else:
                return last_part
        return None

    def _extract_subset_from_case_mapping(self, case_mapping: str) -> Optional[str]:
        """
        Extract subset name from case_mapping.

        Args:
            case_mapping: e.g., "histai/HISTAI-skin-b2/case_00009"

        Returns:
            Subset: e.g., "HISTAI-skin-b2" or None if not found
        """
        parts = case_mapping.split("/")
        # Expecting format: "histai/HISTAI-{organ}-{batch}/case_{id}"
        if len(parts) >= 2:
            # Second part should be "HISTAI-*"
            if parts[1].startswith("HISTAI-"):
                return parts[1]
        return None

    def _build_prism_index(self) -> Dict[str, str]:
        """
        Build index of all PRISM H&E embeddings.

        Returns:
            Dict mapping slide_id to .h5 filepath
        """
        if not self.prism_base_path.exists():
            self.logger.warning(
                f"PRISM base path does not exist: {self.prism_base_path}"
            )
            return {}

        self.logger.info(f"Building PRISM index from: {self.prism_base_path}")

        # Pattern: {base_path}/HISTAI-*/20x_224px_0px_overlap/slide_features_prism/*_slide_H&E_0.h5
        pattern = str(
            self.prism_base_path
            / "HISTAI-*"
            / "20x_224px_0px_overlap"
            / "slide_features_prism"
            / "*_slide_H&E_0.h5"
        )

        h5_files = glob.glob(pattern)

        prism_index = {}
        subset_counts = defaultdict(int)

        for filepath in h5_files:
            # Extract subset and filename
            path_parts = Path(filepath).parts
            subset = path_parts[-4]  # HISTAI-{organ}-{batch}
            filename = Path(filepath).name

            # Extract slide_id from filename: {slide_id}_slide_H&E_0.h5
            match = re.match(r"(.+?)_slide_H&E_0\.h5", filename)
            if match:
                slide_id = match.group(1)

                # Use subset/slide_id as key to handle duplicates across subsets
                key = f"{subset}/{slide_id}"

                if key in prism_index:
                    self.logger.warning(
                        f"Duplicate slide_id '{slide_id}' in subset {subset} - keeping first occurrence"
                    )
                else:
                    prism_index[key] = filepath
                    subset_counts[subset] += 1

        self.logger.info(f"PRISM index built: {len(prism_index)} H&E slides found")
        for subset, count in sorted(subset_counts.items()):
            self.logger.info(f"  {subset}: {count} slides")

        return prism_index

    def _load_cases_from_json(self, json_path: str) -> Dict[str, List[str]]:
        """
        Load cases from HISTAI-Instruct JSON and group by subset.

        Args:
            json_path: Path to JSON file (converted from JSONL format)

        Returns:
            Dict mapping subset to list of case_mappings
        """
        self.logger.info(f"Loading cases from: {json_path}")

        with open(json_path, "r") as f:
            data = json.load(f)

        self.logger.info(f"Loaded {len(data)} cases from JSON")

        # Group by subset
        cases_by_subset = defaultdict(list)
        skipped_no_subset = 0
        skipped_excluded = 0

        for item in data:
            case_mapping = item.get("case_mapping", "")
            if not case_mapping:
                continue

            # Check if case should be excluded
            if case_mapping in self.excluded_cases:
                skipped_excluded += 1
                self.logger.debug(f"Excluding case: {case_mapping}")
                continue

            subset = self._extract_subset_from_case_mapping(case_mapping)
            if subset:
                cases_by_subset[subset].append(case_mapping)
            else:
                skipped_no_subset += 1
                self.logger.debug(f"Could not extract subset from: {case_mapping}")

        if skipped_excluded > 0:
            self.logger.info(
                f"Excluded {skipped_excluded} cases based on exclude files"
            )

        if skipped_no_subset > 0:
            self.logger.warning(
                f"Skipped {skipped_no_subset} cases without clear subset"
            )

        self.logger.info(f"Cases grouped into {len(cases_by_subset)} subsets:")
        for subset, cases in sorted(cases_by_subset.items()):
            self.logger.info(f"  {subset}: {len(cases)} cases")

        return dict(cases_by_subset)

    def _load_first_prototypes(
        self, case_mappings: List[str]
    ) -> Tuple[List[str], np.ndarray]:
        """
        Load first prototypes (1280-d) from PRISM .h5 files.

        Args:
            case_mappings: List of case_mapping strings

        Returns:
            Tuple of (valid_cases, embeddings) where:
            - valid_cases: list of case_mappings with embeddings
            - embeddings: numpy array of shape (N, 1280)
        """
        valid_cases = []
        embeddings_list = []
        missing_count = 0

        for case_mapping in case_mappings:
            slide_id = self._extract_slide_id(case_mapping)
            subset = self._extract_subset_from_case_mapping(case_mapping)

            if not slide_id or not subset:
                missing_count += 1
                self.logger.debug(
                    f"Could not extract slide_id/subset from: {case_mapping}"
                )
                continue

            # Use subset/slide_id as key
            key = f"{subset}/{slide_id}"

            if key not in self.prism_index:
                missing_count += 1
                self.logger.debug(f"No PRISM embedding for: {case_mapping}")
                continue

            filepath = self.prism_index[key]

            try:
                with h5py.File(filepath, "r") as f:
                    if "features" not in f:
                        self.logger.warning(f"No 'features' in {filepath}")
                        missing_count += 1
                        continue

                    features = f["features"][:]  # Shape: (513, 1280)

                    if features.shape != (513, 1280):
                        self.logger.warning(
                            f"Unexpected shape {features.shape} in {filepath}"
                        )
                        missing_count += 1
                        continue

                    # Extract first prototype
                    first_prototype = features[0, :].astype(
                        np.float32
                    )  # Shape: (1280,)

                    valid_cases.append(case_mapping)
                    embeddings_list.append(first_prototype)

            except Exception as e:
                self.logger.error(f"Error loading {filepath}: {e}")
                missing_count += 1
                continue

        if missing_count > 0:
            self.logger.warning(
                f"Excluded {missing_count} cases without valid embeddings"
            )

        if not embeddings_list:
            self.logger.error("No valid embeddings loaded!")
            return [], np.array([])

        embeddings = np.array(embeddings_list, dtype=np.float32)  # Shape: (N, 1280)
        self.logger.info(
            f"Loaded {len(valid_cases)} embeddings, shape: {embeddings.shape}"
        )

        return valid_cases, embeddings

    def _cluster_subset(
        self, embeddings: np.ndarray, n_clusters: int, seed: int
    ) -> np.ndarray:
        """
        Cluster embeddings using K-means.

        Args:
            embeddings: numpy array of shape (N, 1280)
            n_clusters: number of clusters
            seed: random seed

        Returns:
            cluster_labels: numpy array of shape (N,)
        """
        if len(embeddings) == 0:
            return np.array([])

        # Adjust n_clusters if we have fewer samples
        actual_n_clusters = min(n_clusters, len(embeddings))

        if actual_n_clusters < n_clusters:
            self.logger.warning(
                f"Only {len(embeddings)} samples, using {actual_n_clusters} clusters"
            )

        # Normalize embeddings
        embeddings_normalized = normalize(embeddings, norm="l2")

        # K-means clustering
        kmeans = KMeans(
            n_clusters=actual_n_clusters,
            random_state=seed,
            n_init=10,
        )
        cluster_labels = kmeans.fit_predict(embeddings_normalized)

        # Log cluster distribution
        unique, counts = np.unique(cluster_labels, return_counts=True)
        self.logger.debug(f"Cluster distribution: {dict(zip(unique, counts))}")

        return cluster_labels

    def _simple_random_split(
        self,
        cases: List[str],
        ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 42,
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Perform simple random split without clustering.

        Args:
            cases: list of case_mappings
            ratios: (train, val, test) ratios
            seed: random seed

        Returns:
            Tuple of (train_cases, val_cases, test_cases)
        """
        np.random.seed(seed)

        n_total = len(cases)

        # Shuffle
        cases = cases.copy()
        np.random.shuffle(cases)

        # Calculate split sizes using floor for train/val, remainder for test
        n_train = int(n_total * ratios[0])
        n_val = int(n_total * ratios[1])
        n_test = n_total - n_train - n_val

        # Split
        train_cases = cases[:n_train]
        val_cases = cases[n_train : n_train + n_val]
        test_cases = cases[n_train + n_val :]

        return train_cases, val_cases, test_cases

    def _split_clusters(
        self,
        cases: List[str],
        cluster_labels: np.ndarray,
        ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 42,
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Perform stratified split by cluster.

        Args:
            cases: list of case_mappings
            cluster_labels: cluster assignment for each case
            ratios: (train, val, test) ratios
            seed: random seed

        Returns:
            Tuple of (train_cases, val_cases, test_cases)
        """
        np.random.seed(seed)

        train_cases, val_cases, test_cases = [], [], []

        # Group cases by cluster
        clusters = defaultdict(list)
        for case, label in zip(cases, cluster_labels):
            clusters[label].append(case)

        # Split each cluster
        for cluster_id, cluster_cases in clusters.items():
            n_total = len(cluster_cases)

            # Shuffle cases
            np.random.shuffle(cluster_cases)

            # Calculate split sizes using rounding for better distribution
            if n_total == 1:
                # Single sample: assign to train
                n_train, n_val, n_test = 1, 0, 0
            elif n_total == 2:
                # Two samples: 1 train, 1 test
                n_train, n_val, n_test = 1, 0, 1
            elif n_total == 3:
                # Three samples: 1 each
                n_train, n_val, n_test = 1, 1, 1
            else:
                # Four or more: use rounding for proportional allocation
                n_val = max(1, round(n_total * ratios[1]))
                n_test = max(1, round(n_total * ratios[2]))
                n_train = n_total - n_val - n_test

                # Ensure train is at least 1
                if n_train < 1:
                    # Adjust val or test
                    if n_val > 1:
                        n_val -= 1
                    elif n_test > 1:
                        n_test -= 1
                    n_train = n_total - n_val - n_test

            # Split
            train_cases.extend(cluster_cases[:n_train])
            val_cases.extend(cluster_cases[n_train : n_train + n_val])
            test_cases.extend(cluster_cases[n_train + n_val :])

            self.logger.debug(
                f"Cluster {cluster_id}: {n_total} total → "
                f"{n_train} train, {n_val} val, {n_test} test"
            )

        return train_cases, val_cases, test_cases

    def _write_splits(
        self,
        train_cases: List[str],
        val_cases: List[str],
        test_cases: List[str],
        output_dir: str,
    ):
        """
        Write splits to text files.

        Args:
            train_cases: list of train case_mappings
            val_cases: list of val case_mappings
            test_cases: list of test case_mappings
            output_dir: output directory path
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        splits = {
            "train.txt": train_cases,
            "val.txt": val_cases,
            "test.txt": test_cases,
        }

        for filename, cases in splits.items():
            filepath = output_path / filename
            with open(filepath, "w") as f:
                for case in cases:
                    f.write(f"{case}\n")
            self.logger.info(f"Wrote {len(cases)} cases to {filepath}")

    def run(
        self,
        json_path: str,
        output_dir: str = "data/splits/",
    ):
        """
        Main execution flow.

        Args:
            json_path: path to HISTAI-Instruct JSON (converted from JSONL)
            output_dir: output directory for splits
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting cluster sampling split creation")
        self.logger.info(f"Parameters: n_clusters={self.n_clusters}, seed={self.seed}")
        self.logger.info(
            f"  min_samples_for_clustering={self.min_samples_for_clustering}"
        )
        self.logger.info("=" * 60)

        # 0. Load excluded cases
        if self.exclude_files:
            self.logger.info("")
            self.logger.info("Loading excluded cases...")
            self.excluded_cases = self._load_excluded_cases()

        # 1. Load cases and group by subset
        self.logger.info("")
        cases_by_subset = self._load_cases_from_json(json_path)

        # 2. Build PRISM index
        self.prism_index = self._build_prism_index()

        # 3. Process each subset
        all_train, all_val, all_test = [], [], []

        for subset, cases in sorted(cases_by_subset.items()):
            self.logger.info("")
            self.logger.info(f"Processing subset: {subset} ({len(cases)} cases)")

            # Determine if we should use clustering based on sample size
            use_clustering = len(cases) >= self.min_samples_for_clustering

            if not use_clustering:
                self.logger.info(
                    f"  Using simple random split (< {self.min_samples_for_clustering} samples)"
                )

                # Load embeddings to filter valid cases (but don't need to cluster)
                valid_cases, _ = self._load_first_prototypes(cases)

                if len(valid_cases) == 0:
                    self.logger.warning(f"No valid embeddings for {subset}, skipping")
                    continue

                # Simple random split
                train, val, test = self._simple_random_split(
                    valid_cases, seed=self.seed
                )

            else:
                self.logger.info(f"  Using cluster-based split")

                # Load embeddings
                valid_cases, embeddings = self._load_first_prototypes(cases)

                if len(valid_cases) == 0:
                    self.logger.warning(f"No valid embeddings for {subset}, skipping")
                    continue

                # Cluster
                cluster_labels = self._cluster_subset(
                    embeddings, self.n_clusters, self.seed
                )

                # Split
                train, val, test = self._split_clusters(
                    valid_cases, cluster_labels, seed=self.seed
                )

            self.logger.info(
                f"  Split: {len(train)} train, {len(val)} val, {len(test)} test"
            )

            all_train.extend(train)
            all_val.extend(val)
            all_test.extend(test)

        # 4. Write outputs
        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info(f"Final totals:")
        self.logger.info(
            f"  Train: {len(all_train)} cases ({len(all_train) / (len(all_train) + len(all_val) + len(all_test)) * 100:.1f}%)"
        )
        self.logger.info(
            f"  Val:   {len(all_val)} cases ({len(all_val) / (len(all_train) + len(all_val) + len(all_test)) * 100:.1f}%)"
        )
        self.logger.info(
            f"  Test:  {len(all_test)} cases ({len(all_test) / (len(all_train) + len(all_val) + len(all_test)) * 100:.1f}%)"
        )
        self.logger.info("=" * 60)

        self._write_splits(all_train, all_val, all_test, output_dir)

        self.logger.info("Split creation completed successfully!")


def main():
    """Main function for command line usage"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Create train/val/test splits using cluster sampling"
    )
    parser.add_argument(
        "--input", required=True, help="Path to HISTAI-Instruct JSON file (converted from JSONL)"
    )
    parser.add_argument(
        "--output-dir",
        default="data/splits/",
        help="Output directory for split files (default: data/splits/)",
    )
    parser.add_argument(
        "--prism-base-path",
        required=True,
        help="Base path for PRISM embeddings (created using TRIDENT: https://github.com/mahmoodlab/TRIDENT)",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=32,
        help="Number of K-means clusters per subset (default: 32)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        help="Text files with case_mappings to exclude (e.g., data/cases_without_he_images_formatted.txt)",
    )
    parser.add_argument(
        "--min-samples-for-clustering",
        type=int,
        help="Minimum samples required to use clustering (default: max(100, n_clusters * 3))",
    )

    args = parser.parse_args()

    # Create splitter
    splitter = ClusterSamplingSplitter(
        prism_base_path=args.prism_base_path,
        n_clusters=args.n_clusters,
        seed=args.seed,
        exclude_files=args.exclude,
        min_samples_for_clustering=args.min_samples_for_clustering,
    )

    # Run
    splitter.run(
        json_path=args.input,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
