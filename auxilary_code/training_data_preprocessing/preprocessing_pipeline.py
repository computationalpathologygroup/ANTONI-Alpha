#!/usr/bin/env python3
"""
This pipeline processes HISTAI-Instruct data through three main steps:
1. Case filtering based on provided filter list (e.g., train.txt from create_splits.py)
2. PRISM-based clustering by organ type using first prototype embeddings
3. HDF5 finalization with PRISM features and metadata

Features:
- Resumable: each step can be skipped if already completed
- Uses PRISM embeddings (513 prototypes x 1280-d) from H&E slides (created with TRIDENT)
- Comprehensive logging and error handling
- Modular design for easy testing and maintenance
- HDF5 output compatible with ANTONI-Alpha dataset classes
"""

import json
import h5py
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict

# ML imports
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


class PreprocessingPipeline:
    """
    Main preprocessing pipeline class that orchestrates the four-step data processing.

    Args:
        input_dataset_path (str): Path to the input JSON dataset
        filter_cases_path (str): Path to the text file with case mappings to keep
        output_hdf5_path (str): Path to the output HDF5 file
        image_encodings_path (Optional[str]): Path to image encodings (future use)
        resume (bool): Whether to resume from existing progress
        config (Optional[Dict]): Configuration parameters
    """

    def __init__(
        self,
        input_dataset_path: str,
        filter_cases_path: Optional[str] = None,
        output_hdf5_path: str = None,
        image_encodings_path: Optional[str] = None,
        resume: bool = False,
        config: Optional[Dict[str, Any]] = None,
        force_fresh: bool = False,
    ):
        self.input_dataset_path = Path(input_dataset_path)
        self.filter_cases_path = Path(filter_cases_path) if filter_cases_path else None
        self.output_hdf5_path = Path(output_hdf5_path)
        self.image_encodings_path = (
            Path(image_encodings_path) if image_encodings_path else None
        )
        self.resume = resume and not force_fresh
        self.force_fresh = force_fresh

        # Default configuration
        self.config = {
            "n_clusters": 10,
            "image_feature_shape": (
                513,
                1280,
            ),  # PRISM prototypes: 513 prototypes x 1280 dimensions
            "random_seed": 42,
            "clustering_mode": "image",  # Use PRISM image features for clustering
            "embedding_type": "prism",  # Type of embeddings to store: 'prism' or 'virchow'
            "prism_embeddings": {
                "base_path": None,  # Must be provided via config or command line
                "enabled": True,
            },
        }
        if config:
            self.config.update(config)

        # Set up logging
        self.logger = self._setup_logging()

        # Initialize state tracking
        self.data: List[Dict] = []
        self.filtered_data: List[Dict] = []
        self.filter_cases: Set[str] = set()
        self.cluster_stats: Dict[str, Dict] = {}
        self.processing_log: Dict[str, str] = {}
        self.prism_index: Dict[str, str] = {}  # Maps subset/slide_id to .h5 filepath
        self.virchow_index: Dict[str, str] = {}  # Maps subset/slide_id to .h5 filepath

        # Create output directory
        self.output_hdf5_path.parent.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Initialized preprocessing pipeline")
        self.logger.info(f"Input dataset: {self.input_dataset_path}")
        if self.filter_cases_path:
            self.logger.info(f"Filter cases: {self.filter_cases_path}")
        else:
            self.logger.info(f"Filter cases: Auto-discovery from PRISM index")
        self.logger.info(f"Output HDF5: {self.output_hdf5_path}")
        self.logger.info(f"Resume mode: {self.resume}")
        self.logger.info(
            f"Embedding type: {self.config.get('embedding_type', 'prism').upper()}"
        )

        # Build PRISM and Virchow embeddings indices
        if self.config.get("prism_embeddings", {}).get("enabled", False):
            self.prism_index = self._build_prism_index()
            self.virchow_index = self._build_virchow_index()
        else:
            self.logger.info("PRISM embeddings disabled in config")

    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration"""
        logger = logging.getLogger(f"{__name__}.{id(self)}")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _load_processing_log(self) -> Dict[str, str]:
        """Load processing log from existing HDF5 file if available"""
        if not self.resume or not self.output_hdf5_path.exists():
            return {}

        try:
            with h5py.File(self.output_hdf5_path, "r") as f:
                if "metadata/processing_log" in f:
                    log_data = f["metadata/processing_log"][()].decode("utf-8")
                    return json.loads(log_data)
        except Exception as e:
            self.logger.warning(f"Could not load processing log: {e}")

        return {}

    def _save_processing_log(self, step_name: str):
        """Save processing log to HDF5 file"""
        self.processing_log[f"{step_name}_completed"] = datetime.now().isoformat()

        # Save to HDF5 if file exists
        if self.output_hdf5_path.exists():
            try:
                with h5py.File(self.output_hdf5_path, "a") as f:
                    # Ensure metadata group exists
                    if "metadata" not in f:
                        f.create_group("metadata")

                    # Save processing log
                    log_str = json.dumps(self.processing_log, indent=2)
                    if "metadata/processing_log" in f:
                        del f["metadata/processing_log"]
                    f.create_dataset(
                        "metadata/processing_log", data=log_str.encode("utf-8")
                    )

            except Exception as e:
                self.logger.error(f"Failed to save processing log: {e}")

    def _is_step_completed(self, step_name: str) -> bool:
        """Check if a processing step has been completed"""
        if not self.resume or self.force_fresh:
            return False

        return f"{step_name}_completed" in self.processing_log

    def run(self):
        """Run the complete preprocessing pipeline"""
        self.logger.info("Starting preprocessing pipeline")

        # Load existing processing log
        self.processing_log = self._load_processing_log()

        try:
            # Step 1: Filter cases
            if not self._is_step_completed("step_1_filtering"):
                self.logger.info("Step 1: Filtering cases")
                self._step_1_filter_cases()
                self._save_processing_log("step_1_filtering")
            else:
                self.logger.info(
                    "Step 1: Filtering cases (SKIPPED - already completed)"
                )
                self._load_filtered_data()

            # Step 2: Text embeddings removed - using PRISM embeddings instead
            self.logger.info(
                "Step 2: Text embeddings (SKIPPED - using PRISM slide embeddings)"
            )
            self._save_processing_log("step_2_embeddings")

            # Step 3: Cluster by organ
            if not self._is_step_completed("step_3_clustering"):
                self.logger.info("Step 3: Clustering by organ")
                self._step_3_cluster_by_organ()
                self._save_processing_log("step_3_clustering")
            else:
                self.logger.info("Step 3: Clustering (SKIPPED - already completed)")
                self._load_cluster_stats()

            # Step 4: Generate image features and create final HDF5
            if not self._is_step_completed("step_4_finalization"):
                self.logger.info(
                    "Step 4: Generating image features and finalizing HDF5"
                )
                self._step_4_finalize_hdf5()
                self._save_processing_log("step_4_finalization")
            else:
                self.logger.info("Step 4: Finalization (SKIPPED - already completed)")

            self.logger.info("Preprocessing pipeline completed successfully!")
            self._print_summary()

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise

    def _step_1_filter_cases(self):
        """Step 1: Filter dataset based on provided case list or auto-discover from PRISM"""
        self.logger.info("Loading input dataset...")
        with open(self.input_dataset_path, "r") as f:
            self.data = json.load(f)
        self.logger.info(f"Loaded {len(self.data)} cases from input dataset")

        # Filter based on provided file or auto-discover
        if self.filter_cases_path and self.filter_cases_path.exists():
            self.logger.info("Loading filter cases from file...")
            with open(self.filter_cases_path, "r") as f:
                self.filter_cases = {line.strip() for line in f if line.strip()}
            self.logger.info(f"Loaded {len(self.filter_cases)} cases to keep")

            # Filter the dataset
            self.filtered_data = []
            for case in self.data:
                case_mapping = case.get("case_mapping", "")
                if case_mapping in self.filter_cases:
                    self.filtered_data.append(case)

            filter_source = str(self.filter_cases_path)
        else:
            self.logger.info(
                "No filter file provided - auto-discovering cases with PRISM embeddings"
            )

            # Filter to only cases that have PRISM embeddings
            self.filtered_data = []
            for case in self.data:
                case_mapping = case.get("case_mapping", "")
                slide_id = self._extract_slide_id_from_case_mapping(case_mapping)

                if slide_id and slide_id in self.prism_index:
                    self.filtered_data.append(case)

            filter_source = "auto-discovery (PRISM index)"

        self.logger.info(
            f"Filtered dataset: {len(self.filtered_data)} cases (from {len(self.data)})"
        )

        # Save filtering statistics
        self.filtering_stats = {
            "original_count": len(self.data),
            "filtered_count": len(self.filtered_data),
            "filter_source": filter_source,
            "filter_ratio": len(self.filtered_data) / len(self.data)
            if self.data
            else 0,
        }

        # Add image embeddings validation if using image clustering
        clustering_mode = self.config.get("clustering_mode", "text")
        if clustering_mode in ["image", "both"]:
            self._validate_image_embeddings_coverage()

        # Save filtered data and stats to HDF5
        self._create_initial_hdf5(self.filtering_stats)

    def _load_filtered_data(self):
        """Load filtered data when resuming"""
        # Load from HDF5 text_attributes
        if self.output_hdf5_path.exists():
            with h5py.File(self.output_hdf5_path, "r") as f:
                self.filtered_data = []
                if "text_attributes" in f:
                    for slide_id in f["text_attributes"].keys():
                        text_data = f["text_attributes"][slide_id][()].decode("utf-8")
                        case_data = json.loads(text_data)
                        self.filtered_data.append(case_data)
            self.logger.info(
                f"Loaded {len(self.filtered_data)} filtered cases from HDF5"
            )

    def _step_2_generate_embeddings(self):
        """Step 2: Generate text embeddings from clean_report fields"""
        # Load existing cache first
        self._load_embeddings_cache()

        # Collect texts that need embedding (skip cached ones)
        texts_to_embed = []
        case_mappings_to_embed = []

        for case in self.filtered_data:
            case_mapping = case.get("case_mapping", "")

            # Skip if already cached
            if case_mapping in self.embeddings_cache:
                continue

            clean_report = case.get("clean_report", [])
            assistant_answer = self._extract_assistant_answer(clean_report)

            if assistant_answer:
                texts_to_embed.append(assistant_answer)
                case_mappings_to_embed.append(case_mapping)
            else:
                self.logger.warning(f"No assistant answer found for {case_mapping}")
                # Store empty embedding for cases without clean_report
                self.embeddings_cache[case_mapping] = [0.0] * self.config[
                    "embedding_dim"
                ]

        # Check if we need to generate any new embeddings
        if not texts_to_embed:
            self.logger.info("All embeddings already cached!")
            return

        self.logger.info(f"Found {len(self.embeddings_cache)} cached embeddings")
        self.logger.info(
            f"Generating embeddings for {len(texts_to_embed)} new cases..."
        )

        # Load embedding model only if needed
        self.logger.info("Loading embedding model...")
        self.embedding_model = SentenceTransformer(self.config["embedding_model"])

        # Detect actual embedding dimension from the model
        test_embedding = self.embedding_model.encode(["test text"])
        actual_dim = (
            test_embedding.shape[1]
            if len(test_embedding.shape) > 1
            else len(test_embedding[0])
        )

        if actual_dim != self.config["embedding_dim"]:
            self.logger.warning(
                f"Model embedding dimension ({actual_dim}) differs from config ({self.config['embedding_dim']})"
            )
            self.logger.info(f"Updating config to use actual dimension: {actual_dim}")
            self.config["embedding_dim"] = actual_dim

        # Generate embeddings in batches with incremental caching
        batch_size = self.config["batch_size"]
        total_batches = (len(texts_to_embed) - 1) // batch_size + 1

        for i in range(0, len(texts_to_embed), batch_size):
            batch_end = min(i + batch_size, len(texts_to_embed))
            batch_texts = texts_to_embed[i:batch_end]
            batch_mappings = case_mappings_to_embed[i:batch_end]

            batch_num = i // batch_size + 1
            self.logger.info(
                f"Processing batch {batch_num}/{total_batches} ({len(batch_texts)} cases)"
            )

            # Generate embeddings for this batch
            try:
                embeddings = self.embedding_model.encode(
                    batch_texts, show_progress_bar=True
                )

                # Add to cache with dimension validation
                for case_mapping, embedding in zip(batch_mappings, embeddings):
                    embedding_array = np.array(embedding, dtype=np.float32)

                    # Validate embedding dimension
                    if embedding_array.shape[0] != self.config["embedding_dim"]:
                        self.logger.warning(
                            f"Unexpected embedding dimension for {case_mapping}: "
                            f"got {embedding_array.shape[0]}, expected {self.config['embedding_dim']}"
                        )

                        # Adjust to expected dimension
                        if embedding_array.shape[0] < self.config["embedding_dim"]:
                            # Pad with zeros
                            padding_size = (
                                self.config["embedding_dim"] - embedding_array.shape[0]
                            )
                            padding = np.zeros(padding_size, dtype=np.float32)
                            embedding_array = np.concatenate([embedding_array, padding])
                        else:
                            # Truncate
                            embedding_array = embedding_array[
                                : self.config["embedding_dim"]
                            ]

                    self.embeddings_cache[case_mapping] = embedding_array.tolist()

                # Save cache after each batch for crash recovery
                self._save_embeddings_cache()
                self.logger.debug(
                    f"Batch {batch_num} cached ({len(self.embeddings_cache)} total)"
                )

            except Exception as e:
                self.logger.error(f"Failed to process batch {batch_num}: {e}")
                raise

        self.logger.info(
            f"Generated embeddings for {len(self.embeddings_cache)} total cases"
        )
        self.logger.info(f"Cache saved to: {self.embeddings_cache_file}")

    def _load_embeddings_cache(self):
        """Load embeddings cache when resuming"""
        self.embeddings_cache = {}

        # Skip cache loading if force_fresh is set
        if self.force_fresh:
            self.logger.info("Force fresh mode: skipping cache loading")
            return

        # Try to load from JSON cache file first (more reliable)
        if self.embeddings_cache_file.exists():
            try:
                self.logger.info(
                    f"Loading embeddings cache from: {self.embeddings_cache_file}"
                )
                with open(self.embeddings_cache_file, "r") as f:
                    cache_data = json.load(f)

                # Validate cache format and add metadata checks
                if isinstance(cache_data, dict) and "embeddings" in cache_data:
                    raw_embeddings = cache_data["embeddings"]
                    cache_metadata = cache_data.get("metadata", {})

                    # Check if cache is compatible with current config
                    cached_model = cache_metadata.get("embedding_model", "")
                    current_model = self.config["embedding_model"]

                    if cached_model != current_model:
                        self.logger.warning(
                            f"Cache model mismatch: cached={cached_model}, current={current_model}"
                        )
                        self.logger.warning("Cache will be rebuilt with new model")
                        self.embeddings_cache = {}
                    else:
                        # Validate embedding dimensions
                        valid_embeddings = {}
                        invalid_count = 0

                        for case_mapping, embedding in raw_embeddings.items():
                            if (
                                isinstance(embedding, list)
                                and len(embedding) == self.config["embedding_dim"]
                            ):
                                valid_embeddings[case_mapping] = embedding
                            else:
                                invalid_count += 1
                                self.logger.debug(
                                    f"Invalid embedding for {case_mapping}: "
                                    f"expected dim {self.config['embedding_dim']}, got {len(embedding) if isinstance(embedding, list) else 'non-list'}"
                                )

                        self.embeddings_cache = valid_embeddings

                        if invalid_count > 0:
                            self.logger.warning(
                                f"Discarded {invalid_count} invalid embeddings from cache"
                            )

                        self.logger.info(
                            f"Loaded {len(self.embeddings_cache)} valid embeddings from cache"
                        )
                        self.logger.info(f"Cache model: {cached_model}")
                else:
                    self.logger.warning("Invalid cache format, starting fresh")

            except Exception as e:
                self.logger.error(f"Failed to load embeddings cache: {e}")
                self.embeddings_cache = {}

        # Fallback: try to load from existing HDF5 file
        elif self.output_hdf5_path.exists():
            self.logger.info("No JSON cache found, trying to load from HDF5...")
            try:
                with h5py.File(self.output_hdf5_path, "r") as f:
                    if "embeddings" in f:
                        # Build cache from HDF5 data using actual case mappings
                        for case in self.filtered_data:
                            case_mapping = case.get("case_mapping", "")
                            slide_id = (
                                case_mapping.split("/")[-1]
                                if "/" in case_mapping
                                else case_mapping
                            )

                            if (
                                slide_id in f["embeddings"]
                                and "text_embedding" in f["embeddings"][slide_id]
                            ):
                                embedding = f["embeddings"][slide_id]["text_embedding"][
                                    :
                                ]
                                self.embeddings_cache[case_mapping] = embedding.tolist()

                self.logger.info(
                    f"Loaded {len(self.embeddings_cache)} embeddings from HDF5"
                )

                # Save to JSON cache for future use
                self._save_embeddings_cache()

            except Exception as e:
                self.logger.error(f"Failed to load embeddings from HDF5: {e}")
                self.embeddings_cache = {}

        else:
            self.logger.info("No existing cache found, starting fresh")

    def _save_embeddings_cache(self):
        """Save embeddings cache to JSON file with metadata"""
        try:
            cache_data = {
                "metadata": {
                    "embedding_model": self.config["embedding_model"],
                    "embedding_dim": self.config["embedding_dim"],
                    "created_at": datetime.now().isoformat(),
                    "total_embeddings": len(self.embeddings_cache),
                },
                "embeddings": self.embeddings_cache,
            }

            # Write to temporary file first, then rename (atomic operation)
            temp_file = self.embeddings_cache_file.with_suffix(".tmp")
            with open(temp_file, "w") as f:
                json.dump(cache_data, f, indent=2)

            # Atomic rename
            temp_file.rename(self.embeddings_cache_file)

            self.logger.debug(f"Saved {len(self.embeddings_cache)} embeddings to cache")

        except Exception as e:
            self.logger.error(f"Failed to save embeddings cache: {e}")

    def _extract_assistant_answer(self, clean_report: List[Dict]) -> str:
        """Extract assistant answer from clean_report field"""
        if isinstance(clean_report, list):
            for entry in clean_report:
                if entry.get("role") == "assistant":
                    return entry.get("content", "")
        return ""

    def _extract_slide_id_from_case_mapping(self, case_mapping: str) -> Optional[str]:
        """
        Extract slide_id from case_mapping.

        Args:
            case_mapping: e.g., "histai/HISTAI-skin-b2/case_00009"

        Returns:
            Slide ID: e.g., "00009" or None if not found
        """
        # Case mapping format: "histai/HISTAI-{organ}-{batch}/case_{slide_id}"
        # Extract the last part and remove "case_" prefix
        parts = case_mapping.split("/")
        if parts:
            last_part = parts[-1]
            if last_part.startswith("case_"):
                return last_part[5:]  # Remove "case_" prefix
            else:
                # Fallback: return last part as-is
                return last_part
        return None

    def _get_hdf5_key_from_case_mapping(self, case_mapping: str) -> str:
        """
        Convert case_mapping to a valid HDF5 key that preserves uniqueness.

        Args:
            case_mapping: e.g., "histai/HISTAI-skin-b2/case_00009"

        Returns:
            HDF5-safe key: e.g., "HISTAI-skin-b2__case_00009"
        """
        # Remove leading "histai/" prefix if present
        if case_mapping.startswith("histai/"):
            case_mapping = case_mapping[7:]

        # Replace "/" with "__" to create HDF5-safe key while preserving uniqueness
        # Format: "HISTAI-skin-b2__case_00009"
        hdf5_key = case_mapping.replace("/", "__")

        return hdf5_key

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
        Build index of all PRISM H&E embeddings by recursively scanning base path.

        Returns:
            Dict mapping "subset/slide_id" to .h5 filepath
            Example: {"HISTAI-skin-b2/00009": "/path/to/HISTAI-skin-b2/.../00009_slide_H&E_0.h5"}
        """
        prism_base_path = Path(
            self.config.get("prism_embeddings", {}).get("base_path", "")
        )

        if not prism_base_path.exists():
            self.logger.warning(f"PRISM base path does not exist: {prism_base_path}")
            return {}

        self.logger.info(f"Building PRISM index from: {prism_base_path}")

        # Pattern: {base_path}/HISTAI-*/20x_224px_0px_overlap/slide_features_prism/*_slide_H&E_0.h5
        pattern = str(
            prism_base_path
            / "HISTAI-*"
            / "20x_224px_0px_overlap"
            / "slide_features_prism"
            / "*_slide_H&E_0.h5"
        )

        import glob
        import re

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

                # Check for duplicates within same subset
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

    def _build_virchow_index(self) -> Dict[str, str]:
        """
        Build index of all Virchow H&E embeddings by recursively scanning base path.

        Returns:
            Dict mapping "subset/slide_id" to .h5 filepath
            Example: {"HISTAI-skin-b2/00009": "/path/to/HISTAI-skin-b2/.../00009_slide_H&E_0.h5"}
        """
        prism_base_path = Path(
            self.config.get("prism_embeddings", {}).get("base_path", "")
        )

        if not prism_base_path.exists():
            self.logger.warning(f"PRISM base path does not exist: {prism_base_path}")
            return {}

        self.logger.info(f"Building Virchow index from: {prism_base_path}")

        # Pattern: {base_path}/HISTAI-*/20x_224px_0px_overlap/features_virchow/*_slide_H&E_0.h5
        pattern = str(
            prism_base_path
            / "HISTAI-*"
            / "20x_224px_0px_overlap"
            / "features_virchow"
            / "*_slide_H&E_0.h5"
        )

        import glob
        import re

        h5_files = glob.glob(pattern)

        virchow_index = {}
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

                # Check for duplicates within same subset
                if key in virchow_index:
                    self.logger.warning(
                        f"Duplicate slide_id '{slide_id}' in subset {subset} - keeping first occurrence"
                    )
                else:
                    virchow_index[key] = filepath
                    subset_counts[subset] += 1

        self.logger.info(f"Virchow index built: {len(virchow_index)} H&E slides found")
        for subset, count in sorted(subset_counts.items()):
            self.logger.info(f"  {subset}: {count} slides")

        return virchow_index

    def _load_prism_embeddings(self, filepath: str) -> Optional[np.ndarray]:
        """
        Load PRISM prototype features from .h5 file.

        Args:
            filepath: Path to .h5 file

        Returns:
            Features array of shape (513, 1280) or None if failed
        """
        try:
            with h5py.File(filepath, "r") as f:
                if "features" not in f:
                    self.logger.warning(f"No 'features' dataset in {filepath}")
                    return None

                features = f["features"][:]

                # Validate shape
                if features.shape != (513, 1280):
                    self.logger.warning(
                        f"Unexpected features shape in {filepath}: "
                        f"expected (513, 1280), got {features.shape}"
                    )
                    return None

                return features.astype(np.float32)

        except Exception as e:
            self.logger.error(f"Error loading PRISM embeddings from {filepath}: {e}")
            return None

    def _get_prism_features(self, case_mapping: str) -> Optional[np.ndarray]:
        """
        Get PRISM features for a case mapping.

        Args:
            case_mapping: Case mapping string (e.g., "histai/HISTAI-skin-b2/case_00009")

        Returns:
            Features array of shape (513, 1280) or None if not found
        """
        slide_id = self._extract_slide_id_from_case_mapping(case_mapping)
        subset = self._extract_subset_from_case_mapping(case_mapping)

        if slide_id is None or subset is None:
            self.logger.debug(
                f"Could not extract slide_id/subset from case_mapping: {case_mapping}"
            )
            return None

        # Use subset/slide_id as key
        key = f"{subset}/{slide_id}"

        if key not in self.prism_index:
            self.logger.debug(f"No PRISM embeddings found for: {key}")
            return None

        filepath = self.prism_index[key]
        return self._load_prism_embeddings(filepath)

    def _get_prism_first_prototype(self, case_mapping: str) -> Optional[np.ndarray]:
        """
        Get first PRISM prototype (1280-d) as slide-level embedding.

        Args:
            case_mapping: Case mapping string

        Returns:
            First prototype array of shape (1280,) or None if not found
        """
        prism_features = self._get_prism_features(case_mapping)
        if prism_features is not None:
            # Extract first prototype (row 0)
            return prism_features[0, :].astype(np.float32)  # Shape: (1280,)
        return None

    def _load_virchow_embeddings(self, filepath: str) -> Optional[np.ndarray]:
        """
        Load Virchow patch features from .h5 file.

        Args:
            filepath: Path to .h5 file

        Returns:
            Features array of shape (N, 2560) or None if failed
        """
        try:
            with h5py.File(filepath, "r") as f:
                if "features" not in f:
                    self.logger.warning(f"No 'features' dataset in {filepath}")
                    return None

                features = f["features"][:]

                # Validate shape - should be (N, 2560) where N is variable
                if len(features.shape) != 2 or features.shape[1] != 2560:
                    self.logger.warning(
                        f"Unexpected features shape in {filepath}: "
                        f"expected (N, 2560), got {features.shape}"
                    )
                    return None

                return features.astype(np.float32)

        except Exception as e:
            self.logger.error(f"Error loading Virchow embeddings from {filepath}: {e}")
            return None

    def _get_virchow_features(self, case_mapping: str) -> Optional[np.ndarray]:
        """
        Get Virchow features for a case mapping.

        Args:
            case_mapping: Case mapping string (e.g., "histai/HISTAI-skin-b2/case_00009")

        Returns:
            Features array of shape (N, 2560) or None if not found
        """
        slide_id = self._extract_slide_id_from_case_mapping(case_mapping)
        subset = self._extract_subset_from_case_mapping(case_mapping)

        if slide_id is None or subset is None:
            self.logger.debug(
                f"Could not extract slide_id/subset from case_mapping: {case_mapping}"
            )
            return None

        # Use subset/slide_id as key
        key = f"{subset}/{slide_id}"

        if key not in self.virchow_index:
            self.logger.debug(f"No Virchow embeddings found for: {key}")
            return None

        filepath = self.virchow_index[key]
        return self._load_virchow_embeddings(filepath)

    def _validate_image_embeddings_coverage(self):
        """
        Validate which cases have embeddings and filter dataset accordingly.

        Uses pre-built index (PRISM or Virchow) for instant validation.
        Saves excluded cases to a text file for analysis.
        """
        embedding_type = self.config.get("embedding_type", "prism")

        # Select appropriate index based on embedding type
        if embedding_type == "virchow":
            index = self.virchow_index
            index_name = "Virchow"
        else:
            index = self.prism_index
            index_name = "PRISM"

        if not index:
            self.logger.info(
                f"No {index_name} index available, skipping coverage validation"
            )
            return

        self.logger.info(f"Validating case coverage against {index_name} index...")

        available_cases = []
        excluded_cases = []

        for case in self.filtered_data:
            case_mapping = case.get("case_mapping", "")
            slide_id = self._extract_slide_id_from_case_mapping(case_mapping)
            subset = self._extract_subset_from_case_mapping(case_mapping)

            # Build key in format "subset/slide_id" to match index
            key = f"{subset}/{slide_id}" if subset and slide_id else None

            if key and key in index:
                available_cases.append(case)
                self.logger.debug(
                    f"{index_name} embeddings found for {case_mapping} (key: {key})"
                )
            else:
                excluded_cases.append(
                    {
                        "case_mapping": case_mapping,
                        "slide_id": slide_id,
                        "subset": subset,
                        "key": key,
                        "reason": f"missing_{embedding_type}_h5_file",
                    }
                )
                self.logger.debug(
                    f"No {index_name} embeddings for {case_mapping} (key: {key})"
                )

        # Filter dataset to only include cases with embeddings
        original_count = len(self.filtered_data)
        self.filtered_data = available_cases
        final_count = len(self.filtered_data)

        # Log statistics
        self.logger.info(f"{index_name} embeddings coverage validation completed:")
        self.logger.info(f"  Original filtered cases: {original_count}")
        self.logger.info(f"  Cases with {index_name} embeddings: {final_count}")
        self.logger.info(f"  Excluded cases: {len(excluded_cases)}")

        if excluded_cases:
            self.logger.warning(
                f"Excluded {len(excluded_cases)} cases without {index_name} embeddings"
            )

            # Save excluded cases list with details
            excluded_file = (
                self.output_hdf5_path.parent
                / f"{self.output_hdf5_path.stem}_excluded_cases.txt"
            )
            excluded_file.parent.mkdir(parents=True, exist_ok=True)

            with open(excluded_file, "w") as f:
                f.write(
                    f"# Cases excluded from dataset due to missing {index_name} embeddings\n"
                )
                f.write(f"# Generated: {datetime.now().isoformat()}\n")
                f.write(f"# Total excluded: {len(excluded_cases)}\n")
                f.write("# Format: case_mapping\tslide_id\treason\n\n")

                for excluded in excluded_cases:
                    f.write(
                        f"{excluded['case_mapping']}\t{excluded['slide_id']}\t{excluded['reason']}\n"
                    )

            self.logger.info(f"Excluded cases saved to: {excluded_file}")

            # Update filtering statistics
            if hasattr(self, "filtering_stats"):
                self.filtering_stats = getattr(self, "filtering_stats", {})
                self.filtering_stats.update(
                    {
                        f"{embedding_type}_embeddings_filtered_count": final_count,
                        f"{embedding_type}_embeddings_excluded_count": len(
                            excluded_cases
                        ),
                        f"{embedding_type}_embeddings_coverage_ratio": final_count
                        / original_count
                        if original_count > 0
                        else 0,
                    }
                )
        else:
            self.logger.info(f"All filtered cases have {index_name} embeddings")

    def _step_3_cluster_by_organ(self):
        """Step 3: Cluster cases by organ type using selected clustering mode"""
        clustering_mode = self.config.get("clustering_mode", "text")
        self.logger.info(
            f"Clustering cases by organ type using {clustering_mode} mode..."
        )

        if clustering_mode == "text":
            self._cluster_by_text_embeddings()
        elif clustering_mode == "image":
            self._cluster_by_image_features()
        elif clustering_mode == "both":
            self.logger.info("Running both text and image clustering for comparison")
            self._cluster_by_text_embeddings()
            # For "both" mode, we use text clustering as primary but also log image clustering results
            self._validate_with_image_clustering()
        else:
            self.logger.error(
                f"Unknown clustering mode: {clustering_mode}. Using text clustering as fallback."
            )
            self._cluster_by_text_embeddings()

    def _cluster_by_text_embeddings(self):
        """Cluster cases by organ type using text embeddings"""
        self.logger.info("Performing text-based clustering...")

        # Group cases by organ
        organ_cases = defaultdict(list)
        for case in self.filtered_data:
            organ = case.get("organ", "Unknown")
            case_mapping = case.get("case_mapping", "")

            if case_mapping in self.embeddings_cache:
                organ_cases[organ].append(
                    {
                        "case": case,
                        "embedding": np.array(self.embeddings_cache[case_mapping]),
                    }
                )

        self.logger.info(f"Found {len(organ_cases)} organ types")

        self.cluster_stats = {}

        # Cluster each organ group
        for organ, cases_with_embeddings in organ_cases.items():
            self.logger.info(
                f"Processing organ: {organ} ({len(cases_with_embeddings)} cases)"
            )

            if len(cases_with_embeddings) < self.config["n_clusters"]:
                n_clusters = max(1, len(cases_with_embeddings))
                self.logger.warning(
                    f"Only {len(cases_with_embeddings)} cases for {organ}, using {n_clusters} clusters"
                )
            else:
                n_clusters = self.config["n_clusters"]

            # Extract embeddings with dimension validation
            embedding_list = []
            expected_dim = self.config["embedding_dim"]

            for i, item in enumerate(cases_with_embeddings):
                embedding = item["embedding"]
                case_mapping = item["case"].get("case_mapping", f"case_{i}")

                # Validate embedding shape
                if not isinstance(embedding, (list, np.ndarray)):
                    self.logger.error(
                        f"Invalid embedding type for {case_mapping}: {type(embedding)}"
                    )
                    # Create zero embedding as fallback
                    embedding = [0.0] * expected_dim

                embedding = np.array(embedding, dtype=np.float32)

                if embedding.shape != (expected_dim,):
                    self.logger.warning(
                        f"Embedding shape mismatch for {case_mapping}: "
                        f"expected ({expected_dim},), got {embedding.shape}"
                    )

                    if embedding.shape[0] < expected_dim:
                        # Pad with zeros
                        padding = np.zeros(
                            expected_dim - embedding.shape[0], dtype=np.float32
                        )
                        embedding = np.concatenate([embedding, padding])
                        self.logger.debug(f"Padded embedding for {case_mapping}")
                    elif embedding.shape[0] > expected_dim:
                        # Truncate
                        embedding = embedding[:expected_dim]
                        self.logger.debug(f"Truncated embedding for {case_mapping}")
                    else:
                        # Wrong shape entirely, create zero embedding
                        embedding = np.zeros(expected_dim, dtype=np.float32)
                        self.logger.warning(
                            f"Reset embedding to zeros for {case_mapping}"
                        )

                embedding_list.append(embedding)

            # Convert to numpy array - should now have consistent shapes
            embeddings = np.array(embedding_list, dtype=np.float32)
            self.logger.debug(f"Embeddings array shape for {organ}: {embeddings.shape}")

            # Check for zero embeddings (cases without clean_report)
            zero_mask = np.all(embeddings == 0, axis=1)
            non_zero_mask = ~zero_mask

            if np.any(non_zero_mask):
                # Normalize non-zero embeddings
                non_zero_embeddings = embeddings[non_zero_mask]
                embeddings_normalized = normalize(non_zero_embeddings, norm="l2")

                # Perform clustering on non-zero embeddings
                if len(non_zero_embeddings) >= n_clusters:
                    kmeans = KMeans(
                        n_clusters=n_clusters,
                        random_state=self.config["random_seed"],
                        n_init=10,
                    )
                    cluster_labels_non_zero = kmeans.fit_predict(embeddings_normalized)
                else:
                    # If too few non-zero embeddings, assign all to cluster 0
                    cluster_labels_non_zero = np.zeros(
                        len(non_zero_embeddings), dtype=int
                    )

                # Assign cluster labels
                cluster_labels = np.full(
                    len(cases_with_embeddings), -1, dtype=int
                )  # -1 for zero embeddings
                cluster_labels[non_zero_mask] = cluster_labels_non_zero
            else:
                # All embeddings are zero
                cluster_labels = np.full(len(cases_with_embeddings), -1, dtype=int)

            # Assign cluster labels to cases
            for i, item in enumerate(cases_with_embeddings):
                item["case"]["cluster_label"] = int(cluster_labels[i])
                item["case"]["cluster_organ"] = organ

            # Compute statistics
            unique_labels, counts = np.unique(cluster_labels, return_counts=True)
            self.cluster_stats[organ] = {
                "total_cases": len(cases_with_embeddings),
                "n_clusters": n_clusters,
                "cluster_sizes": {
                    int(label): int(count)
                    for label, count in zip(unique_labels, counts)
                },
            }

            self.logger.info(f"Created {n_clusters} clusters for {organ}")
            self.logger.info(f"Cluster sizes: {dict(zip(unique_labels, counts))}")

    def _cluster_by_image_features(self):
        """Cluster cases by organ type using PRISM features"""
        self.logger.info("Performing PRISM-based clustering...")

        # Group cases by organ
        organ_cases = defaultdict(list)
        for case in self.filtered_data:
            organ = case.get("organ", "Unknown")
            case_mapping = case.get("case_mapping", "")

            # Load PRISM features and use first prototype for clustering
            prism_features = self._get_prism_features(case_mapping)
            if prism_features is not None:
                # Use first prototype (shape: 1280) for clustering
                first_prototype = prism_features[0, :]
                organ_cases[organ].append(
                    {
                        "case": case,
                        "embedding": first_prototype,
                    }
                )
            else:
                self.logger.warning(
                    f"No PRISM features found for {case_mapping}, excluding from clustering"
                )

        self.logger.info(f"Found {len(organ_cases)} organ types for PRISM clustering")

        self.cluster_stats = {}

        # Cluster each organ group
        for organ, cases_with_embeddings in organ_cases.items():
            self.logger.info(
                f"Processing organ: {organ} ({len(cases_with_embeddings)} cases with PRISM features)"
            )

            if len(cases_with_embeddings) < self.config["n_clusters"]:
                n_clusters = max(1, len(cases_with_embeddings))
                self.logger.warning(
                    f"Only {len(cases_with_embeddings)} cases for {organ}, using {n_clusters} clusters"
                )
            else:
                n_clusters = self.config["n_clusters"]

            # Extract embeddings with dimension validation
            embedding_list = []
            expected_dim = 1280  # Single feature vector dimension

            for i, item in enumerate(cases_with_embeddings):
                embedding = item["embedding"]
                case_mapping = item["case"].get("case_mapping", f"case_{i}")

                # Validate embedding shape
                if not isinstance(embedding, (list, np.ndarray)):
                    self.logger.error(
                        f"Invalid embedding type for {case_mapping}: {type(embedding)}"
                    )
                    # Create zero embedding as fallback
                    embedding = [0.0] * expected_dim

                embedding = np.array(embedding, dtype=np.float32)

                if embedding.shape != (expected_dim,):
                    self.logger.warning(
                        f"Embedding shape mismatch for {case_mapping}: "
                        f"expected ({expected_dim},), got {embedding.shape}"
                    )

                    if embedding.shape[0] < expected_dim:
                        # Pad with zeros
                        padding = np.zeros(
                            expected_dim - embedding.shape[0], dtype=np.float32
                        )
                        embedding = np.concatenate([embedding, padding])
                        self.logger.debug(f"Padded embedding for {case_mapping}")
                    elif embedding.shape[0] > expected_dim:
                        # Truncate
                        embedding = embedding[:expected_dim]
                        self.logger.debug(f"Truncated embedding for {case_mapping}")
                    else:
                        # Wrong shape entirely, create zero embedding
                        embedding = np.zeros(expected_dim, dtype=np.float32)
                        self.logger.warning(
                            f"Reset embedding to zeros for {case_mapping}"
                        )

                embedding_list.append(embedding)

            # Convert to numpy array - should now have consistent shapes
            embeddings = np.array(embedding_list, dtype=np.float32)
            self.logger.debug(f"Embeddings array shape for {organ}: {embeddings.shape}")

            # Check for zero embeddings (cases without PRISM features)
            zero_mask = np.all(embeddings == 0, axis=1)
            non_zero_mask = ~zero_mask

            if np.any(non_zero_mask):
                # Normalize non-zero embeddings
                non_zero_embeddings = embeddings[non_zero_mask]
                embeddings_normalized = normalize(non_zero_embeddings, norm="l2")

                # Perform clustering on non-zero embeddings
                if len(non_zero_embeddings) >= n_clusters:
                    kmeans = KMeans(
                        n_clusters=n_clusters,
                        random_state=self.config["random_seed"],
                        n_init=10,
                    )
                    cluster_labels_non_zero = kmeans.fit_predict(embeddings_normalized)
                else:
                    # If too few non-zero embeddings, assign all to cluster 0
                    cluster_labels_non_zero = np.zeros(
                        len(non_zero_embeddings), dtype=int
                    )

                # Assign cluster labels
                cluster_labels = np.full(
                    len(cases_with_embeddings), -1, dtype=int
                )  # -1 for zero embeddings
                cluster_labels[non_zero_mask] = cluster_labels_non_zero
            else:
                # All embeddings are zero
                cluster_labels = np.full(len(cases_with_embeddings), -1, dtype=int)

            # Assign cluster labels to cases
            for i, item in enumerate(cases_with_embeddings):
                item["case"]["cluster_label"] = int(cluster_labels[i])
                item["case"]["cluster_organ"] = organ

            # Compute statistics
            unique_labels, counts = np.unique(cluster_labels, return_counts=True)
            self.cluster_stats[organ] = {
                "total_cases": len(cases_with_embeddings),
                "n_clusters": n_clusters,
                "cluster_sizes": {
                    int(label): int(count)
                    for label, count in zip(unique_labels, counts)
                },
            }

            self.logger.info(f"Created {n_clusters} clusters for {organ}")
            self.logger.info(f"Cluster sizes: {dict(zip(unique_labels, counts))}")

    def _validate_with_image_clustering(self):
        """Run image clustering for comparison when in 'both' mode"""
        self.logger.info("Running image clustering for validation/comparison...")

        # Store original cluster stats
        original_cluster_stats = self.cluster_stats.copy()

        # Run image clustering to compare
        temp_filtered_data = [
            case.copy() for case in self.filtered_data
        ]  # Backup case data
        self._cluster_by_image_features()
        image_cluster_stats = self.cluster_stats.copy()

        # Restore original clustering results
        self.cluster_stats = original_cluster_stats
        self.filtered_data = temp_filtered_data

        # Log comparison
        self.logger.info("Clustering comparison (Text vs Image):")
        for organ in original_cluster_stats.keys():
            text_clusters = original_cluster_stats.get(organ, {}).get("n_clusters", 0)
            image_clusters = image_cluster_stats.get(organ, {}).get("n_clusters", 0)
            text_cases = original_cluster_stats.get(organ, {}).get("total_cases", 0)
            image_cases = image_cluster_stats.get(organ, {}).get("total_cases", 0)

            self.logger.info(
                f"{organ}: Text({text_cases} cases, {text_clusters} clusters) vs Image({image_cases} cases, {image_clusters} clusters)"
            )

    def _load_cluster_stats(self):
        """Load cluster statistics when resuming"""
        if self.output_hdf5_path.exists():
            with h5py.File(self.output_hdf5_path, "r") as f:
                if "metadata/cluster_info" in f:
                    cluster_data = f["metadata/cluster_info"][()].decode("utf-8")
                    self.cluster_stats = json.loads(cluster_data)
            self.logger.info(
                f"Loaded cluster statistics for {len(self.cluster_stats)} organs"
            )

    def _step_4_finalize_hdf5(self):
        """Step 4: Generate image features and create final HDF5 structure"""
        self.logger.info("Creating final HDF5 structure...")

        embedding_type = self.config.get("embedding_type", "prism")
        self.logger.info(f"Using {embedding_type.upper()} embeddings for storage")

        np.random.seed(self.config["random_seed"])

        with h5py.File(self.output_hdf5_path, "a") as f:
            # Ensure all groups exist
            embeddings_group = f.require_group("embeddings")
            text_group = f.require_group("text_attributes")
            metadata_group = f.require_group("metadata")

            # Process each case
            processed_count = 0
            skipped_count = 0

            for case in self.filtered_data:
                case_mapping = case.get("case_mapping", "")

                # Use full case_mapping as HDF5 key to preserve uniqueness across subsets
                hdf5_key = self._get_hdf5_key_from_case_mapping(case_mapping)

                # Create slide group in embeddings
                if hdf5_key in embeddings_group:
                    slide_group = embeddings_group[hdf5_key]
                else:
                    slide_group = embeddings_group.create_group(hdf5_key)

                # Track what we're adding
                datasets_added = []

                # Load embeddings based on selected type
                if "features" not in slide_group:
                    if embedding_type == "virchow":
                        # Load Virchow features
                        features = self._get_virchow_features(case_mapping)
                        feature_type = "Virchow"
                    else:
                        # Load PRISM features (default)
                        features = self._get_prism_features(case_mapping)
                        feature_type = "PRISM"

                    if features is not None:
                        # Successfully loaded embeddings
                        slide_group.create_dataset("features", data=features)
                        datasets_added.append(f"features ({feature_type})")
                        self.logger.debug(
                            f"Loaded {feature_type} embeddings for {case_mapping}: {features.shape}"
                        )
                    else:
                        # Fallback to placeholder (random tensors)
                        self.logger.warning(
                            f"No {feature_type} embeddings found for {case_mapping}, using placeholder"
                        )
                        image_features = np.random.randn(
                            *self.config["image_feature_shape"]
                        ).astype(np.float32)
                        slide_group.create_dataset("features", data=image_features)
                        datasets_added.append("features (placeholder)")

                # Add cluster information
                if "cluster_id" not in slide_group:
                    cluster_id = case.get("cluster_label", -1)
                    slide_group.create_dataset(
                        "cluster_id", data=np.array(cluster_id, dtype=np.int32)
                    )
                    datasets_added.append("cluster_id")
                elif "cluster_id" in slide_group:
                    # Update cluster_id if it has changed
                    existing_cluster = slide_group["cluster_id"][()]
                    new_cluster = case.get("cluster_label", -1)
                    if existing_cluster != new_cluster:
                        del slide_group["cluster_id"]
                        slide_group.create_dataset(
                            "cluster_id", data=np.array(new_cluster, dtype=np.int32)
                        )
                        datasets_added.append("cluster_id (updated)")

                if "organ" not in slide_group:
                    organ = case.get("organ", "Unknown")
                    slide_group.create_dataset("organ", data=organ.encode("utf-8"))
                    datasets_added.append("organ")

                # Add text attributes
                if hdf5_key not in text_group:
                    text_data = json.dumps(case, indent=2)
                    text_group.create_dataset(hdf5_key, data=text_data.encode("utf-8"))
                    datasets_added.append("text_attributes")

                if datasets_added:
                    processed_count += 1
                    if processed_count <= 5:  # Log first few for debugging
                        self.logger.debug(
                            f"Processed {hdf5_key}: added {datasets_added}"
                        )
                else:
                    skipped_count += 1

            self.logger.info(
                f"Processed {processed_count} cases, skipped {skipped_count} existing cases"
            )

            # Save metadata
            self._save_metadata_to_hdf5(metadata_group)

        self.logger.info("HDF5 finalization completed")

    def _create_initial_hdf5(self, filtering_stats: Dict):
        """Create initial HDF5 file with filtering results"""
        # Determine file mode based on resume setting and file existence
        if self.resume and self.output_hdf5_path.exists():
            file_mode = "a"  # Append mode for resume
            self.logger.info("Opening existing HDF5 file in append mode")
        else:
            file_mode = "w"  # Write mode for fresh start
            self.logger.info("Creating new HDF5 file")

        with h5py.File(self.output_hdf5_path, file_mode) as f:
            # Create groups if they don't exist
            embeddings_group = f.require_group("embeddings")
            text_group = f.require_group("text_attributes")
            metadata_group = f.require_group("metadata")

            # Save filtered cases to text_attributes
            for case in self.filtered_data:
                case_mapping = case.get("case_mapping", "")
                # Use full case_mapping as HDF5 key to preserve uniqueness across subsets
                hdf5_key = self._get_hdf5_key_from_case_mapping(case_mapping)

                # Only create dataset if it doesn't exist
                if hdf5_key not in text_group:
                    text_data = json.dumps(case, indent=2)
                    text_group.create_dataset(hdf5_key, data=text_data.encode("utf-8"))
                else:
                    self.logger.debug(
                        f"Dataset {hdf5_key} already exists in text_attributes"
                    )

            # Save filtering statistics (including image embeddings stats if available)
            stats_str = json.dumps(filtering_stats, indent=2)
            if "filtering_stats" in metadata_group:
                del metadata_group["filtering_stats"]  # Remove existing
            metadata_group.create_dataset(
                "filtering_stats", data=stats_str.encode("utf-8")
            )

    def _save_metadata_to_hdf5(self, metadata_group):
        """Save all metadata to HDF5"""
        # Cluster info
        if "cluster_info" in metadata_group:
            del metadata_group["cluster_info"]
        cluster_str = json.dumps(self.cluster_stats, indent=2)
        metadata_group.create_dataset("cluster_info", data=cluster_str.encode("utf-8"))

        # Pipeline config
        if "pipeline_config" in metadata_group:
            del metadata_group["pipeline_config"]
        config_str = json.dumps(self.config, indent=2)
        metadata_group.create_dataset(
            "pipeline_config", data=config_str.encode("utf-8")
        )

    def _print_summary(self):
        """Print pipeline execution summary"""
        self.logger.info("Pipeline Summary:")
        self.logger.info("=" * 50)

        embedding_type = self.config.get("embedding_type", "prism")
        self.logger.info(f"Embedding type: {embedding_type.upper()}")

        if self.filtered_data:
            self.logger.info(f"Total cases processed: {len(self.filtered_data)}")

        if self.cluster_stats:
            for organ, stats in self.cluster_stats.items():
                self.logger.info(
                    f"{organ}: {stats['total_cases']} cases, {stats['n_clusters']} clusters"
                )

        self.logger.info(f"Output file: {self.output_hdf5_path}")
        self.logger.info(
            f"File size: {self.output_hdf5_path.stat().st_size / (1024 * 1024):.1f} MB"
        )


def main():
    """Main function for command line usage"""
    import argparse

    parser = argparse.ArgumentParser(description="Run the preprocessing pipeline")
    parser.add_argument("--input", required=True, help="Input HISTAI-Instruct JSON path (converted from JSONL)")
    parser.add_argument(
        "--filter",
        required=True,
        help="Filter cases text file path (e.g., train.txt, val.txt, or test.txt from create_splits.py)",
    )
    parser.add_argument("--output", required=True, help="Output HDF5 file path")
    parser.add_argument("--images", help="Image encodings path (optional, deprecated)")
    parser.add_argument("--no-resume", action="store_true", help="Disable resume mode")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Force fresh start (delete existing output)",
    )
    parser.add_argument(
        "--n-clusters", type=int, default=15, help="Number of clusters per organ"
    )
    parser.add_argument(
        "--prism-base-path",
        required=True,
        help="Base path for PRISM embeddings directory (created using TRIDENT: https://github.com/mahmoodlab/TRIDENT)",
    )
    parser.add_argument(
        "--embedding-type",
        choices=["prism", "virchow"],
        default="prism",
        help="Type of embeddings to store in HDF5 file (default: prism). Clustering always uses PRISM.",
    )

    args = parser.parse_args()

    # Handle fresh start option
    if args.fresh:
        import os

        if os.path.exists(args.output):
            os.remove(args.output)
            print(f"Removed existing output file: {args.output}")

    config = {
        "n_clusters": args.n_clusters,
        "embedding_type": args.embedding_type,
        "prism_embeddings": {
            "base_path": args.prism_base_path,
            "enabled": True,
        },
    }

    print(f"PRISM embeddings base path: {args.prism_base_path}")
    print(f"Embedding type: {args.embedding_type.upper()}")
    print(f"Clustering: Using PRISM image features (k={args.n_clusters})")
    if args.embedding_type == "virchow":
        print(f"Storage: Virchow embeddings (variable N x 2560)")

    pipeline = PreprocessingPipeline(
        input_dataset_path=args.input,
        filter_cases_path=args.filter,
        output_hdf5_path=args.output,
        image_encodings_path=args.images,
        resume=False,  # Resume disabled by default
        config=config,
        force_fresh=args.fresh,
    )

    pipeline.run()


if __name__ == "__main__":
    main()
