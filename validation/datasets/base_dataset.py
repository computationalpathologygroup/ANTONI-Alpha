"""
Base Dataset Interface for Validation

Defines the abstract interface that all validation datasets must implement.
This enables different benchmarks to work with the same evaluation infrastructure.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from pathlib import Path


class BaseDataset(ABC):
    """
    Abstract base class for validation datasets.

    Each dataset implementation must provide:
    - Question iteration (via __len__ and __getitem__)
    - Context loading for each question
    - Dataset statistics and metadata
    """

    def __init__(self, config: Dict):
        """
        Initialize dataset with configuration.

        Args:
            config: Configuration dictionary (typically from YAML)
        """
        self.config = config

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of questions in the dataset."""
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single question item by index.

        Returns:
            Dictionary with at minimum:
                - 'question_id': Unique identifier for this question
                - 'question': Question text
                - 'ground_truth': Ground truth answer
                - 'context_info': Information needed to load context (e.g., file paths, case IDs)
                - (optional) 'choices': Dict of multiple-choice options for MC questions
                - (optional) 'metadata': Additional metadata (cohort, category, etc.)
        """
        pass

    @abstractmethod
    def load_context(
        self,
        question_item: Dict[str, Any],
        model_type: str
    ) -> Any:
        """
        Load context (features/images) for a question.

        Args:
            question_item: Question dictionary from __getitem__
            model_type: Type of model ("antoni_alpha" or "medgemma")

        Returns:
            Context appropriate for the model type:
                - For ProtoAntoni: torch.Tensor or numpy array of features
                - For MedGemma: PIL Image or list of PIL Images
        """
        pass

    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get dataset statistics.

        Returns:
            Dictionary with dataset statistics (e.g., num_questions, num_unique_slides, etc.)
        """
        pass

    def get_dataset_type(self) -> str:
        """
        Get dataset type identifier.

        Returns:
            String identifier for this dataset type (e.g., "vqa")
        """
        return self.__class__.__name__.replace("Dataset", "").lower()

    def get_unique_contexts(self) -> List[Any]:
        """
        Get list of unique contexts (e.g., slides/cases) in the dataset.

        Useful for grouping questions by context to minimize loading overhead.

        Returns:
            List of unique context identifiers
        """
        # Default implementation: extract unique context_info from all questions
        contexts = set()
        for i in range(len(self)):
            item = self[i]
            # Convert context_info to a hashable type
            context_info = item.get('context_info', {})
            context_key = tuple(sorted(context_info.items()))
            contexts.add(context_key)
        return [dict(ctx) for ctx in contexts]

    def group_questions_by_context(self) -> Dict[Any, List[Dict[str, Any]]]:
        """
        Group questions by their context to enable efficient batch processing.

        Returns:
            Dictionary mapping context identifiers to lists of question items
        """
        grouped = {}
        for i in range(len(self)):
            item = self[i]
            context_info = item.get('context_info', {})
            context_key = tuple(sorted(context_info.items()))

            if context_key not in grouped:
                grouped[context_key] = []
            grouped[context_key].append(item)

        return grouped

    def filter_by_whitelist(self, whitelist: Optional[List[str]]) -> None:
        """
        Filter dataset to only include questions from a whitelist.

        Args:
            whitelist: List of question IDs or context IDs to keep (None = keep all)
        """
        # Optional method - datasets can override if needed
        pass
