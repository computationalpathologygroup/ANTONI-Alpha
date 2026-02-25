"""Dataset abstractions for validation and benchmarking."""

from .base_dataset import BaseDataset
from .vqa_dataset import VQADataset

__all__ = ["BaseDataset", "VQADataset"]
