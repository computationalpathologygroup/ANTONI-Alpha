import h5py
import json
import torch
import random
from collections import defaultdict
from torch.utils.data import Dataset


class WsiHdf5Dataset(Dataset):
    """
    PyTorch Dataset for loading slide embeddings and text attributes from HDF5.
    Supports multiprocessing via __getstate__/__setstate__.
    """

    def __init__(self, hdf5_path):
        self.hdf5_path = hdf5_path
        self.h5_file = h5py.File(self.hdf5_path, "r")
        self.slide_ids = list(self.h5_file["embeddings"].keys())

    def __len__(self):
        return len(self.slide_ids)

    def __getstate__(self):
        """Prepare state for pickling (exclude h5py file handle)."""
        state = self.__dict__.copy()
        if "h5_file" in state:
            del state["h5_file"]
        return state

    def __setstate__(self, state):
        """Restore state after unpickling and reopen HDF5 file."""
        self.__dict__.update(state)
        self.h5_file = h5py.File(self.hdf5_path, "r")

    def __del__(self):
        """Close HDF5 file handle."""
        if hasattr(self, "h5_file") and self.h5_file:
            self.h5_file.close()

    def __getitem__(self, idx):
        slide_id = self.slide_ids[idx]

        features = self.h5_file["embeddings"][slide_id]["features"][:]
        text_attrs_json = self.h5_file["text_attributes"][slide_id][()]

        if isinstance(text_attrs_json, bytes):
            text_attrs_json = text_attrs_json.decode("utf-8")
        text_attributes = json.loads(text_attrs_json)

        return torch.from_numpy(features), text_attributes


def sample_conversation_turns(conversation: list, min_turns: int = 1) -> list:
    """
    Randomly sample N turns from a conversation while maintaining user/assistant alternation.

    Args:
        conversation: Conversation in OpenAI format with alternating user/assistant messages
        min_turns: Minimum number of turns to include

    Returns:
        Truncated conversation with randomly selected turns
    """
    if not conversation or len(conversation) < 2:
        return conversation

    max_turns = len(conversation) // 2

    if max_turns < min_turns:
        return conversation

    num_turns = random.randint(min_turns, max_turns)
    return conversation[: num_turns * 2]


def sample_text_attribute(text_attributes: dict, min_turns: int = 1) -> list:
    """
    Randomly sample and truncate a conversation from text attributes.

    Args:
        text_attributes: Dictionary of text attributes containing conversations
        min_turns: Minimum number of turns to include

    Returns:
        Chat conversation in OpenAI format, potentially truncated
    """
    if not text_attributes:
        raise ValueError("No text attributes available")

    # Filter for valid conversation attributes (list of dicts with 'role' and 'content')
    valid_keys = []
    for key, value in text_attributes.items():
        if isinstance(value, list) and len(value) > 0:
            if (
                isinstance(value[0], dict)
                and "role" in value[0]
                and "content" in value[0]
            ):
                valid_keys.append(key)

    if not valid_keys:
        raise ValueError(
            f"No valid conversation attributes found. Available keys: {list(text_attributes.keys())}"
        )

    key = random.choice(valid_keys)
    conversation = text_attributes[key]
    conversation = sample_conversation_turns(conversation, min_turns)

    return conversation


def collate_fn(batch, min_turns: int = 1):
    """
    Batch slide embeddings and sample text attributes.

    Args:
        batch: List of (slide_latents, text_attributes) tuples
        min_turns: Minimum number of conversation turns to include
    """
    slide_latents, text_inputs = zip(*batch)
    batched_slide_latents = torch.stack(slide_latents, dim=0)
    batched_conversations = [
        sample_text_attribute(attrs, min_turns=min_turns) for attrs in text_inputs
    ]

    return batched_slide_latents, batched_conversations


class ClusteredDataset(Dataset):
    """
    Dataset for loading cases from clustered JSON with text attribute tracking.
    Works with EffectiveBatchBalancedSampler for balanced sampling without text attribute duplicates.
    """

    def __init__(self, dataset_path: str, hdf5_path: str):
        self.dataset_path = dataset_path
        self.hdf5_path = hdf5_path

        with open(dataset_path, "r") as f:
            self.dataset = json.load(f)

        self.h5_file = h5py.File(self.hdf5_path, "r")

        self.case_mapping_to_idx = {
            case.get("case_mapping", f"case_{idx}"): idx
            for idx, case in enumerate(self.dataset)
        }

        self.current_attr_selection = {}

    def __len__(self):
        return len(self.dataset)

    def __getstate__(self):
        """Prepare state for pickling (exclude h5py file handle)."""
        state = self.__dict__.copy()
        if "h5_file" in state:
            del state["h5_file"]
        return state

    def __setstate__(self, state):
        """Restore state after unpickling and reopen HDF5 file."""
        self.__dict__.update(state)
        self.h5_file = h5py.File(self.hdf5_path, "r")

    def __del__(self):
        """Close HDF5 file handle."""
        if hasattr(self, "h5_file") and self.h5_file:
            self.h5_file.close()

    def get_text_attribute_keys(self) -> list:
        """Get all available text attribute keys from the dataset."""
        if not self.dataset:
            return []

        sample_case = self.dataset[0]
        text_attr_keys = []

        for key, value in sample_case.items():
            if isinstance(value, list) and len(value) > 0:
                if isinstance(value[0], dict) and "role" in value[0]:
                    text_attr_keys.append(key)

        return text_attr_keys

    def __getitem__(self, idx):
        case = self.dataset[idx]
        case_mapping = case.get("case_mapping", f"case_{idx}")

        slide_id = None
        if "filenames" in case and case["filenames"]:
            filename = case["filenames"][0]
            slide_id = filename.split("/")[0]

        if slide_id and slide_id in self.h5_file["embeddings"]:
            features = self.h5_file["embeddings"][slide_id]["features"][:]
            slide_tensor = torch.from_numpy(features)
        else:
            slide_tensor = torch.zeros(1, 768)

        return slide_tensor, case


def balanced_collate_fn(batch, sampler=None, min_turns: int = 1):
    """
    Collate function with EffectiveBatchBalancedSampler support for text attribute
    selection without replacement.

    Args:
        batch: List of (slide_tensor, case_data) tuples
        sampler: EffectiveBatchBalancedSampler instance
        min_turns: Minimum number of conversation turns to include
    """
    slide_tensors, case_data = zip(*batch)
    batched_slide_tensors = torch.stack(slide_tensors, dim=0)
    batched_conversations = []
    selected_attrs = []

    for case in case_data:
        available_attrs = []
        for key, value in case.items():
            if isinstance(value, list) and len(value) > 0:
                if isinstance(value[0], dict) and "role" in value[0]:
                    available_attrs.append(key)

        if available_attrs:
            selected_attr = random.choice(available_attrs)
            conversation = case[selected_attr]
            conversation = sample_conversation_turns(conversation, min_turns)
            batched_conversations.append(conversation)
            selected_attrs.append(selected_attr)
        else:
            batched_conversations.append([])
            selected_attrs.append(None)

    if sampler is not None and hasattr(sampler, "_mark_attributes_used"):
        case_indices = []
        for case in case_data:
            case_mapping = case.get("case_mapping", "")
            if (
                hasattr(sampler, "case_to_index")
                and case_mapping in sampler.case_to_index
            ):
                case_indices.append(sampler.case_to_index[case_mapping])

        if len(case_indices) == len(selected_attrs):
            sampler._mark_attributes_used(case_indices, selected_attrs)

    return batched_slide_tensors, batched_conversations
