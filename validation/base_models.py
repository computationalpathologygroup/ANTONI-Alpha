"""
Unified Model Base Classes for Validation

This module provides shared model infrastructure for validation benchmarking.

Design principles:
- Models handle loading, generation, and perplexity evaluation
- Data loading (get_context) is delegated to datasets
- Models are dataset-agnostic and reusable
"""

import torch
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

from antoni_alpha.models.antoni_pretrained import AntoniAlphaPreTrained
from validation.preprocessor import TilePreprocessor


class BaseModel(ABC):
    """
    Abstract base class for all validation models.

    Handles model lifecycle, generation, and perplexity evaluation.
    Datasets are responsible for providing context (features/images).
    """

    def __init__(self, model_cfg: Dict, device: Optional[str] = None):
        """
        Args:
            model_cfg: Model configuration dictionary with 'path' and 'type' keys
            device: Device to load model on (cuda/cpu). Auto-detects if None.
        """
        self.model_cfg = model_cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None

    @abstractmethod
    def load_model(self):
        """Load the model. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def generate(
        self, context, question: str, history: Optional[List[Dict]] = None
    ) -> str:
        """
        Generate text response to a question.

        Args:
            context: Model-specific context (features for AntoniAlpha, images for MedGemma)
            question: Question text
            history: Optional conversation history

        Returns:
            Generated text response
        """
        pass

    @abstractmethod
    def evaluate_perplexity(
        self, context, question: str, choices: Dict[str, str]
    ) -> Tuple[str, Dict[str, float]]:
        """
        Evaluate multiple-choice question using perplexity-based ranking.

        Args:
            context: Model-specific context
            question: Question text
            choices: Dict mapping choice keys to choice text (e.g., {"A": "text", "Yes": "Yes"})

        Returns:
            Tuple of (predicted_choice_key, normalized_nlls_dict)
        """
        pass

    def cleanup(self):
        """Clean up model resources."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        torch.cuda.empty_cache()

    def __del__(self):
        """Ensure cleanup on deletion."""
        try:
            self.cleanup()
        except:
            pass


class AntoniAlphaModel(BaseModel):
    """
    ANTONI-Alpha model for validation and benchmarking.

    Works with PRISM features (num_patches, 1280) as context.
    """

    def load_model(self):
        """Load ANTONI-Alpha model from HuggingFace."""
        model_id = self.model_cfg["path"]
        revision = self.model_cfg.get("revision", "main")
        print(f"Loading ANTONI-Alpha from {model_id} (revision: {revision})")

        self.model = AntoniAlphaPreTrained.from_pretrained(
            model_id,
            revision=revision,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            token=os.getenv("HF_TOKEN")
        )
        self.model.eval()
        self.processor = self.model.processor

        print(f"ANTONI-Alpha loaded successfully")

    def generate(
        self, context, question: str, history: Optional[List[Dict]] = None
    ) -> str:
        """
        Generate answer using ANTONI-Alpha.

        Args:
            context: Torch tensor of slide latents (batch_size, num_patches, 1280) or numpy array
            question: Question text
            history: Optional conversation history in format [{"user": "...", "assistant": "..."}, ...]

        Returns:
            Generated text response
        """
        if context is None:
            raise ValueError("Context cannot be None for generation")

        # Convert to tensor if needed
        if not isinstance(context, torch.Tensor):
            context = torch.from_numpy(context).float()

        # Add batch dimension if needed
        if context.dim() == 2:
            context = context.unsqueeze(0)

        # Ensure correct dtype for projection layer
        context = context.to(
            device=next(self.model.projection_layer.parameters()).device,
            dtype=torch.bfloat16
        )

        # Build conversation
        conversation_turns = []
        if history:
            for turn in history:
                conversation_turns.append({"role": "user", "content": turn["user"]})
                conversation_turns.append({"role": "assistant", "content": turn["assistant"]})
        
        conversation_turns.append({"role": "user", "content": question})

        # Generate
        with torch.no_grad():
            gen_kwargs = {
                "slide_latents": context,
                "conversations": [conversation_turns],
                "max_new_tokens": 1024,
                "do_sample": False,
            }
            generated_ids = self.model.generate(**gen_kwargs)
            response = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]

        return response

    def evaluate_perplexity(
        self, context, question: str, choices: Dict[str, str]
    ) -> Tuple[str, Dict[str, float]]:
        """
        Evaluate multiple-choice question using perplexity (length-normalized log-likelihood).

        Args:
            context: Torch tensor of slide latents or numpy array
            question: Question text
            choices: Dict mapping choice keys to choice text (e.g., {"A": "text", "Yes": "Yes"})

        Returns:
            Tuple of (predicted_choice_key, normalized_nlls_dict)
        """
        if context is None:
            raise ValueError("Context cannot be None for perplexity evaluation")

        # Convert to tensor if needed
        if not isinstance(context, torch.Tensor):
            context = torch.from_numpy(context).float()

        # Add batch dimension if needed
        if context.dim() == 2:
            context = context.unsqueeze(0)

        # Ensure correct dtype and device
        context = context.to(
            device=next(self.model.projection_layer.parameters()).device,
            dtype=torch.bfloat16
        )

        # Use model's evaluate_multiple_choice method
        with torch.no_grad():
            predicted_choice, normalized_nlls = self.model.evaluate_multiple_choice(
                slide_latents=context, question=question, choices=choices
            )

        return predicted_choice, normalized_nlls


class MedGemmaModel(BaseModel):
    """
    MedGemma model for validation and benchmarking.

    Works with PIL Images (thumbnails) as context.
    Supports optional tiling for high-resolution images.
    """

    def load_model(self):
        """Load MedGemma model."""
        print(f"Loading MedGemma {self.model_cfg['path']}")

        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_cfg["path"],
            torch_dtype=torch.bfloat16,
        )
        self.model = self.model.to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_cfg["path"])

        # Tiling configuration
        self.use_tiling = self.model_cfg.get("use_tiling", True)
        if self.use_tiling:
            self.preprocessor = TilePreprocessor(target_size=892)
            print("Tiling enabled with target_size=892")

        # Tile saving for debugging (optional)
        self.save_tiles = self.model_cfg.get("save_tiles", False)
        if self.save_tiles:
            self.tiles_dir = Path("validation/debug_tiles") / self.model_cfg["name"]
            self.tiles_dir.mkdir(parents=True, exist_ok=True)
            print(f"Tile saving enabled: {self.tiles_dir}")

        print(f"MedGemma loaded successfully on {self.device}")

    def generate(
        self, context, question: str, history: Optional[List[Dict]] = None
    ) -> str:
        """
        Generate answer using MedGemma.

        Args:
            context: PIL Image or list of PIL Images (tiles)
            question: Question text
            history: Optional conversation history in format [{"user": "...", "assistant": "..."}, ...]

        Returns:
            Generated text response
        """
        if context is None:
            raise ValueError("Context cannot be None for generation")

        # Handle context (single image or list of tiles)
        if isinstance(context, Image.Image):
            if self.use_tiling:
                tiles = self.preprocessor.preprocess(context)
            else:
                tiles = [context]
        elif isinstance(context, list):
            tiles = context
        else:
            raise ValueError(
                f"Context must be PIL Image or list of PIL Images, got {type(context)}"
            )

        # Build messages
        messages = []

        # System prompt
        messages.append(
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a helpful assistant for pathologists. You are given fragments of a single whole slide image.",
                    }
                ],
            }
        )

        if history:
            # Subsequent turns are text-only (image context already established)
            for turn in history:
                messages.append(
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": turn["user"]}],
                    }
                )
                messages.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": turn["assistant"]}],
                    }
                )
            messages.append(
                {"role": "user", "content": [{"type": "text", "text": question}]}
            )
        else:
            # First turn: include images and question
            user_content = []
            for tile in tiles:
                user_content.append({"type": "image", "image": tile})
            user_content.append({"type": "text", "text": question})
            messages.append({"role": "user", "content": user_content})

        # Process and generate
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(
                **inputs, max_new_tokens=1024, do_sample=False
            )
            generation = generation[0][input_len:]

        decoded = self.processor.decode(generation, skip_special_tokens=True)
        return decoded

    def _compute_choice_log_likelihood(
        self, context, question: str, choice_text: str
    ) -> Dict[str, float]:
        """
        Compute length-normalized log-likelihood for a specific answer choice.

        Args:
            context: PIL Image or list of PIL Images (tiles)
            question: Question text
            choice_text: Answer choice text

        Returns:
            Dictionary with keys:
                - 'nll': Negative log-likelihood (sum of -log P)
                - 'normalized_nll': NLL divided by number of tokens
                - 'perplexity': exp(normalized_nll)
                - 'num_tokens': Number of tokens in the choice
        """
        # Handle context (single image or list of tiles)
        if isinstance(context, Image.Image):
            if self.use_tiling:
                tiles = self.preprocessor.preprocess(context)
            else:
                tiles = [context]
        elif isinstance(context, list):
            tiles = context
        else:
            raise ValueError(f"Context must be PIL Image or list of PIL Images")

        # Build messages with question + answer
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a helpful assistant for pathologists. You are given fragments of a single whole slide image.",
                    }
                ],
            }
        ]

        # Add images and question + answer
        user_content = []
        for tile in tiles:
            user_content.append({"type": "image", "image": tile})

        # User turn: Question + "Answer:" prompt
        user_content.append({"type": "text", "text": f"{question}\nAnswer:"})
        messages.append({"role": "user", "content": user_content})

        # Model turn: The answer to score (must be structured when images are present)
        messages.append(
            {"role": "model", "content": [{"type": "text", "text": choice_text}]}
        )

        # Apply chat template (without generation prompt for scoring)
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device, dtype=torch.bfloat16)

        # Tokenize just the answer to know which tokens to score
        answer_tokens = (
            self.processor.tokenizer(
                choice_text,
                return_tensors="pt",
                add_special_tokens=False,
            )
            .input_ids[0]
            .to(self.device)
        )

        # Forward pass to get logits
        with torch.inference_mode():
            outputs = self.model(**inputs, return_dict=True)

        # Get logits
        logits = outputs.logits[0]  # [seq_len, vocab_size]

        # Find answer tokens in sequence
        full_ids = inputs["input_ids"][0]
        answer_len = len(answer_tokens)
        start_idx = -1

        for i in range(len(full_ids) - answer_len + 1):
            if torch.equal(full_ids[i : i + answer_len], answer_tokens):
                start_idx = i
                # Keep searching to find the *last* occurrence if multiple
                pass

        if start_idx == -1:
            raise ValueError(
                f"Answer tokens for '{choice_text}' not found in sequence."
            )

        # Get log probabilities for answer tokens (shifted by 1 for causal LM)
        log_probs = torch.nn.functional.log_softmax(
            logits[start_idx - 1 : start_idx + answer_len - 1], dim=-1
        )

        # Score each answer token
        token_log_probs = log_probs[torch.arange(answer_len), answer_tokens]

        # Compute metrics with length normalization
        nll = -token_log_probs.sum().item()
        num_tokens = answer_len
        normalized_nll = nll / num_tokens
        perplexity = torch.exp(torch.tensor(normalized_nll)).item()

        return {
            "nll": nll,
            "normalized_nll": normalized_nll,
            "perplexity": perplexity,
            "num_tokens": num_tokens,
        }

    def evaluate_perplexity(
        self, context, question: str, choices: Dict[str, str]
    ) -> Tuple[str, Dict[str, float]]:
        """
        Evaluate multiple-choice question using length-normalized log-likelihood ranking.

        Args:
            context: PIL Image or list of PIL Images (tiles)
            question: Question text
            choices: Dict mapping choice keys to choice text

        Returns:
            Tuple of (predicted_choice_key, normalized_nlls_dict)
        """
        if context is None:
            raise ValueError("Context cannot be None for perplexity evaluation")

        normalized_nlls = {}

        for choice_key, choice_text in choices.items():
            result = self._compute_choice_log_likelihood(context, question, choice_text)
            # Use normalized NLL for fair comparison across different length options
            normalized_nlls[choice_key] = result["normalized_nll"]

        # Return choice with lowest normalized NLL (equivalently, lowest perplexity)
        best_choice = min(normalized_nlls, key=normalized_nlls.get)
        return best_choice, normalized_nlls


def create_model(model_cfg: Dict, device: Optional[str] = None) -> BaseModel:
    """
    Factory function to create model based on configuration.

    Args:
        model_cfg: Model configuration dictionary with 'type' key
        device: Device to load model on

    Returns:
        Instantiated model wrapper
    """
    model_type = model_cfg.get("type", "antoni_alpha")

    if model_type == "antoni_alpha":
        model = AntoniAlphaModel(model_cfg, device)
    elif model_type == "medgemma":
        model = MedGemmaModel(model_cfg, device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_model()
    return model
