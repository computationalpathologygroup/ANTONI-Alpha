"""ANTONI-Alpha: Vision-Language Model for Computational Pathology."""

from transformers import AutoConfig, AutoModel

from .configuration_antoni_alpha import AntoniAlphaConfig
from .models.antoni import AntoniAlpha
from .models.antoni_pretrained import AntoniAlphaPreTrained

__version__ = "0.1.0"

# Register with HuggingFace transformers
AutoConfig.register("antoni_alpha", AntoniAlphaConfig)
AutoModel.register(AntoniAlphaConfig, AntoniAlphaPreTrained)

__all__ = [
    "AntoniAlphaConfig",
    "AntoniAlpha",  # Keep for backward compatibility
    "AntoniAlphaPreTrained",  # New PreTrainedModel version
]
