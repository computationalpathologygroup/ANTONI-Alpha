import torch
import torch.nn as nn
import logging
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
)
from antoni_alpha.models.antoni_projector import AntoniProjector
from antoni_alpha.models.antoni_mixin import AntoniAlphaMixin
from antoni_alpha.configuration_antoni_alpha import AntoniAlphaConfig
from transformers.utils.quantization_config import BitsAndBytesConfig

logger = logging.getLogger(__name__)


class AntoniAlpha(nn.Module, AntoniAlphaMixin):
    """
    LLaVA-style multimodal model combining vision backbone, LLM, and projection layer.

    This class uses AntoniAlphaMixin for all business logic, providing only
    the initialization specific to the nn.Module-based implementation.
    """

    def __init__(self, config: AntoniAlphaConfig, device: str | None = None):
        super().__init__()

        # Store configuration (standard HF pattern)
        self.config = config

        self.vision_backbone_output_dim = 1280  # Based on Prism documentation

        # Set up projector parameters from config
        self.num_vision_embeddings = config.projector_num_output_tokens

        # 2. Large Language Model (MedGemma with LoRA)
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8:
            raise ValueError("A GPU with bfloat16 support is required.")

        model_kwargs = dict(
            torch_dtype=torch.bfloat16,
        )

        # Try to use flash attention if available, otherwise fall back to eager
        try:
            import flash_attn
            model_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("Using FlashAttention2")
        except ImportError:
            logger.warning("FlashAttention2 not available, using eager attention")
            model_kwargs["attn_implementation"] = "eager"

        # For FSDP compatibility: load on CPU when device is None
        if device:
            model_kwargs["device_map"] = device

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_storage=torch.bfloat16,
        )
        model_kwargs["quantization_config"] = quantization_config

        llm = AutoModelForImageTextToText.from_pretrained(config.llm_model_id, **model_kwargs)
        self.processor = AutoProcessor.from_pretrained(config.llm_model_id)
        self.processor.tokenizer.padding_side = "right"

        peft_config = LoraConfig(
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            r=config.lora_r,
            bias=config.lora_bias,
            target_modules=config.lora_target_modules,
            task_type=config.lora_task_type,
            modules_to_save=config.lora_modules_to_save,
        )
        self.llm = get_peft_model(llm, peft_config)
        # Use non-reentrant checkpointing for FSDP compatibility
        # Reentrant mode breaks FSDP semantics and causes memory issues
        self.llm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        self.llm = self.llm.to(torch.bfloat16)

        self.llm_hidden_size = self.llm.config.text_config.hidden_size

        # 3. Projection Layer (AntoniProjector with configurable parameters)
        self.projection_layer = AntoniProjector(
            vision_embedding_dim=self.vision_backbone_output_dim,
            llm_hidden_size=self.llm_hidden_size,
            num_output_tokens=config.projector_num_output_tokens,
            num_query_heads=config.projector_num_query_heads,
            num_kv_heads=config.projector_num_kv_heads,
            dropout=config.projector_dropout,
            ffn_hidden_dim=config.projector_ffn_hidden_dim,
        )
        self.projection_layer = self.projection_layer.to(torch.bfloat16)

        # All business logic methods are provided by AntoniAlphaMixin

    # Explicitly delegate to mixin's forward to avoid MRO issues with nn.Module
    def forward(self, slide_latents, conversations=None, text_input=None):
        """Forward pass - delegates to mixin implementation."""
        return AntoniAlphaMixin.forward(self, slide_latents, conversations, text_input)
