import torch
import torch.nn as nn
import logging
import os
import json
from pathlib import Path
from peft import LoraConfig, get_peft_model
from transformers import (
    PreTrainedModel,
    AutoModelForImageTextToText,
    AutoProcessor,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import hub, SAFE_WEIGHTS_NAME, SAFE_WEIGHTS_INDEX_NAME, WEIGHTS_NAME, WEIGHTS_INDEX_NAME
from antoni_alpha.models.antoni_projector import AntoniProjector
from antoni_alpha.models.antoni_mixin import AntoniAlphaMixin
from antoni_alpha.configuration_antoni_alpha import AntoniAlphaConfig
from transformers.utils.quantization_config import BitsAndBytesConfig
from safetensors.torch import load_file as safe_load_file

logger = logging.getLogger(__name__)


class AntoniAlphaPreTrained(PreTrainedModel, AntoniAlphaMixin):
    """
    ANTONI-Alpha: Vision-Language Model for Computational Pathology.

    HuggingFace PreTrainedModel-compatible version with full backward compatibility.
    Enables standard HF workflows (from_pretrained, save_pretrained, AutoModel) while
    preserving the custom API for slide latents and conversation handling.

    This class uses AntoniAlphaMixin for all business logic, providing only
    the PreTrainedModel-specific methods and initialization.
    """

    config_class = AntoniAlphaConfig
    base_model_prefix = "antoni_alpha"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer", "AntoniProjector"]
    _skip_keys_device_placement = ["llm"]  # LLM handles its own device placement via BitsAndBytes

    def __init__(self, config: AntoniAlphaConfig):
        super().__init__(config)

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
            low_cpu_mem_usage=False,  # Disable meta device to avoid nested initialization issues
        )

        # Try to use flash attention if available, otherwise fall back to eager
        try:
            import flash_attn
            model_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("Using FlashAttention2")
        except ImportError:
            logger.warning("FlashAttention2 not available, using eager attention")
            model_kwargs["attn_implementation"] = "eager"

        # Device map handling for PreTrainedModel
        # During initialization, we don't set device_map to allow proper loading
        # Users can specify device_map in from_pretrained()

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

        # --- FIX START: Materialize the projector if it was initialized on Meta ---
        # This prevents "Cannot copy out of meta tensor" errors during dispatch
        # because accelerate's init_empty_weights context leaves parameters empty.
        # We force them to CPU so they exist as real tensors.
        first_param = next(self.projection_layer.parameters(), None)
        if first_param is not None and first_param.device.type == "meta":
            # Materialize the tensors (allocate memory)
            self.projection_layer.to_empty(device="cpu")
            # Initialize the weights (since to_empty leaves them with garbage data)
            self._init_weights(self.projection_layer)
        # --- FIX END ---

        self.projection_layer = self.projection_layer.to(torch.bfloat16)

        # Post-initialization for PreTrainedModel
        # Skip weight tying since the LLM (with PEFT) already handles it
        self.post_init()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Custom loader for ANTONI-Alpha PreTrained model.

        Bypasses standard Transformers loading to prevent conflicts between
        accelerate/bitsandbytes and our composite architecture with nested quantized LLM.

        Args:
            pretrained_model_name_or_path: Path to checkpoint directory or HuggingFace model ID
            config: Optional AntoniAlphaConfig instance
            token: HuggingFace Hub authentication token
            device_map: Device placement map (applied after loading)
            **kwargs: Additional arguments passed to config loading

        Returns:
            AntoniAlphaPreTrained model with loaded weights
        """
        config = kwargs.pop("config", None)
        token = kwargs.pop("token", None)
        device_map = kwargs.pop("device_map", None)
        ignore_mismatched_sizes = kwargs.pop("ignore_mismatched_sizes", False)

        # 1. Load Configuration
        if not config:
            # Convert to absolute path if it's a local path (prevents treating it as Hub ID)
            config_path = pretrained_model_name_or_path
            if Path(pretrained_model_name_or_path).exists():
                config_path = str(Path(pretrained_model_name_or_path).resolve())

            config = AntoniAlphaConfig.from_pretrained(config_path, token=token, **kwargs)
            logger.info(f"Loaded config from {config_path}")

        # 2. Initialize Model (without device_map to avoid meta tensor issues)
        logger.info("Initializing ANTONI-Alpha composite model...")
        model = cls(config)

        # 3. Locate Checkpoint Files
        pretrained_path = Path(pretrained_model_name_or_path)

        # Check if local path or hub model
        if pretrained_path.exists() and pretrained_path.is_dir():
            # Local directory
            logger.info(f"Loading from local directory: {pretrained_path}")
            state_dict = cls._load_local_checkpoint(pretrained_path)
        else:
            # HuggingFace Hub model
            logger.info(f"Loading from HuggingFace Hub: {pretrained_model_name_or_path}")
            state_dict = cls._load_hub_checkpoint(pretrained_model_name_or_path, token)

        # 4. Apply Weights
        logger.info("Loading state dict into model...")
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        # 5. Verification
        # We expect 'llm' keys to be missing (loaded internally during __init__)
        # We expect 'projection_layer' keys to be present in checkpoint
        relevant_missing = [k for k in missing_keys if "projection_layer" in k]
        if relevant_missing:
            logger.warning(f"Projection layer weights missing: {relevant_missing}")

        # Filter out expected missing keys (LLM base model)
        llm_missing = [k for k in missing_keys if "llm" in k]
        if llm_missing:
            logger.info(f"LLM keys not loaded from checkpoint (expected): {len(llm_missing)} keys")

        if unexpected_keys:
            logger.info(f"Unexpected keys in checkpoint: {len(unexpected_keys)} keys")

        # 6. Device Placement
        # Note: For quantized models, device placement is already handled by bitsandbytes
        # Only move to device if it's a specific torch device (not "auto")
        if device_map is not None and device_map not in ["auto", "balanced", "balanced_low_0", "sequential"]:
            logger.info(f"Moving model to device: {device_map}")
            model = model.to(device_map)
        elif device_map == "auto":
            logger.info("Using automatic device placement (handled by bitsandbytes)")

        model.eval()
        logger.info("Model loaded successfully!")

        return model

    @staticmethod
    def _load_local_checkpoint(checkpoint_path: Path) -> dict:
        """Load checkpoint from local directory (handles single file and sharded)."""

        # Try safetensors first (sharded)
        if (checkpoint_path / SAFE_WEIGHTS_INDEX_NAME).exists():
            logger.info("Loading sharded safetensors checkpoint...")
            return AntoniAlphaPreTrained._load_sharded_checkpoint(
                checkpoint_path, SAFE_WEIGHTS_INDEX_NAME, use_safetensors=True
            )

        # Try safetensors (single file)
        elif (checkpoint_path / SAFE_WEIGHTS_NAME).exists():
            logger.info("Loading single safetensors checkpoint...")
            return safe_load_file(checkpoint_path / SAFE_WEIGHTS_NAME)

        # Try pytorch bin (sharded)
        elif (checkpoint_path / WEIGHTS_INDEX_NAME).exists():
            logger.info("Loading sharded pytorch checkpoint...")
            return AntoniAlphaPreTrained._load_sharded_checkpoint(
                checkpoint_path, WEIGHTS_INDEX_NAME, use_safetensors=False
            )

        # Try pytorch bin (single file)
        elif (checkpoint_path / WEIGHTS_NAME).exists():
            logger.info("Loading single pytorch checkpoint...")
            checkpoint = torch.load(checkpoint_path / WEIGHTS_NAME, map_location="cpu")
            return checkpoint.get("model_state_dict", checkpoint)

        # Legacy format (pytorch_model.bin without "model." prefix in index name)
        elif (checkpoint_path / "pytorch_model.bin").exists():
            logger.info("Loading legacy pytorch checkpoint...")
            checkpoint = torch.load(checkpoint_path / "pytorch_model.bin", map_location="cpu")
            return checkpoint.get("model_state_dict", checkpoint)

        else:
            raise OSError(f"No checkpoint files found in {checkpoint_path}")

    @staticmethod
    def _load_hub_checkpoint(model_id: str, token: str = None) -> dict:
        """Load checkpoint from HuggingFace Hub."""
        user_agent = {"file_type": "model", "framework": "pytorch", "from_auto_class": False}

        # Try safetensors (sharded)
        try:
            index_file = hub.cached_file(
                model_id,
                filename=SAFE_WEIGHTS_INDEX_NAME,
                token=token,
                user_agent=user_agent
            )
            logger.info("Found sharded safetensors on hub")
            checkpoint_dir = Path(index_file).parent
            return AntoniAlphaPreTrained._load_sharded_checkpoint(
                checkpoint_dir, SAFE_WEIGHTS_INDEX_NAME, use_safetensors=True,
                model_id=model_id, token=token
            )
        except OSError:
            pass

        # Try safetensors (single file)
        try:
            model_file = hub.cached_file(
                model_id,
                filename=SAFE_WEIGHTS_NAME,
                token=token,
                user_agent=user_agent
            )
            logger.info("Found single safetensors on hub")
            return safe_load_file(model_file)
        except OSError:
            pass

        # Try pytorch bin (sharded)
        try:
            index_file = hub.cached_file(
                model_id,
                filename=WEIGHTS_INDEX_NAME,
                token=token,
                user_agent=user_agent
            )
            logger.info("Found sharded pytorch checkpoint on hub")
            checkpoint_dir = Path(index_file).parent
            return AntoniAlphaPreTrained._load_sharded_checkpoint(
                checkpoint_dir, WEIGHTS_INDEX_NAME, use_safetensors=False,
                model_id=model_id, token=token
            )
        except OSError:
            pass

        # Try pytorch bin (single file)
        try:
            model_file = hub.cached_file(
                model_id,
                filename=WEIGHTS_NAME,
                token=token,
                user_agent=user_agent
            )
            logger.info("Found single pytorch checkpoint on hub")
            checkpoint = torch.load(model_file, map_location="cpu")
            return checkpoint.get("model_state_dict", checkpoint)
        except OSError:
            pass

        raise OSError(f"No checkpoint files found for model {model_id}")

    @staticmethod
    def _load_sharded_checkpoint(checkpoint_dir: Path, index_filename: str, use_safetensors: bool,
                                 model_id: str = None, token: str = None) -> dict:
        """Load sharded checkpoint from index file."""
        index_path = checkpoint_dir / index_filename

        with open(index_path, 'r') as f:
            index = json.load(f)

        # Get unique shard files
        shard_files = set(index['weight_map'].values())
        logger.info(f"Loading {len(shard_files)} shards...")

        # Load all shards
        state_dict = {}
        for shard_file in shard_files:
            # If loading from Hub, download the shard file first
            if model_id:
                user_agent = {"file_type": "model", "framework": "pytorch", "from_auto_class": False}
                shard_path = hub.cached_file(
                    model_id,
                    filename=shard_file,
                    token=token,
                    user_agent=user_agent
                )
                shard_path = Path(shard_path)
            else:
                shard_path = checkpoint_dir / shard_file

            if use_safetensors:
                shard = safe_load_file(shard_path)
            else:
                shard = torch.load(shard_path, map_location="cpu")

            state_dict.update(shard)

        logger.info(f"Loaded {len(state_dict)} keys from sharded checkpoint")
        return state_dict

    # PreTrainedModel-specific methods
    def tie_weights(self):
        """Override to prevent weight tying issues with PEFT-wrapped LLM."""
        # The LLM already handles its own weight tying internally
        # Skip tying to avoid conflicts with PEFT adapter
        pass

    def _init_weights(self, module):
        """
        Initialize weights for new modules (required by PreTrainedModel).

        This is called during initialization for modules that need weight initialization.
        The LLM and processor are loaded from pretrained, so we only initialize custom modules.
        """
        std = self.config.initializer_range if hasattr(self.config, 'initializer_range') else 0.02

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    # Required PreTrainedModel methods for embedding handling
    def get_input_embeddings(self):
        """Get input embeddings from the underlying LLM."""
        return self.llm.get_input_embeddings()

    def set_input_embeddings(self, value):
        """Set input embeddings for the underlying LLM."""
        self.llm.set_input_embeddings(value)

    def get_output_embeddings(self):
        """Get output embeddings from the underlying LLM."""
        return self.llm.get_output_embeddings()

    def set_output_embeddings(self, value):
        """Set output embeddings for the underlying LLM."""
        self.llm.set_output_embeddings(value)

    def resize_token_embeddings(self, new_num_tokens: int | None = None):
        """Resize token embeddings of the underlying LLM."""
        return self.llm.resize_token_embeddings(new_num_tokens)

    # Override forward to add return type annotation and delegate to mixin
    def forward(
        self,
        slide_latents: torch.Tensor,
        conversations: list = None,
        text_input: list[str] = None,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass with vision embedding insertion.

        Args:
            slide_latents: Input slide latents [batch_size, num_latents, embedding_dim]
            conversations: Conversation lists in OpenAI format (training mode)
            text_input: Input text strings (inference mode)

        Returns:
            CausalLMOutputWithPast from the LLM
        """
        # Explicitly call mixin's forward to avoid MRO issues
        return AntoniAlphaMixin.forward(self, slide_latents, conversations, text_input)

    # All other business logic methods (freeze_component, create_assistant_only_labels,
    # add_image_tokens_to_text, insert_vision_embeddings, prepare_labels_with_image_masking,
    # generate, compute_choice_log_likelihood, evaluate_multiple_choice,
    # debug_choice_evaluation) are provided by AntoniAlphaMixin
