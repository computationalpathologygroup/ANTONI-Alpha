"""
ANTONI-Alpha configuration
"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


class AntoniAlphaConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`AntoniAlpha`]. It is used to instantiate an
    ANTONI-Alpha model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the ANTONI-Alpha architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        llm_model_id (`str`, *optional*, defaults to `"google/medgemma-4b-it"`):
            The model ID of the underlying language model to use.
        llm_hidden_size (`int`, *optional*, defaults to 2048):
            Hidden size of the language model. Required for PreTrainedModel compatibility.
        projector_num_output_tokens (`int`, *optional*, defaults to 256):
            Number of output tokens from the vision projector.
        projector_num_query_heads (`int`, *optional*, defaults to 8):
            Number of query heads in the vision projector attention mechanism.
        projector_num_kv_heads (`int`, *optional*, defaults to 8):
            Number of key-value heads in the vision projector attention mechanism.
        projector_dropout (`float`, *optional*, defaults to 0.0):
            Dropout rate for the vision projector.
        projector_ffn_hidden_dim (`int`, *optional*):
            Hidden dimension of the feed-forward network in the vision projector. If None, no FFN is used.
        lora_alpha (`int`, *optional*, defaults to 16):
            LoRA scaling parameter (alpha).
        lora_dropout (`float`, *optional*, defaults to 0.05):
            LoRA dropout rate.
        lora_r (`int`, *optional*, defaults to 16):
            LoRA rank parameter.
        lora_bias (`str`, *optional*, defaults to `"none"`):
            LoRA bias configuration. Can be "none", "all", or "lora_only".
        lora_target_modules (`str`, *optional*, defaults to `"all-linear"`):
            Target modules for LoRA adaptation.
        lora_task_type (`str`, *optional*, defaults to `"CAUSAL_LM"`):
            Task type for LoRA configuration.
        lora_modules_to_save (`list`, *optional*):
            List of modules to save during LoRA fine-tuning.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Standard deviation of the truncated normal initializer for weight initialization.
            Required for PreTrainedModel compatibility.
        **kwargs:
            Additional keyword arguments passed to the parent configuration.

    Example:

    ```python
    >>> from antoni_alpha import AntoniAlphaConfig, AntoniAlpha

    >>> # Initializing a ANTONI-Alpha configuration
    >>> configuration = AntoniAlphaConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = AntoniAlpha(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "antoni_alpha"

    def __init__(
        self,
        llm_model_id: str = "google/medgemma-4b-it",
        llm_hidden_size: int = 2048,
        projector_num_output_tokens: int = 256,
        projector_num_query_heads: int = 8,
        projector_num_kv_heads: int = 8,
        projector_dropout: float = 0.0,
        projector_ffn_hidden_dim: int | None = None,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_r: int = 16,
        lora_bias: str = "none",
        lora_target_modules: str = "all-linear",
        lora_task_type: str = "CAUSAL_LM",
        lora_modules_to_save: list | None = None,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        self.llm_model_id = llm_model_id
        self.llm_hidden_size = llm_hidden_size
        self.projector_num_output_tokens = projector_num_output_tokens
        self.projector_num_query_heads = projector_num_query_heads
        self.projector_num_kv_heads = projector_num_kv_heads
        self.projector_dropout = projector_dropout
        self.projector_ffn_hidden_dim = projector_ffn_hidden_dim

        # LoRA configuration
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_r = lora_r
        self.lora_bias = lora_bias
        self.lora_target_modules = lora_target_modules
        self.lora_task_type = lora_task_type
        self.lora_modules_to_save = lora_modules_to_save or ["lm_head", "embed_tokens"]

        # PreTrainedModel compatibility
        self.initializer_range = initializer_range

        super().__init__(**kwargs)