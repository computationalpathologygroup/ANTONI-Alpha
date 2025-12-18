"""
Configuration loading and validation utilities for ANTONI-Alpha training.

Configuration File Structure
============================

model:
  llm_model_id: str              # Base LLM model (e.g., "google/medgemma-4b-it")
  freeze_llm: bool               # Freeze LLM weights during training
  freeze_projection: bool        # Freeze projection layer weights
  projector:
    num_output_tokens: int       # Number of output tokens from projector
    num_query_heads: int         # Number of query attention heads
    num_kv_heads: int            # Number of key-value attention heads
    dropout: float               # Dropout rate
    ffn_hidden_dim: int          # Hidden dimension for FFN (optional)

lora:                            # LoRA configuration (optional)
  r: int                         # LoRA rank
  alpha: int                     # LoRA alpha scaling
  dropout: float                 # LoRA dropout
  bias: str                      # Bias configuration ("none", "all", "lora_only")
  target_modules: str|list       # Modules to apply LoRA ("all-linear" or list)
  task_type: str                 # Task type (e.g., "CAUSAL_LM")
  modules_to_save: list          # Additional modules to train

data:
  hdf5_path: str                 # Path to training HDF5 dataset
  val_hdf5_path: str             # Path to validation HDF5 dataset (optional)
  text_attributes: list          # List of text attribute keys to sample
  min_turns: int                 # Minimum conversation turns
  num_workers: int               # Number of dataloader workers

training:
  num_epochs: int                # Number of training epochs
  batch_size: int                # Batch size per device
  learning_rate: float           # Learning rate
  weight_decay: float            # Weight decay for optimizer
  gradient_accumulation_steps: int  # Gradient accumulation steps
  max_grad_norm: float           # Maximum gradient norm for clipping
  lr_scheduler_type: str         # LR scheduler ("cosine", "linear", etc.)
  mixed_precision: str           # Mixed precision ("bf16", "fp16", "no")
  validate_every_n_epochs: int   # Run validation every N epochs (optional)

checkpointing:
  output_dir: str                # Base output directory
  save_every_n_epochs: int       # Save checkpoint every N epochs
  keep_last_n_checkpoints: int   # Number of recent checkpoints to keep
  resume_from_checkpoint: str    # Checkpoint name to resume from (optional)
  auto_merge_final: bool         # Auto-merge final FSDP checkpoint

logging:
  logging_steps: int             # Log metrics every N steps
  use_wandb: bool                # Enable Weights & Biases logging
  wandb_project: str             # W&B project name
  wandb_run_name: str            # W&B run name (optional)

training_stage: str              # Training stage ("pretrain" or "finetune")
training_plan: str               # Training plan name for organization

Example:
--------
See config/pretrain.yaml and config/finetune.yaml for complete examples.
"""

import os
import yaml
from typing import Dict, Any
from pathlib import Path


def load_config(config_path: str, base_config_path: str = "config/base.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file with base config support.

    Args:
        config_path: Path to the specific config file
        base_config_path: Path to the base config file with defaults

    Returns:
        Dictionary containing merged configuration
    """
    config = {}

    # Load base config if it exists
    if os.path.exists(base_config_path):
        with open(base_config_path, 'r') as f:
            config = yaml.safe_load(f)

    # Load and merge specific config
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            specific_config = yaml.safe_load(f)
            config = merge_configs(config, specific_config)
    elif config_path:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    return config


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge configuration dictionaries.

    Args:
        base: Base configuration dictionary
        override: Override configuration dictionary

    Returns:
        Merged configuration dictionary
    """
    merged = base.copy()

    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value

    return merged




def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration values and ensure proper types.

    Args:
        config: Configuration dictionary to validate

    Raises:
        ValueError: If configuration is invalid
    """
    required_sections = ['model', 'data', 'training', 'checkpointing', 'logging']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")

    # Validate training stage
    if config.get('training_stage') not in ['pretrain', 'finetune']:
        raise ValueError("training_stage must be either 'pretrain' or 'finetune'")

    # Validate text attributes
    if not config['data'].get('text_attributes'):
        raise ValueError("data.text_attributes must be a non-empty list")

    # Validate and convert numeric values
    try:
        num_epochs = config['training']['num_epochs']
        if isinstance(num_epochs, str):
            num_epochs = int(num_epochs)
            config['training']['num_epochs'] = num_epochs
        if num_epochs <= 0:
            raise ValueError("training.num_epochs must be positive")
    except (ValueError, TypeError) as e:
        raise ValueError(f"training.num_epochs must be a positive integer: {e}")

    try:
        batch_size = config['training']['batch_size']
        if isinstance(batch_size, str):
            batch_size = int(batch_size)
            config['training']['batch_size'] = batch_size
        if batch_size <= 0:
            raise ValueError("training.batch_size must be positive")
    except (ValueError, TypeError) as e:
        raise ValueError(f"training.batch_size must be a positive integer: {e}")

    try:
        learning_rate = config['training']['learning_rate']
        if isinstance(learning_rate, str):
            learning_rate = float(learning_rate)
            config['training']['learning_rate'] = learning_rate
        if learning_rate <= 0:
            raise ValueError("training.learning_rate must be positive")
    except (ValueError, TypeError) as e:
        raise ValueError(f"training.learning_rate must be a positive number: {e}")

    # Validate training plan
    training_plan = config.get('training_plan', 'default')
    if not isinstance(training_plan, str) or not training_plan.strip():
        raise ValueError("training_plan must be a non-empty string")


def get_training_plan_name(config: Dict[str, Any]) -> str:
    """
    Get the training plan name from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Training plan name
    """
    return config.get('training_plan', 'default')


def setup_plan_output_paths(config: Dict[str, Any]) -> Dict[str, Path]:
    """
    Setup output paths for training plan organization.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary with organized paths
    """
    base_output_dir = Path(config['checkpointing']['output_dir'])
    training_plan = get_training_plan_name(config)
    training_stage = config.get('training_stage', 'finetune')

    # Create plan-specific directory structure
    plan_dir = base_output_dir / training_plan
    stage_dir = plan_dir / training_stage

    paths = {
        'plan_dir': plan_dir,
        'stage_dir': stage_dir,
        'checkpoints_dir': stage_dir / 'checkpoints',
        'logs_dir': stage_dir / 'logs',
        'pretrain_dir': plan_dir / 'pretrain',
        'finetune_dir': plan_dir / 'finetune',
    }

    return paths