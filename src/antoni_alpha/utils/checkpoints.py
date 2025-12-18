"""
Checkpoint management utilities for ANTONI-Alpha training plans.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

from accelerate import Accelerator


def setup_checkpoint_paths(config: Dict[str, Any]) -> Dict[str, Path]:
    """
    Setup checkpoint paths for training plan organization.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary with checkpoint paths organized by training plan and stage
    """
    from .config import setup_plan_output_paths
    return setup_plan_output_paths(config)


def find_latest_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    """
    Find the latest checkpoint in a directory.

    Args:
        checkpoint_dir: Directory to search for checkpoints

    Returns:
        Path to the latest checkpoint directory, or None if no checkpoints found
    """
    if not checkpoint_dir.exists():
        return None

    checkpoints = sorted(checkpoint_dir.glob("epoch*_step*"))
    if not checkpoints:
        return None

    return checkpoints[-1]


def find_pretrain_checkpoint_for_finetune(config: Dict[str, Any]) -> Optional[Path]:
    """
    Find the latest pretrain checkpoint for finetune stage initialization.

    Args:
        config: Configuration dictionary

    Returns:
        Path to the latest pretrain checkpoint, or None if not found
    """
    paths = setup_checkpoint_paths(config)
    pretrain_checkpoints_dir = paths['pretrain_dir'] / 'checkpoints'

    return find_latest_checkpoint(pretrain_checkpoints_dir)


def save_checkpoint_with_plan(
    accelerator: Accelerator,
    epoch: int,
    step: int,
    loss: float,
    config: Dict[str, Any],
    keep_last_n: int = 3,
    val_loss: Optional[float] = None,
) -> Path:
    """
    Save a checkpoint with training plan organization.

    Args:
        accelerator: The Accelerator instance
        epoch: Current epoch
        step: Current global training step
        loss: Current loss value
        config: Configuration dictionary
        keep_last_n: Number of recent checkpoints to keep
        val_loss: Optional validation loss value

    Returns:
        Path to saved checkpoint directory
    """
    paths = setup_checkpoint_paths(config)
    checkpoint_dir = paths['checkpoints_dir']
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # The state will be saved in a directory with this name
    checkpoint_save_dir = checkpoint_dir / f"epoch{epoch:03d}_step{step:06d}"

    accelerator.wait_for_everyone()

    # Save the sharded state
    accelerator.save_state(checkpoint_save_dir)

    # On the main process, save metadata and model config
    if accelerator.is_main_process:
        logger = logging.getLogger(__name__)

        metadata = {
            "epoch": epoch,
            "step": step,
            "loss": loss,
            "val_loss": val_loss,
            "training_stage": config.get('training_stage', 'finetune'),
            "training_plan": config.get('training_plan', 'default'),
            "timestamp": datetime.now().isoformat(),
        }

        with open(checkpoint_save_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Save the full training configuration
        with open(checkpoint_save_dir / "training_config.json", "w") as f:
            json.dump(config, f, indent=2)

        # Save AntoniAlphaConfig specifically for easy model loading
        try:
            from antoni_alpha.configuration_antoni_alpha import AntoniAlphaConfig

            # Extract model config from training config
            llm_model_id = config["model"].get("llm_model_id", "google/medgemma-4b-it")
            projector_config = config["model"].get("projector", {})

            # Extract LoRA config if present
            lora_config = config.get("lora", {})

            # Create AntoniAlphaConfig
            model_config = AntoniAlphaConfig(
                llm_model_id=llm_model_id,
                projector_num_output_tokens=projector_config.get("num_output_tokens", 256),
                projector_num_query_heads=projector_config.get("num_query_heads", 8),
                projector_num_kv_heads=projector_config.get("num_kv_heads", 8),
                projector_dropout=projector_config.get("dropout", 0.0),
                projector_ffn_hidden_dim=projector_config.get("ffn_hidden_dim", None),
                lora_alpha=lora_config.get("alpha", 16),
                lora_dropout=lora_config.get("dropout", 0.05),
                lora_r=lora_config.get("r", 16),
                lora_bias=lora_config.get("bias", "none"),
                lora_target_modules=lora_config.get("target_modules", "all-linear"),
                lora_task_type=lora_config.get("task_type", "CAUSAL_LM"),
                lora_modules_to_save=lora_config.get("modules_to_save", ["lm_head", "embed_tokens"]),
            )

            # Save as JSON (HF standard format)
            model_config_dict = {
                "model_type": "antoni_alpha",
                "llm_model_id": model_config.llm_model_id,
                "projector_num_output_tokens": model_config.projector_num_output_tokens,
                "projector_num_query_heads": model_config.projector_num_query_heads,
                "projector_num_kv_heads": model_config.projector_num_kv_heads,
                "projector_dropout": model_config.projector_dropout,
                "projector_ffn_hidden_dim": model_config.projector_ffn_hidden_dim,
                "lora_alpha": model_config.lora_alpha,
                "lora_dropout": model_config.lora_dropout,
                "lora_r": model_config.lora_r,
                "lora_bias": model_config.lora_bias,
                "lora_target_modules": model_config.lora_target_modules,
                "lora_task_type": model_config.lora_task_type,
                "lora_modules_to_save": model_config.lora_modules_to_save,
            }

            with open(checkpoint_save_dir / "config.json", "w") as f:
                json.dump(model_config_dict, f, indent=2)

        except Exception as e:
            logger.warning(f"Could not save model config: {e}")

        logger.info(f"Saved checkpoint to {checkpoint_save_dir}")

        # Clean up old checkpoints
        if keep_last_n > 0:
            cleanup_old_checkpoints(checkpoint_dir, keep_last_n, logger)

    return checkpoint_save_dir


def cleanup_old_checkpoints(checkpoint_dir: Path, keep_last_n: int, logger: logging.Logger) -> None:
    """
    Clean up old checkpoints, keeping only the most recent ones.

    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_last_n: Number of recent checkpoints to keep
        logger: Logger instance
    """
    checkpoints = sorted(checkpoint_dir.glob("epoch*_step*"))
    if len(checkpoints) > keep_last_n:
        for old_checkpoint in checkpoints[:-keep_last_n]:
            # Recursively delete old checkpoint directories
            for p in old_checkpoint.rglob("*"):
                if p.is_file():
                    p.unlink()
            # Remove empty directories
            for p in sorted(old_checkpoint.rglob("*"), reverse=True):
                if p.is_dir():
                    try:
                        p.rmdir()
                    except OSError:
                        pass  # Directory not empty, skip
            try:
                old_checkpoint.rmdir()
                logger.info(f"Removed old checkpoint: {old_checkpoint}")
            except OSError:
                logger.warning(f"Could not remove checkpoint directory: {old_checkpoint}")


def load_checkpoint_with_plan(
    accelerator: Accelerator,
    config: Dict[str, Any],
    checkpoint_name: Optional[str] = None,
    model_only: bool = False,
) -> Tuple[int, int, Optional[Dict[str, Any]]]:
    """
    Load a checkpoint with training plan support.

    Args:
        accelerator: The Accelerator instance
        config: Configuration dictionary
        checkpoint_name: Specific checkpoint name (e.g., "epoch002_step001000")
                        If None, will auto-discover based on stage
        model_only: If True, load only model weights (fresh optimizer for stage transitions)

    Returns:
        Tuple of (start_epoch, global_step, metadata)
    """
    logger = logging.getLogger(__name__)
    paths = setup_checkpoint_paths(config)
    training_stage = config.get('training_stage', 'finetune')

    checkpoint_path = None

    if checkpoint_name:
        # Load specific checkpoint from current stage
        checkpoint_path = paths['checkpoints_dir'] / checkpoint_name
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    else:
        # Auto-discovery logic
        if training_stage == 'finetune':
            # For finetune stage, first try to find existing finetune checkpoints
            checkpoint_path = find_latest_checkpoint(paths['checkpoints_dir'])

            if checkpoint_path is None:
                # No finetune checkpoints, look for pretrain checkpoint to initialize from
                pretrain_checkpoint = find_pretrain_checkpoint_for_finetune(config)
                if pretrain_checkpoint is not None:
                    checkpoint_path = pretrain_checkpoint
                    logger.info(f"Initializing finetune from pretrain checkpoint: {pretrain_checkpoint}")
                else:
                    logger.info("No pretrain checkpoint found, starting fresh finetune training")
                    return 0, 0, None
            else:
                logger.info(f"Resuming finetune from checkpoint: {checkpoint_path}")
        else:
            # For pretrain stage, look for existing pretrain checkpoints
            checkpoint_path = find_latest_checkpoint(paths['checkpoints_dir'])
            if checkpoint_path is None:
                logger.info("No pretrain checkpoint found, starting fresh pretrain training")
                return 0, 0, None
            else:
                logger.info(f"Resuming pretrain from checkpoint: {checkpoint_path}")

    # Load the checkpoint
    if checkpoint_path and checkpoint_path.exists():
        # Load metadata first
        metadata_file = checkpoint_path / "metadata.json"
        metadata = None
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            source_stage = metadata.get("training_stage")

            # Determine if this is a stage transition
            is_stage_transition = source_stage != training_stage
            should_load_model_only = model_only or is_stage_transition

            if should_load_model_only:
                logger.info(f"Loading model weights only (fresh optimizer) from {checkpoint_path}")
                if is_stage_transition:
                    logger.info(f"Stage transition detected: {source_stage} -> {training_stage}")

                # Load only model weights using accelerator's model loading
                try:
                    # Get the model from accelerator
                    model = accelerator._models[0] if accelerator._models else None
                    if model is None:
                        raise RuntimeError("No model found in accelerator to load weights into.")

                    is_fsdp = hasattr(accelerator.state, "fsdp_plugin") and accelerator.state.fsdp_plugin is not None

                    if is_fsdp:
                        logger.info("FSDP is enabled. Loading sharded model weights.")
                        from accelerate.utils import load_fsdp_model
                        load_fsdp_model(accelerator.state.fsdp_plugin, accelerator, model, checkpoint_path, 0)
                        accelerator.wait_for_everyone()
                        logger.info("FSDP model weights loaded successfully")
                    else:
                        logger.info("FSDP is not enabled. Loading standard model weights.")
                        import torch
                        
                        try:
                            from safetensors.torch import load_file
                        except ImportError:
                            logger.warning("safetensors not installed. Falling back to torch.load.")
                            load_file = None

                        unwrapped_model = accelerator.unwrap_model(model)

                        model_file = checkpoint_path / "model.safetensors"
                        if not model_file.exists():
                            model_file = checkpoint_path / "pytorch_model.bin"

                        if model_file.exists():
                            if model_file.suffix == ".safetensors" and load_file is not None:
                                state_dict = load_file(model_file, device="cpu")
                            else:
                                state_dict = torch.load(model_file, map_location="cpu")
                            
                            unwrapped_model.load_state_dict(state_dict, strict=False)
                            logger.info(f"Non-FSDP model weights loaded successfully from {model_file}")
                        else:
                            raise FileNotFoundError(
                                f"Could not find model weights file (pytorch_model.bin or model.safetensors) in {checkpoint_path}"
                            )

                except Exception as e:
                    logger.error(f"Failed to load model weights: {e}")
                    if is_stage_transition:
                        # For stage transitions, do NOT fall back to full state loading
                        # because it would load incompatible dataloader/optimizer state
                        logger.error("Cannot load checkpoint for stage transition. Skipping checkpoint loading.")
                        logger.error("Consider training from scratch or fixing the checkpoint format.")
                        return 0, 0, None
                    else:
                        logger.info("Falling back to full state loading...")
                        accelerator.load_state(checkpoint_path)
            else:
                # Load full state (model + optimizer + scheduler)
                logger.info(f"Loading full checkpoint state from {checkpoint_path}")
                accelerator.load_state(checkpoint_path)

            # For stage transitions, reset epoch and step counters
            if is_stage_transition:
                start_epoch = 0  # Start fresh for new stage
                global_step = 0
                logger.info("Reset epoch and step counters for stage transition")
            else:
                start_epoch = metadata.get("epoch", 0) + 1
                global_step = metadata.get("step", 0)

            logger.info(f"Loaded checkpoint from epoch {metadata.get('epoch', 0)}, step {global_step}")
            return start_epoch, global_step, metadata
        else:
            logger.warning(f"Checkpoint metadata not found: {metadata_file}")
            return 0, 0, None
    else:
        return 0, 0, None


def get_checkpoint_info(checkpoint_path: Path) -> Optional[Dict[str, Any]]:
    """
    Get information about a checkpoint.

    Args:
        checkpoint_path: Path to checkpoint directory

    Returns:
        Checkpoint metadata dictionary, or None if not found
    """
    metadata_file = checkpoint_path / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            return json.load(f)
    return None


def list_available_checkpoints(config: Dict[str, Any]) -> Dict[str, list]:
    """
    List all available checkpoints for a training plan.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary with 'pretrain' and 'finetune' checkpoint lists
    """
    paths = setup_checkpoint_paths(config)

    checkpoints = {
        'pretrain': [],
        'finetune': []
    }

    # List pretrain checkpoints
    pretrain_dir = paths['pretrain_dir'] / 'checkpoints'
    if pretrain_dir.exists():
        for checkpoint_dir in sorted(pretrain_dir.glob("epoch*_step*")):
            metadata = get_checkpoint_info(checkpoint_dir)
            checkpoints['pretrain'].append({
                'name': checkpoint_dir.name,
                'path': checkpoint_dir,
                'metadata': metadata
            })

    # List finetune checkpoints
    finetune_dir = paths['finetune_dir'] / 'checkpoints'
    if finetune_dir.exists():
        for checkpoint_dir in sorted(finetune_dir.glob("epoch*_step*")):
            metadata = get_checkpoint_info(checkpoint_dir)
            checkpoints['finetune'].append({
                'name': checkpoint_dir.name,
                'path': checkpoint_dir,
                'metadata': metadata
            })

    return checkpoints


def load_checkpoint_config(checkpoint_path: Path) -> Dict[str, Any]:
    """
    Load configuration from checkpoint directory.

    Args:
        checkpoint_path: Path to checkpoint directory

    Returns:
        Configuration dictionary for model creation
    """
    config_file = checkpoint_path / "config.json"
    if config_file.exists():
        with open(config_file, "r") as f:
            return json.load(f)

    # Fallback to training config
    training_config_file = checkpoint_path / "training_config.json"
    if training_config_file.exists():
        with open(training_config_file, "r") as f:
            training_config = json.load(f)

        # Extract model config from training config
        llm_model_id = training_config["model"].get(
            "llm_model_id", "google/medgemma-4b-it"
        )
        projector_config = training_config["model"].get("projector", {})
        lora_config = training_config.get("lora", {})

        return {
            "model_type": "antoni_alpha",
            "llm_model_id": llm_model_id,
            "projector_num_output_tokens": projector_config.get(
                "num_output_tokens", 256
            ),
            "projector_num_query_heads": projector_config.get("num_query_heads", 8),
            "projector_num_kv_heads": projector_config.get("num_kv_heads", 8),
            "projector_dropout": projector_config.get("dropout", 0.0),
            "projector_ffn_hidden_dim": projector_config.get("ffn_hidden_dim", None),
            "lora_alpha": lora_config.get("alpha", 16),
            "lora_dropout": lora_config.get("dropout", 0.05),
            "lora_r": lora_config.get("r", 16),
            "lora_bias": lora_config.get("bias", "none"),
            "lora_target_modules": lora_config.get("target_modules", "all-linear"),
            "lora_task_type": lora_config.get("task_type", "CAUSAL_LM"),
            "lora_modules_to_save": lora_config.get(
                "modules_to_save", ["lm_head", "embed_tokens"]
            ),
        }

    raise FileNotFoundError(f"No configuration files found in {checkpoint_path}")


def create_model_from_config(config_dict: Dict[str, Any]):
    """
    Create model instance from configuration dictionary.

    Args:
        config_dict: Configuration dictionary containing model parameters

    Returns:
        AntoniAlpha model instance
    """
    from src.antoni_alpha.models import AntoniAlpha
    from src.antoni_alpha.configuration_antoni_alpha import AntoniAlphaConfig

    model_config = AntoniAlphaConfig(
        llm_model_id=config_dict["llm_model_id"],
        projector_num_output_tokens=config_dict["projector_num_output_tokens"],
        projector_num_query_heads=config_dict["projector_num_query_heads"],
        projector_num_kv_heads=config_dict["projector_num_kv_heads"],
        projector_dropout=config_dict["projector_dropout"],
        projector_ffn_hidden_dim=config_dict.get("projector_ffn_hidden_dim"),
        lora_alpha=config_dict.get("lora_alpha", 16),
        lora_dropout=config_dict.get("lora_dropout", 0.05),
        lora_r=config_dict.get("lora_r", 16),
        lora_bias=config_dict.get("lora_bias", "none"),
        lora_target_modules=config_dict.get("lora_target_modules", "all-linear"),
        lora_task_type=config_dict.get("lora_task_type", "CAUSAL_LM"),
        lora_modules_to_save=config_dict.get(
            "lora_modules_to_save", ["lm_head", "embed_tokens"]
        ),
    )

    return AntoniAlpha(config=model_config, device=None)


def merge_final_checkpoint(
    config: Dict[str, Any],
    checkpoint_paths: Dict[str, Path],
    logger: logging.Logger,
    accelerator: Accelerator,
) -> Optional[Path]:
    """
    Merge the final FSDP sharded checkpoint into a single consolidated model.

    Args:
        config: Training configuration
        checkpoint_paths: Checkpoint paths dictionary
        logger: Logger instance
        accelerator: Accelerator instance

    Returns:
        Path to merged checkpoint if successful, None otherwise
    """
    import torch

    if not accelerator.is_main_process:
        return None

    try:
        # Find the latest checkpoint in the current stage
        latest_checkpoint = find_latest_checkpoint(checkpoint_paths["checkpoints_dir"])

        if latest_checkpoint is None:
            logger.warning("No checkpoint found to merge")
            return None

        logger.info(f"Merging final checkpoint: {latest_checkpoint}")

        # Create merged models directory
        merged_dir = checkpoint_paths["stage_dir"] / "merged_models"
        merged_dir.mkdir(parents=True, exist_ok=True)

        # Create output path with timestamp and stage info
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        training_stage = config.get("training_stage", "finetune")
        merged_output = merged_dir / f"{training_stage}_final_{timestamp}"

        # Load checkpoint configuration
        config_dict = load_checkpoint_config(latest_checkpoint)

        # Create model instance
        model = create_model_from_config(config_dict)

        # Load FSDP state dict directly
        fsdp_model_file = latest_checkpoint / "pytorch_model_fsdp.bin"
        if not fsdp_model_file.exists():
            logger.error(f"FSDP model file not found: {fsdp_model_file}")
            return None

        logger.info("Loading FSDP state dict for merging...")
        fsdp_state_dict = torch.load(fsdp_model_file, map_location="cpu")

        # Load the state into the model
        missing_keys, unexpected_keys = model.load_state_dict(
            fsdp_state_dict, strict=False
        )

        if missing_keys:
            logger.warning(f"Missing keys when loading state dict: {missing_keys}")
        if unexpected_keys:
            logger.warning(
                f"Unexpected keys when loading state dict: {unexpected_keys}"
            )

        # Create output directory
        merged_output.mkdir(parents=True, exist_ok=True)

        # Save the full state dict
        model_path = merged_output / "pytorch_model.bin"
        logger.info(f"Saving merged model to: {model_path}")
        torch.save(model.state_dict(), model_path)

        # Save the configuration
        config_path = merged_output / "config.json"
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)

        # Copy metadata
        metadata_src = latest_checkpoint / "metadata.json"
        if metadata_src.exists():
            metadata_dst = merged_output / "checkpoint_metadata.json"
            with open(metadata_src, "r") as f_src, open(metadata_dst, "w") as f_dst:
                metadata = json.load(f_src)
                json.dump(metadata, f_dst, indent=2)

        # Create a README with usage instructions
        readme_path = merged_output / "README.md"
        with open(readme_path, "w") as f:
            f.write(f"""# Merged ANTONI-Alpha Model

This directory contains a merged (consolidated) version of the FSDP sharded checkpoint.

## Model Information
- Training Stage: {training_stage}
- Original Checkpoint: {latest_checkpoint.name}
- Merged On: {timestamp}

## Files
- `pytorch_model.bin`: Full consolidated model weights
- `config.json`: Model configuration
- `checkpoint_metadata.json`: Original checkpoint metadata

## Loading the Model
```python
from src.antoni_alpha.models import AntoniAlpha
from src.antoni_alpha.configuration_antoni_alpha import AntoniAlphaConfig
import torch
import json

# Load config
with open('config.json', 'r') as f:
    config_dict = json.load(f)

# Create model config
model_config = AntoniAlphaConfig(**config_dict)

# Create model
model = AntoniAlpha(config=model_config, device=None)

# Load weights
state_dict = torch.load('pytorch_model.bin', map_location='cpu')
model.load_state_dict(state_dict)
```
""")

        logger.info(f"Successfully merged checkpoint to: {merged_output}")
        logger.info(f"Model size: {model_path.stat().st_size / (1024**3):.2f} GB")

        return merged_output

    except Exception as e:
        logger.error(f"Failed to merge checkpoint: {e}")
        return None
