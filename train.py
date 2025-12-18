import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from torch.optim import AdamW  # Use PyTorch's native AdamW
from transformers.optimization import (
    get_scheduler,
)  # Import from transformers.optimization
from src.antoni_alpha.models import AntoniAlpha
from src.antoni_alpha.data import WsiHdf5Dataset, collate_fn
from functools import partial
import os
import logging
from pathlib import Path
from datetime import datetime
from tqdm.auto import tqdm
import argparse
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from src.antoni_alpha.utils import (
    load_config,
    validate_config,
    save_checkpoint_with_plan,
    load_checkpoint_with_plan,
    setup_checkpoint_paths,
)
from src.antoni_alpha.utils.checkpoints import (
    find_latest_checkpoint,
    merge_final_checkpoint,
    load_checkpoint_config,
    create_model_from_config,
)
from src.antoni_alpha.configuration_antoni_alpha import AntoniAlphaConfig
from accelerate import FullyShardedDataParallelPlugin, DistributedDataParallelKwargs
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullStateDictConfig,
    FullOptimStateDictConfig,
    ShardingStrategy,
)
from torch.distributed.fsdp.api import (
    ShardedStateDictConfig,
    ShardedOptimStateDictConfig,
)
from torch.distributed.fsdp import StateDictType
import json

load_dotenv()


def setup_logging(output_dir: Path, accelerator: Accelerator) -> logging.Logger:
    """
    Setup logging to both file and console.

    Args:
        output_dir: Directory to save log files
        accelerator: Accelerator instance for distributed training

    Returns:
        Logger instance
    """
    logger = logging.getLogger(__name__)

    if accelerator.is_main_process:
        # Create logs directory
        log_dir = output_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        # Setup file handler with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"training_{timestamp}.log"

        # Configure logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level=logging.INFO,
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )

    return logger


def validate(model, val_dataloader, accelerator, logger):
    """
    Run validation and calculate average loss.

    Args:
        model: The model to validate
        val_dataloader: Validation data loader
        accelerator: Accelerator instance
        logger: Logger instance

    Returns:
        Average validation loss
    """
    model.eval()
    total_val_loss = 0
    num_val_batches = 0

    with torch.no_grad():
        for batch in val_dataloader:
            # Extract batch data
            slide_latents, conversations = batch

            # Forward pass
            outputs = model(slide_latents=slide_latents, conversations=conversations)
            loss = outputs.loss

            # Gather loss across all processes
            avg_loss = accelerator.gather(loss.detach()).mean().item()
            total_val_loss += avg_loss
            num_val_batches += 1

    avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else 0

    model.train()
    return avg_val_loss


def train(config_path: str):
    """
    Train the AntoniAlpha model using Hugging Face Accelerate with FSDP support.

    Args:
        config_path: Path to YAML configuration file
    """
    # --- 1. Configuration Loading ---
    config = load_config(config_path)

    # Validate configuration
    validate_config(config)

    # Extract configuration values
    num_epochs = config["training"]["num_epochs"]
    learning_rate = config["training"]["learning_rate"]
    batch_size = config["training"]["batch_size"]
    gradient_accumulation_steps = config["training"]["gradient_accumulation_steps"]
    save_every_n_epochs = config["checkpointing"]["save_every_n_epochs"]

    # Extract other config values
    training_stage = config.get("training_stage", "finetune")
    training_plan = config.get("training_plan", "default")
    text_attributes = config["data"]["text_attributes"]
    min_turns = config["data"].get("min_turns", 1)
    freeze_llm = config["model"].get("freeze_llm", False)
    freeze_projection = config["model"].get("freeze_projection", False)
    mixed_precision = config["training"]["mixed_precision"]
    use_wandb = config["logging"]["use_wandb"]
    wandb_project = config["logging"]["wandb_project"]
    wandb_run_name = config["logging"]["wandb_run_name"]

    # Setup training plan directory structure
    checkpoint_paths = setup_checkpoint_paths(config)
    output_dir = checkpoint_paths["stage_dir"]  # Use stage-specific output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 2. Initialize Accelerator ---
    # FSDP configuration (optional for multi-GPU training)
    use_fsdp = config["training"].get("use_fsdp", True)

    accelerator_kwargs = {"gradient_accumulation_steps": gradient_accumulation_steps}
    if mixed_precision and mixed_precision != "no":
        accelerator_kwargs["mixed_precision"] = mixed_precision

    if use_wandb:
        accelerator_kwargs["log_with"] = "wandb"

    if use_fsdp:
        fsdp_plugin = FullyShardedDataParallelPlugin(
            fsdp_version=2,
            reshard_after_forward=True,
            cpu_offload=False,
            cpu_ram_efficient_loading=False,
            state_dict_type=StateDictType.FULL_STATE_DICT,
        )
        accelerator_kwargs["fsdp_plugin"] = fsdp_plugin

    accelerator = Accelerator(**accelerator_kwargs)

    # --- 3. Setup Logging ---
    logger = setup_logging(output_dir, accelerator)

    if accelerator.is_main_process:
        logger.info("=" * 50)
        logger.info(f"Training: {training_plan}/{training_stage} | Epochs: {num_epochs} | LR: {learning_rate}")
        logger.info(
            f"Batch size: {batch_size * gradient_accumulation_steps * accelerator.num_processes} "
            f"(per_device={batch_size}, accum={gradient_accumulation_steps}, processes={accelerator.num_processes})"
        )
        logger.info(f"FSDP: {'Enabled' if use_fsdp else 'Disabled (single-GPU mode)'}")
        logger.info(f"Output: {output_dir}")
        logger.info("=" * 50)

    # --- 4. Initialize Weights & Biases (if enabled) ---
    if use_wandb:
        init_kwargs = {"wandb": {"name": wandb_run_name}} if wandb_run_name else {}
        accelerator.init_trackers(
            project_name=wandb_project,
            config={
                "training_plan": training_plan,
                "training_stage": training_stage,
                "freeze_llm": freeze_llm,
                "freeze_projection": freeze_projection,
                "text_attributes": text_attributes,
                "learning_rate": learning_rate,
                "epochs": num_epochs,
                "batch_size": batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "total_batch_size": batch_size
                * gradient_accumulation_steps
                * accelerator.num_processes,
                "mixed_precision": mixed_precision,
            },
            init_kwargs=init_kwargs,
        )

    # --- 5. Data Setup ---
    hdf5_path = config["data"]["hdf5_path"]

    if accelerator.is_main_process:
        logger.info(f"Loading training dataset from {hdf5_path}")

    # Create dataset
    dataset = WsiHdf5Dataset(hdf5_path=hdf5_path)

    # Setup collate function with conversation sampling
    collate_function = partial(
        collate_fn,
        min_turns=min_turns,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_function,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
    )

    # --- 5b. Validation Data Setup ---
    val_hdf5_path = config["data"].get("val_hdf5_path")
    val_dataloader = None

    if val_hdf5_path:
        if accelerator.is_main_process:
            logger.info(f"Loading validation dataset from {val_hdf5_path}")

        val_dataset = WsiHdf5Dataset(hdf5_path=val_hdf5_path)

        val_collate_function = partial(
            collate_fn,
            min_turns=min_turns,
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=val_collate_function,
            num_workers=config["data"]["num_workers"],
            pin_memory=True,
        )

    # --- 6. Model and Optimizer Setup ---
    # Create model config from YAML (standard HF pattern)
    llm_model_id = config["model"].get("llm_model_id", "google/medgemma-4b-it")
    projector_config = config["model"].get("projector", {})

    # Create AntoniAlphaConfig from YAML configuration
    model_config = AntoniAlphaConfig(
        llm_model_id=llm_model_id,
        projector_num_output_tokens=projector_config.get("num_output_tokens", 256),
        projector_num_query_heads=projector_config.get("num_query_heads", 8),
        projector_num_kv_heads=projector_config.get("num_kv_heads", 8),
        projector_dropout=projector_config.get("dropout", 0.0),
        projector_ffn_hidden_dim=projector_config.get("ffn_hidden_dim", None),
    )

    # Create model with config object (standard HF pattern)
    model = AntoniAlpha(config=model_config, device=None)

    # Apply component freezing based on configuration
    if freeze_llm:
        model.freeze_component("llm", freeze=True)

    if freeze_projection:
        model.freeze_component("projection", freeze=True)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if accelerator.is_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_param_count = sum(p.numel() for p in trainable_params)
        logger.info(f"Parameters: {trainable_param_count:,} / {total_params:,} trainable ({trainable_param_count / total_params:.2%})")

    optimizer = AdamW(
        trainable_params,
        lr=learning_rate,
        weight_decay=config["training"]["weight_decay"],
    )

    # --- 7. Prepare for Distributed Training ---
    # Set the FSDP auto-wrap policy using the PEFT utility, as per documentation
    if use_fsdp and getattr(accelerator.state, "fsdp_plugin", None):
        from peft.utils.other import fsdp_auto_wrap_policy

        accelerator.state.fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(model)

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    # Prepare validation dataloader if it exists
    if val_dataloader is not None:
        val_dataloader = accelerator.prepare(val_dataloader)

    # --- 8. Learning Rate Scheduler ---
    # Calculate actual number of optimizer steps (accounting for gradient accumulation)
    num_training_steps = num_epochs * len(dataloader) // gradient_accumulation_steps
    num_warmup_steps = int(0.1 * num_training_steps)  # 10% warmup

    lr_scheduler = get_scheduler(
        name=config["training"]["lr_scheduler_type"],
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # --- 9. Resume from checkpoint if provided ---
    resume_checkpoint_name = config["checkpointing"].get("resume_from_checkpoint")
    start_epoch, global_step, checkpoint_metadata = load_checkpoint_with_plan(
        accelerator, config, resume_checkpoint_name
    )

    if checkpoint_metadata and accelerator.is_main_process:
        source_stage = checkpoint_metadata.get("training_stage")
        if source_stage != training_stage:
            logger.info(
                f"Loaded {source_stage} checkpoint for {training_stage} training"
            )
        logger.info(
            f"Resumed from epoch {checkpoint_metadata.get('epoch', 0)}, step {global_step}"
        )

    # --- 10. Training Loop ---
    if accelerator.is_main_process:
        logger.info("Starting Training Loop")
        progress_bar = tqdm(
            total=num_training_steps, initial=global_step, desc="Training"
        )

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0

        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(model):
                # Extract batch data
                slide_latents, conversations = batch

                # Forward pass
                outputs = model(
                    slide_latents=slide_latents, conversations=conversations
                )
                loss = outputs.loss

                # Backward pass
                accelerator.backward(loss)

                # Gradient clipping and norm calculation
                max_grad_norm = config["training"].get("max_grad_norm")
                if max_grad_norm is not None:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                else:
                    # Calculate norm without clipping
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), float('inf'))

                if grad_norm is not None:
                    grad_norm = grad_norm.item()
                else:
                    grad_norm = 0.0

                optimizer.step()
                optimizer.zero_grad()

            # Gather loss across all processes
            avg_loss = accelerator.gather(loss.detach()).mean().item()
            epoch_loss += avg_loss
            num_batches += 1

            # Update metrics after gradient accumulation completes
            if accelerator.sync_gradients:
                lr_scheduler.step()
                global_step += 1

                # Update progress bar
                if accelerator.is_main_process:
                    progress_bar.update(1)
                    progress_bar.set_postfix(
                        {
                            "loss": f"{avg_loss:.4f}",
                            "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}",
                        }
                    )

                # Log to wandb
                logging_steps = config["logging"]["logging_steps"]
                if use_wandb and global_step % logging_steps == 0:
                    accelerator.log(
                        {
                            "train/loss": avg_loss,
                            "train/grad_norm": grad_norm,
                            "train/learning_rate": lr_scheduler.get_last_lr()[0],
                            "train/epoch": epoch,
                            "train/global_step": global_step,
                        },
                        step=global_step,
                    )

        # End of epoch
        if num_batches == 0:
            if accelerator.is_main_process:
                logger.error(f"ERROR: No batches were processed in epoch {epoch + 1}!")
                logger.error(f"Dataloader length: {len(dataloader)}")
            raise RuntimeError(f"No batches processed in epoch {epoch + 1}. Dataloader may be empty.")

        avg_epoch_loss = epoch_loss / num_batches

        if use_wandb and accelerator.is_main_process:
            accelerator.log(
                {
                    "train/epoch_loss": avg_epoch_loss,
                    "train/epoch": epoch + 1,
                },
                step=global_step,
            )

        # Run validation if configured
        validate_every_n_epochs = config["training"].get(
            "validate_every_n_epochs", None
        )
        if val_dataloader is not None and validate_every_n_epochs is not None:
            if (epoch + 1) % validate_every_n_epochs == 0:
                val_loss = validate(model, val_dataloader, accelerator, logger)

                if accelerator.is_main_process:
                    logger.info(f"Epoch {epoch + 1}/{num_epochs} - Train: {avg_epoch_loss:.4f} | Val: {val_loss:.4f}")

                    if use_wandb:
                        accelerator.log(
                            {
                                "val/loss": val_loss,
                                "val/epoch": epoch + 1,
                            },
                            step=global_step,
                        )
            else:
                if accelerator.is_main_process:
                    logger.info(f"Epoch {epoch + 1}/{num_epochs} - Train: {avg_epoch_loss:.4f}")
        else:
            if accelerator.is_main_process:
                logger.info(f"Epoch {epoch + 1}/{num_epochs} - Train: {avg_epoch_loss:.4f}")

        # Save checkpoint every N epochs
        if (epoch + 1) % save_every_n_epochs == 0:
            save_checkpoint_with_plan(
                accelerator=accelerator,
                epoch=epoch,
                step=global_step,
                loss=avg_epoch_loss,
                config=config,
                keep_last_n=config["checkpointing"]["keep_last_n_checkpoints"],
            )

    # --- 11. Save final model ---
    if accelerator.is_main_process:
        progress_bar.close()
        logger.info("Training completed")

    accelerator.wait_for_everyone()

    # --- 12. Merge final checkpoint ---
    auto_merge = config.get("checkpointing", {}).get("auto_merge_final", True)
    if auto_merge:
        merged_path = merge_final_checkpoint(
            config, checkpoint_paths, logger, accelerator
        )
        if merged_path and accelerator.is_main_process:
            logger.info(f"Final model merged: {merged_path}")
        elif accelerator.is_main_process:
            logger.warning("Failed to merge final checkpoint")
    else:
        if accelerator.is_main_process:
            logger.info("Auto-merge disabled. Use merge_sharded_checkpoint.py to merge manually")

    # --- 13. Cleanup ---
    if use_wandb:
        accelerator.end_training()


def parse_args():
    """Parse command line arguments (config file only)."""
    parser = argparse.ArgumentParser(
        description="Train AntoniAlpha model with Accelerate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  accelerate launch train.py --config config/finetune.yaml
  accelerate launch train.py --config config/pretrain.yaml

All training parameters are specified in the config file.
See config/ directory for example configurations.
        """
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.config)
