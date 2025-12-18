#!/usr/bin/env python3
"""
Convert old nn.Module checkpoint to PreTrainedModel format.

This script converts existing ANTONI-Alpha checkpoints to HuggingFace PreTrainedModel format,
enabling standard HF loading patterns while preserving all trained weights.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch
from huggingface_hub import login
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from antoni_alpha import AntoniAlphaConfig, AntoniAlphaPreTrained

# Load environment variables and authenticate with HuggingFace
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    print("Authenticating with HuggingFace...")
    login(token=hf_token)
    print("✓ Authentication successful\n")
else:
    print("Warning: No HF_TOKEN found in .env file. Proceeding without authentication.\n")


def convert_old_checkpoint_to_hf(
    old_checkpoint_path: Path,
    output_path: Path,
    branch: str = "main"
):
    """
    Convert old nn.Module checkpoint to PreTrainedModel format.

    Args:
        old_checkpoint_path: Path to directory containing pytorch_model.bin and config.json
        output_path: Where to save converted checkpoint
        branch: "main" (finetuned) or "pretrain"

    Returns:
        Tuple of (model, missing_keys, unexpected_keys)
    """
    print(f"\n{'='*80}")
    print(f"Converting checkpoint: {old_checkpoint_path.name}")
    print(f"Branch: {branch}")
    print(f"{'='*80}\n")

    # 1. Load old state dict
    print("Step 1: Loading old state dict...")
    weight_file = old_checkpoint_path / "pytorch_model.bin"
    if not weight_file.exists():
        raise FileNotFoundError(f"Weight file not found: {weight_file}")

    old_state_dict = torch.load(weight_file, map_location="cpu")
    print(f"✓ Loaded {len(old_state_dict)} keys from {weight_file.name}")
    print(f"  Sample keys: {list(old_state_dict.keys())[:3]}")

    # 2. Load config
    print("\nStep 2: Loading configuration...")
    config_path = old_checkpoint_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    config = AntoniAlphaConfig(**config_dict)
    print(f"✓ Loaded configuration")
    print(f"  Model type: {config.model_type}")
    print(f"  LLM: {config.llm_model_id}")
    print(f"  Projector tokens: {config.projector_num_output_tokens}")

    # 3. Create new model
    print("\nStep 3: Creating PreTrainedModel instance...")
    print("  (This will download MedGemma if not cached - may take a few minutes)")

    try:
        new_model = AntoniAlphaPreTrained(config)
        print("✓ Model created successfully")
    except Exception as e:
        print(f"✗ Error creating model: {e}")
        raise

    # 4. Map old keys to new keys (if any renaming needed)
    # Most keys should match directly since we're preserving structure
    print("\nStep 4: Mapping state dict keys...")
    new_state_dict = {}
    for key, value in old_state_dict.items():
        # Direct mapping (no changes expected for most keys)
        # If we had renamed attributes (e.g., projection_layer -> vision_projector),
        # we would map them here
        new_state_dict[key] = value

    print(f"✓ Mapped {len(new_state_dict)} keys")

    # 5. Load state dict
    print("\nStep 5: Loading weights into new model...")
    try:
        missing_keys, unexpected_keys = new_model.load_state_dict(
            new_state_dict,
            strict=False
        )
        print("✓ Weights loaded")
    except Exception as e:
        print(f"✗ Error loading weights: {e}")
        raise

    # 6. Log conversion details
    print("\nConversion Summary:")
    if missing_keys:
        print(f"  Missing keys ({len(missing_keys)}):")
        for key in missing_keys[:10]:  # Show first 10
            print(f"    - {key}")
        if len(missing_keys) > 10:
            print(f"    ... and {len(missing_keys) - 10} more")
    else:
        print("  ✓ No missing keys")

    if unexpected_keys:
        print(f"  Unexpected keys ({len(unexpected_keys)}):")
        for key in unexpected_keys[:10]:  # Show first 10
            print(f"    - {key}")
        if len(unexpected_keys) > 10:
            print(f"    ... and {len(unexpected_keys) - 10} more")
    else:
        print("  ✓ No unexpected keys")

    # 7. Save in HuggingFace format
    print(f"\nStep 6: Saving converted model to {output_path}...")
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        # Use safe_serialization=False to handle shared tensors (weight tying)
        new_model.save_pretrained(output_path, safe_serialization=False)
        print("✓ Model saved using save_pretrained()")
    except Exception as e:
        print(f"✗ Error saving model: {e}")
        raise

    # 8. Copy metadata files
    print("\nStep 7: Copying metadata files...")
    metadata_files = [
        "checkpoint_metadata.json",
        "README.md",
    ]

    for filename in metadata_files:
        src_file = old_checkpoint_path / filename
        if src_file.exists():
            dst_file = output_path / filename
            shutil.copy(src_file, dst_file)
            print(f"✓ Copied {filename}")

    # 9. Verify saved files
    print("\nVerifying saved files...")
    required_files = ["config.json", "pytorch_model.bin"]
    for filename in required_files:
        filepath = output_path / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"✓ {filename} ({size_mb:.1f} MB)")
        else:
            print(f"✗ Missing: {filename}")

    print(f"\n{'='*80}")
    print("✓ Conversion completed successfully!")
    print(f"Converted checkpoint saved to: {output_path}")
    print(f"{'='*80}\n")

    return new_model, missing_keys, unexpected_keys


def main():
    parser = argparse.ArgumentParser(
        description="Convert ANTONI-Alpha checkpoint to HuggingFace PreTrainedModel format"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to input checkpoint directory (contains pytorch_model.bin and config.json)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to output directory for converted checkpoint"
    )
    parser.add_argument(
        "--branch",
        type=str,
        default="main",
        choices=["main", "pretrain"],
        help="Branch name (main for finetuned, pretrain for pretrained)"
    )

    args = parser.parse_args()

    # Validate input
    if not args.input.exists():
        print(f"Error: Input path does not exist: {args.input}")
        sys.exit(1)

    if not (args.input / "pytorch_model.bin").exists():
        print(f"Error: pytorch_model.bin not found in {args.input}")
        sys.exit(1)

    if not (args.input / "config.json").exists():
        print(f"Error: config.json not found in {args.input}")
        sys.exit(1)

    # Run conversion
    try:
        convert_old_checkpoint_to_hf(
            args.input,
            args.output,
            args.branch
        )
    except Exception as e:
        print(f"\nConversion failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
