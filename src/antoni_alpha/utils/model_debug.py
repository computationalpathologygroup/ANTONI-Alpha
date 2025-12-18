"""
Debug utilities for ANTONI-Alpha model development.

These utilities are helpful during development but not needed for production training.
"""

import torch


def debug_label_masking(
    model,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    final_text: list[str],
    sample_idx: int = 0,
):
    """
    Debug function to visualize what's being masked vs trained on.

    Args:
        model: The AntoniAlpha model instance
        input_ids: The tokenized input
        labels: The label tensor with masking
        final_text: The final text strings
        sample_idx: Which sample in the batch to debug
    """
    if sample_idx >= len(final_text):
        print(f"Sample {sample_idx} not found in batch of size {len(final_text)}")
        return

    sample_input_ids = input_ids[sample_idx]
    sample_labels = labels[sample_idx]
    sample_text = final_text[sample_idx]

    print(f"\n=== LABEL MASKING DEBUG (Sample {sample_idx}) ===")
    print(f"Text length: {len(sample_text)} chars")
    print(f"Token sequence length: {len(sample_input_ids)}")
    print(f"Masked tokens (label = -100): {(sample_labels == -100).sum().item()}")
    print(
        f"Training tokens (label != -100): {(sample_labels != -100).sum().item()}"
    )

    # Show text structure
    print(f"\n--- TEXT STRUCTURE ---")
    lines = sample_text.split("\n")
    for i, line in enumerate(lines):
        marker = (
            "🔴"
            if "user" in line or "<start_of_image>" in line
            else "🟢"
            if "model" in line
            else "⚪"
        )
        print(f"{marker} {line[:150]}{'...' if len(line) > 150 else ''}")

    # Dump ALL token details to file for complete analysis
    debug_file = f"label_debug_sample_{sample_idx}.txt"
    print(f"\n--- DUMPING ALL {len(sample_input_ids)} TOKENS TO {debug_file} ---")

    with open(debug_file, 'w', encoding='utf-8') as f:
        f.write(f"=== LABEL MASKING DEBUG (Sample {sample_idx}) ===\n")
        f.write(f"Text length: {len(sample_text)} chars\n")
        f.write(f"Token sequence length: {len(sample_input_ids)}\n")
        f.write(f"Masked tokens (label = -100): {(sample_labels == -100).sum().item()}\n")
        f.write(f"Training tokens (label != -100): {(sample_labels != -100).sum().item()}\n\n")

        f.write("=== ALL TOKEN DETAILS ===\n")
        for i in range(len(sample_input_ids)):
            token_id = sample_input_ids[i].item()
            label_val = sample_labels[i].item()
            token_str = model.processor.tokenizer.decode(
                [token_id], skip_special_tokens=False
            )

            if label_val == -100:
                status = "MASKED"
            else:
                status = "TRAIN"

            # Highlight special tokens
            if token_id == model.processor.tokenizer.pad_token_id:
                token_str = f"<PAD:{token_id}>"
            elif token_id == getattr(model.processor.tokenizer, "image_token_id", -1):
                token_str = f"<IMG:{token_id}>"

            # Escape newlines for better readability
            token_str_escaped = repr(token_str)

            f.write(f"{i:4d}: {status:6s} | ID:{token_id:6d} | {token_str_escaped}\n")

        f.write(f"\n=== TEXT STRUCTURE ===\n")
        lines = sample_text.split("\n")
        for i, line in enumerate(lines):
            marker = (
                "🔴 USER"
                if "user" in line or "<start_of_image>" in line
                else "🟢 ASSISTANT"
                if "model" in line
                else "⚪ OTHER"
            )
            f.write(f"{marker}: {line}\n")

    print(f"Complete token analysis written to {debug_file}")

    # Show just a summary in console
    print(f"\n--- TOKEN SUMMARY ---")
    assistant_tokens = (sample_labels != -100).sum().item()
    masked_tokens = (sample_labels == -100).sum().item()
    print(f"Assistant tokens (TRAIN): {assistant_tokens}")
    print(f"Masked tokens: {masked_tokens}")
    print(f"Total tokens: {len(sample_input_ids)}")

    print(f"\n--- SUMMARY ---")
    image_token_id = getattr(model.processor.tokenizer, "image_token_id", -1)
    pad_token_id = model.processor.tokenizer.pad_token_id

    image_tokens = (sample_input_ids == image_token_id).sum().item()
    pad_tokens = (sample_input_ids == pad_token_id).sum().item()

    print(f"Image tokens: {image_tokens}")
    print(f"Padding tokens: {pad_tokens}")
    print(f"Regular tokens: {len(sample_input_ids) - image_tokens - pad_tokens}")

    # Check if image tokens are properly masked
    image_mask = sample_input_ids == image_token_id
    image_labels_masked = (sample_labels[image_mask] == -100).all()
    print(f"All image tokens masked: {image_labels_masked}")

    # Check if padding tokens are properly masked
    pad_mask = sample_input_ids == pad_token_id
    if pad_mask.any():
        pad_labels_masked = (sample_labels[pad_mask] == -100).all()
        print(f"All padding tokens masked: {pad_labels_masked}")
