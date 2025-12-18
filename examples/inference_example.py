#!/usr/bin/env python3
"""
Minimal inference example for ANTONI-Alpha.
Demonstrates multi-turn conversation with a pathology slide.
"""

import torch
from pathlib import Path
from antoni_alpha.models.antoni_pretrained import AntoniAlphaPreTrained

# Load model from HuggingFace
model = AntoniAlphaPreTrained.from_pretrained(
    "SaltySander/ANTONI-Alpha",
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# Load slide features (Prism embeddings: [num_tiles, 1280])
features_path = Path(__file__).parent / "example_slide_features.pt"
slide_features = torch.load(features_path, map_location="cpu")
slide_latents = slide_features.unsqueeze(0)  # Add batch dimension: [1, num_tiles, 1280]

# Move to same device as model
device = next(model.projection_layer.parameters()).device
slide_latents = slide_latents.to(device)

# Multi-turn conversation
conversation = [{"role": "user", "content": "what is this tissue's classification?"}]

# Turn 1
with torch.no_grad():
    output_ids = model.generate(
        slide_latents=slide_latents,
        conversations=[conversation],
        max_new_tokens=200,
        do_sample=False,
    )

response = model.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
print(f"User: {conversation[0]['content']}")
print(f"Assistant: {response}\n")

# Turn 2
conversation.extend(
    [
        {"role": "assistant", "content": response},
        {"role": "user", "content": "Can you describe the cells in more detail?"},
    ]
)

with torch.no_grad():
    output_ids = model.generate(
        slide_latents=slide_latents,
        conversations=[conversation],
        max_new_tokens=200,
        do_sample=False,
    )

response = model.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
print(f"User: {conversation[2]['content']}")
print(f"Assistant: {response}\n")

# Turn 3
conversation.extend(
    [
        {"role": "assistant", "content": response},
        {
            "role": "user",
            "content": "What is happening with the epidermis above the tumor?",
        },
    ]
)

with torch.no_grad():
    output_ids = model.generate(
        slide_latents=slide_latents,
        conversations=[conversation],
        max_new_tokens=200,
        do_sample=False,
    )

response = model.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
print(f"User: {conversation[4]['content']}")
print(f"Assistant: {response}")
