# ANTONI-Alpha

Vision-Language Model for Computational Pathology

## Resources

- **Paper:** [OpenReview](https://openreview.net/forum?id=aGPowreqPi) (under review)
- **Model:** [SaltySander/ANTONI-Alpha](https://huggingface.co/SaltySander/ANTONI-Alpha)
- **Dataset:** [SaltySander/HISTAI-Instruct](https://huggingface.co/datasets/SaltySander/HISTAI-Instruct)
- **Data Generation Framework:** [Polysome](https://github.com/computationalpathologygroup/Polysome)
- **Base Model:** [MedGemma-4B-IT](https://huggingface.co/google/medgemma-4b-it)

## Authors

Computational Pathology Group RadboudUMC

## Model Information

ANTONI-Alpha is a vision-language model for computational pathology. It combines Prism vision embeddings (1280-dim) with MedGemma-2B language model through a learned cross-attention projector, enabling natural language interactions with whole slide images.

**Architecture:**

- Vision encoder: Prism (produces tile-level embeddings)
- Language model: MedGemma-2B (4-bit quantized with LoRA)
- Projector: Cross-attention with 256 learnable query tokens

**Training:**

- Stage 1: Projector alignment (frozen LLM)
- Stage 2: Instruction tuning (LoRA fine-tuning)
- Dataset: HISTAI-Instruct (multilingual, multimodal)

## Installation

```bash
git clone https://github.com/computationalpathologygroup/ANTONI-Alpha.git
cd ANTONI-Alpha
pip install -e .
```

### Optional: Flash Attention 2

For improved performance on compatible hardware, install Flash Attention 2:

```bash
pip install flash-attn==2.8.3 --no-build-isolation
```

The `--no-build-isolation` flag allows the build process to use your installed PyTorch. Flash Attention 2 requires CUDA-capable hardware and will be used automatically if installed.

## How to Use

```python
import torch
from pathlib import Path
from antoni_alpha.models.antoni_pretrained import AntoniAlphaPreTrained

# Load model
model = AntoniAlphaPreTrained.from_pretrained(
    "SaltySander/ANTONI-Alpha",
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# Load slide features (Prism embeddings: [num_tiles, 1280])
slide_features = torch.load("slide_features.pt")
slide_latents = slide_features.unsqueeze(0)  # Add batch dimension
slide_latents = slide_latents.to(next(model.projection_layer.parameters()).device)

# Run inference
conversation = [{"role": "user", "content": "What tissue is this?"}]

with torch.no_grad():
    output_ids = model.generate(
        slide_latents=slide_latents,
        conversations=[conversation],
        max_new_tokens=200,
        do_sample=False,
    )

response = model.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
print(response)
```

See `examples/inference_example.py` for a complete multi-turn conversation example.

## Input/Output Structure

**Input:**

- `slide_latents`: Tensor of shape `[batch_size, num_tiles, 1280]` (Prism embeddings)
- `conversations`: List of conversation lists in OpenAI format

**Output:**

- Generated text response from the language model

## Training

```bash
# Configure training
python train.py --config config/finetune.yaml
```

Training configurations available in `config/` directory.

## License

This model is released under the [Health AI Developer Foundations License](https://developers.google.com/health-ai-developer-foundations/terms).

## Citation

```bibtex
@inproceedings{moonemans2025open,
  title={Democratizing Pathology Co-Pilots: An Open Pipeline and Dataset for Whole-Slide Vision-Language Modeling},
  author={Sander Moonemans and Sebastiaan Ram and Fr{\'e}d{\'e}rique Meeuwsen and Carlijn Lems and Jeroen van der Laak and Geert Litjens and Francesco Ciompi},
  booktitle={Submitted to Medical Imaging with Deep Learning},
  year={2025},
  url={https://openreview.net/forum?id=aGPowreqPi},
  note={under review}
}
```
