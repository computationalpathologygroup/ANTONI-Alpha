# Validation System

Unified validation infrastructure for the ANTONI-Alpha VQA benchmark.

## Quick Start

The main entry point for running validations is `run_benchmark.py`.

```bash
# Run full VQA validation (auto-detects multiple GPUs for DDP)
python -m validation.run_benchmark --benchmark vqa

# Single GPU mode
python -m validation.run_benchmark --benchmark vqa --no-ddp

# Test mode (2 questions)
python -m validation.run_benchmark --benchmark vqa --test --test-questions 2

# Run specific models only
python -m validation.run_benchmark --benchmark vqa --models antoni_alpha_optimized
```

## Data Requirements

The benchmark requires several data files (labels, features, and images). **The system will automatically download and extract these from the Hugging Face Hub if they are missing from the `data/validation/` directory.**

To enable automatic downloads, ensure you have your Hugging Face token set in your environment:
```bash
export HF_TOKEN="your_token_here"
```

The data files include:
1.  **Labeled Data** (`labeled_data_final.json`): Source of truth containing cases, questions, and pre-verified answers.
2.  **Slide Features** (`test_317.h5`): HDF5 file containing Prism embeddings (1280-dim) for ANTONI models.
3.  **Thumbnails** (`thumbnails/`): Directory of histological slide images for MedGemma models.

## Code Structure

The validation system is modular to support multiple benchmarks and model types:

- **`base_models.py`**: Wrappers for different model architectures (ANTONI-Alpha, MedGemma).
- **`datasets/`**: Data loading logic and context handling (extracting features or images).
- **`evaluator.py`**: Core engine handling the evaluation loop, checkpointing, and recovery.
- **`evaluator_ddp.py`**: Multi-GPU support via Distributed Data Parallel.
- **`parsers/`**: Logic for extracting structured answers from model text responses.
- **`metrics/`**: Scoring logic, including hierarchical organ identification scoring.
- **`organ/`**: Taxonomy-based anatomical scoring system.
- **`postprocessing/`**: Advanced analysis scripts (bootstrapping, LLM-based judging).

## Configuration

Edit `validation/config.yaml` to manage:
- **Models**: Repository IDs, branches (revisions), and local paths.
- **Data Paths**: Locations of JSON, H5, and image data.
- **Evaluation**: Mode (`generation` or `perplexity`) and question selection.

## Output

Results are saved to `output/validation_results/`:
- `results.json`: Detailed raw output and extracted answers for every question.
- `metrics.json`: Final accuracy, F1, and hierarchical scores.
- `checkpoint.json`: Used for resuming interrupted runs.
