# ANTONI-Alpha Data Preprocessing Pipeline

Complete pipeline for preprocessing HISTAI data for ANTONI-Alpha training from raw whole-slide images to final HDF5 datasets.

## Overview

This pipeline transforms HISTAI whole-slide images and instruction data into structured training datasets through 3 main phases:

1. **Download HISTAI Data** - Obtain WSI images and HISTAI-Instruct QA pairs
2. **Generate Slide Embeddings** - Process WSIs through Trident to extract PRISM features
3. **Create Training Datasets** - Generate train/val/test HDF5 files with stratified splits

```
HISTAI WSI Downloads (TIFF) ──────┐
                                   ├──> PRISM Embeddings (H5) ─┐
HISTAI-Instruct (JSON) ───────────┘                            ├──> HDF5 Datasets
                                                                │    (train/val/test)
                                                                └──> Stratified Subsets
```

## Prerequisites

- **Python 3.10+**
- **wholeslidedata**: `pip install wholeslidedata`
- **Trident framework**: Docker recommended (<https://github.com/mahmoodlab/TRIDENT>)
- **Python dependencies**: `pip install numpy h5py scikit-learn`

## Phase 1: Download HISTAI Data

### 1.1 Download HISTAI Whole-Slide Images

Download HISTAI datasets from HuggingFace: <https://huggingface.co/collections/histai/histai-whole-slide-images-dataset>

**Available datasets:**

- **HISTAI-metadata** - Main metadata file (read this first)
- **HISTAI-mixed** - Mixed dataset
- **HISTAI-breast** - Breast tissue
- **HISTAI-skin-b1** - Skin batch 1
- **HISTAI-skin-b2** - Skin batch 2
- **HISTAI-thorax** - Thorax tissue
- **HISTAI-hematologic** - Hematologic tissue
- **HISTAI-gastrointestinal** - Gastrointestinal tissue
- **HISTAI-colorectal-b1** - Colorectal batch 1
- **HISTAI-colorectal-b2** - Colorectal batch 2

**Expected structure after download**:

```
histai_downloads/
├── HISTAI-breast/
│   ├── case_0000/
│   │   ├── slide_H&E_0.tiff
│   │   └── slide_H&E_1_x40.tiff
│   ├── case_0001/
│   │   └── slide_H&E_0.tiff
│   └── ...
├── HISTAI-skin-b1/
│   └── case_*/slide_*.tiff
├── HISTAI-colorectal-b1/
│   └── case_*/slide_*.tiff
└── ... (other datasets)
```

### 1.2 Download HISTAI-Instruct Dataset

Download from HuggingFace: <https://huggingface.co/datasets/SaltySander/HISTAI-Instruct>

The dataset comes in JSONL format and needs to be converted to JSON array:

```bash
# Download the dataset
wget https://huggingface.co/datasets/SaltySander/HISTAI-Instruct/resolve/main/histai-instruct.jsonl

# Convert JSONL to JSON array (one-time conversion)
python -c "
import json
with open('histai-instruct.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]
with open('data/histai-instruct.json', 'w') as f:
    json.dump(data, f, indent=2)
"
```

**Expected JSON format**:

```json
[
  {
    "case_mapping": "histai/HISTAI-breast/case_0001",
    "organ": "breast",
    "clean_report": [...],
    "instruction": "...",
    "output": "..."
  },
  ...
]
```

## Phase 2: Generate Slide Embeddings

This phase processes raw HISTAI slides to generate PRISM embeddings using the Trident framework. **You must run these steps for EACH HISTAI dataset separately.**

### 2.1 Add Spacing Metadata

HISTAI slides lack spacing metadata. This script adds μm/pixel information based on magnification:

- Regular slides: 0.5 μm/px (20x magnification)
- Slides with 'x40' in filename: 0.25 μm/px (40x magnification)

```bash
./tiling_embedding/add_histai_spacing.sh \
  -p /path/to/HISTAI-breast \
  -o ./processed_histai_breast
```

**Options:**

- `-p, --parent-folder PATH`: Input folder with case_* subdirectories
- `-o, --output-base PATH`: Output folder (default: ./processed_histai)
- `-f, --file-list PATH`: Process files from list instead of scanning folder

**Output structure**:

```
processed_histai_breast/
├── case_0000/
│   └── processed/
│       ├── slide_H&E_0.tiff        (0.5 μm/px for 20x)
│       └── slide_H&E_1_x40.tiff    (0.25 μm/px for 40x)
└── case_0001/
    └── processed/
        └── slide_H&E_0.tiff
```

### 2.2 Rename Files with Case Numbers

Prevents filename conflicts when multiple cases have identically-named slides:

```bash
./tiling_embedding/rename_to_case_nr.sh ./processed_histai_breast
```

Renames: `slide_H&E_0.tiff` → `0000_slide_H&E_0.tiff`

**Critical**: This step must be completed before embedding.

### 2.3 Prepare CSV for Embedding

Creates `slide_batch.csv` for Trident embedding framework:

```bash
./tiling_embedding/embedding/preprocessing/make_slides_csv_trident.py \
    ./processed_histai_breast \
    ./embedding_input_breast \
    --batches 1
```

**Arguments:**

- `input_dir`: Directory with processed cases
- `output_dir`: Output directory for CSV files
- `--batches N`: Split into N batches (default: 1)

**Output**: `embedding_input_breast/slide_batch.csv`

### 2.4 Generate PRISM Embeddings

Run Trident embedding framework to generate PRISM features:

```bash
docker run -v $(pwd):/data trident:latest python3 /opt/run.py \
    --task all \
    --wsi_dir /data/processed_histai_breast \
    --job_dir /data/embedding_output_breast \
    --custom_list_of_wsis /data/embedding_input_breast/slide_batch.csv \
    --slide_encoder prism \
    --patch_size 224 \
    --mag 20 \
    --seg_batch_size 512 \
    --feat_batch_size 512 \
    --segmenter hest
```

**Key parameters:**

- `--task all`: Run full pipeline (segmentation → coordinates → features → slide features)
- `--slide_encoder prism`: Use Prism for slide-level features
- `--segmenter hest`: Use HEST for tissue segmentation (built-in)
- `--mag 20`: Use 20x magnification
- `--patch_size 224`: Extract 224x224 patches

**Output structure**:

```
embedding_output_breast/
├── _logs_segmentation.txt
└── 20x_224px_0px_overlap/
    ├── _logs_coords.txt
    ├── _logs_feats_virchow.txt
    ├── _logs_slide_features_prism.txt
    ├── coords/
    ├── feats_virchow/
    └── slide_features_prism/
        ├── 0001_slide_H&E_0.h5
        ├── 0042_slide_H&E_0.h5
        └── ...
```

**Each HDF5 file contains**: `features` dataset with shape (513, 1280) - 513 PRISM prototypes × 1280 dimensions

**IMPORTANT**: Repeat steps 2.1-2.4 for EACH HISTAI dataset (breast, skin, etc.) with different output directories.

### 2.5 Validate Embeddings

Check which slides completed successfully:

```bash
./tiling_embedding/embedding/check_missing_embeddings_compared_to_dataset.py \
    ./embedding_input_breast/slide_batch.csv \
    ./embedding_output_breast
```

The script checks 4 processing stages:

1. Segmentation (tissue detection)
2. Coords (coordinate extraction)
3. Virchow (patch-level features)
4. Prism (slide-level features)

### 2.6 Organize Multi-Dataset Outputs

After running embedding generation for ALL datasets, organize them into the structure expected by the training data pipeline:

**Expected final structure**:

```
prism_embeddings/
├── HISTAI-breast/
│   └── 20x_224px_0px_overlap/
│       └── slide_features_prism/
│           ├── 0001_slide_H&E_0.h5
│           └── ...
├── HISTAI-skin-b1/
│   └── 20x_224px_0px_overlap/
│       └── slide_features_prism/
│           └── ...
├── HISTAI-skin-b2/
│   └── ...
├── HISTAI-colorectal-b1/
│   └── ...
├── HISTAI-colorectal-b2/
│   └── ...
├── HISTAI-thorax/
│   └── ...
├── HISTAI-hematologic/
│   └── ...
└── HISTAI-gastrointestinal/
    └── ...
```

**Organization script**:

```bash
# Create base directory
mkdir -p prism_embeddings

# Organize each dataset
# Assuming your embedding outputs are in:
# - ./embedding_output_breast/20x_224px_0px_overlap/slide_features_prism/
# - ./embedding_output_skin-b1/20x_224px_0px_overlap/slide_features_prism/
# - ./embedding_output_colorectal-b1/20x_224px_0px_overlap/slide_features_prism/
# etc.

# List of HISTAI datasets (excluding metadata and mixed)
for dataset in breast skin-b1 skin-b2 thorax hematologic gastrointestinal colorectal-b1 colorectal-b2; do
    echo "Organizing HISTAI-${dataset}..."
    mkdir -p "prism_embeddings/HISTAI-${dataset}"

    if [ -d "./embedding_output_${dataset}/20x_224px_0px_overlap" ]; then
        cp -r "./embedding_output_${dataset}/20x_224px_0px_overlap" \
              "prism_embeddings/HISTAI-${dataset}/"
        echo "  ✓ Copied embeddings for HISTAI-${dataset}"
    else
        echo "  ⚠ Warning: No embeddings found for HISTAI-${dataset}"
    fi
done

echo "Done! All embeddings organized in prism_embeddings/"
```

## Phase 3: Create Training Datasets

This phase combines HISTAI-Instruct data with PRISM embeddings to create structured HDF5 training datasets.

### 3.1 Create Train/Val/Test Splits

Split the dataset into train/validation/test sets using cluster-based stratified sampling:

```bash
python -m auxilary_code.training_data_preprocessing.create_splits \
  --input data/histai-instruct.json \
  --output-dir data/splits/ \
  --prism-base-path prism_embeddings/ \
  --n-clusters 32 \
  --seed 42
```

**What it does**:

- Groups cases by HISTAI subset (breast, skin, etc.)
- Loads PRISM first prototypes (1280-d) for each case
- Performs K-means clustering (k=32) within each subset if ≥100 samples
- Stratified 80-10-10 split by cluster to ensure representative sampling
- Writes case mappings to text files

**Parameters**:

- `--input`: Path to histai-instruct.json
- `--output-dir`: Directory for output split files (default: data/splits/)
- `--prism-base-path`: Base directory containing PRISM embeddings
- `--n-clusters`: Number of clusters per subset for stratification (default: 32)
- `--seed`: Random seed for reproducibility (default: 42)
- `--exclude-from-train`, `--exclude-from-val`, `--exclude-from-test`: Optional exclusion lists

**Outputs**:

- `data/splits/train.txt` (~80% of cases)
- `data/splits/val.txt` (~10% of cases)
- `data/splits/test.txt` (~10% of cases)

Each file contains one case mapping per line: `histai/HISTAI-breast/case_0001`

### 3.2 Generate HDF5 Datasets

Create HDF5 files for train, validation, and test splits. Run this command **three times** (once for each split):

**3.2a: Training HDF5**

```bash
python -m auxilary_code.training_data_preprocessing.preprocessing_pipeline \
  --input data/histai-instruct.json \
  --filter data/splits/train.txt \
  --output data/train.h5 \
  --prism-base-path prism_embeddings/ \
  --n-clusters 15 \
  --embedding-type prism
```

**3.2b: Validation HDF5**

```bash
python -m auxilary_code.training_data_preprocessing.preprocessing_pipeline \
  --input data/histai-instruct.json \
  --filter data/splits/val.txt \
  --output data/val.h5 \
  --prism-base-path prism_embeddings/ \
  --n-clusters 15
```

**3.2c: Test HDF5**

```bash
python -m auxilary_code.training_data_preprocessing.preprocessing_pipeline \
  --input data/histai-instruct.json \
  --filter data/splits/test.txt \
  --output data/test.h5 \
  --prism-base-path prism_embeddings/ \
  --n-clusters 15
```

**What it does**:

- Filters HISTAI-Instruct to cases in the split file
- Validates that PRISM embeddings exist for each case
- Performs K-means clustering by organ type (k=15) using PRISM features
- Creates structured HDF5 with embeddings, metadata, and cluster assignments
- Tracks excluded cases (no embeddings) in separate file

**Parameters**:

- `--input`: Path to histai-instruct.json
- `--filter`: Path to split file (train.txt, val.txt, or test.txt)
- `--output`: Output HDF5 file path
- `--prism-base-path`: Base directory containing PRISM embeddings
- `--n-clusters`: Number of clusters per organ for grouping (default: 15)
- `--embedding-type`: Embedding type to store (prism or virchow, default: prism)

**HDF5 Structure**:

```
train.h5/
├── embeddings/
│   ├── HISTAI-breast__case_0001/
│   │   ├── features          # (513, 1280) PRISM array
│   │   ├── cluster_id        # scalar int32
│   │   └── organ             # string
│   └── ...
├── text_attributes/
│   ├── HISTAI-breast__case_0001  # JSON string with full case data
│   └── ...
└── metadata/
    ├── cluster_info          # JSON: cluster statistics by organ
    ├── filtering_stats       # JSON: cases processed/excluded
    ├── pipeline_config       # JSON: run parameters
    └── processing_log        # JSON: execution timeline
```

**Note**: Case keys use double underscore (`HISTAI-breast__case_0001`) for HDF5 path safety.

### 3.3 Create Stratified Subsets (Optional)

Create smaller training subsets while maintaining subset proportions:

```bash
python -m auxilary_code.training_data_preprocessing.create_stratified_subsets \
  --train-file data/splits/train.txt \
  --output-dir data/splits/ \
  --seed 42
```

**Outputs**:

- `data/splits/train_2k.txt` - 2000 cases (proportional sampling)
- `data/splits/train_9k.txt` - 9000 cases (proportional sampling)

**To create HDF5 for subsets**, repeat step 3.2 with the subset files:

```bash
python -m auxilary_code.training_data_preprocessing.preprocessing_pipeline \
  --input data/histai-instruct.json \
  --filter data/splits/train_2k.txt \
  --output data/train_2k.h5 \
  --prism-base-path prism_embeddings/ \
  --n-clusters 15
```

## Complete Pipeline Example

Full command sequence from download to final HDF5:

```bash
# ====================
# Phase 1: Download Data
# ====================

# Download HISTAI-Instruct
wget https://huggingface.co/datasets/SaltySander/HISTAI-Instruct/resolve/main/histai-instruct.jsonl
python -c "import json; data=[json.loads(l) for l in open('histai-instruct.jsonl')]; json.dump(data,open('data/histai-instruct.json','w'),indent=2)"

# Download HISTAI WSIs from HuggingFace (manual download)
# Place in histai_downloads/HISTAI-{dataset}/case_*/slide_*.tiff

# ====================
# Phase 2: Generate Embeddings (repeat for each dataset)
# ====================

# Example for HISTAI-breast
./tiling_embedding/add_histai_spacing.sh -p histai_downloads/HISTAI-breast -o processed_histai_breast
./tiling_embedding/rename_to_case_nr.sh processed_histai_breast
./tiling_embedding/embedding/preprocessing/make_slides_csv_trident.py processed_histai_breast embedding_input_breast --batches 1

docker run -v $(pwd):/data trident:latest python3 /opt/run.py \
    --task all \
    --wsi_dir /data/processed_histai_breast \
    --job_dir /data/embedding_output_breast \
    --custom_list_of_wsis /data/embedding_input_breast/slide_batch.csv \
    --slide_encoder prism \
    --patch_size 224 \
    --mag 20 \
    --seg_batch_size 512 \
    --feat_batch_size 512 \
    --segmenter hest

./tiling_embedding/embedding/check_missing_embeddings_compared_to_dataset.py \
    embedding_input_breast/slide_batch.csv \
    embedding_output_breast

# Repeat above for HISTAI-skin-b1, HISTAI-colorectal-b1, etc.

# Organize all embeddings
mkdir -p prism_embeddings
for dataset in breast skin-b1 skin-b2 thorax hematologic gastrointestinal colorectal-b1 colorectal-b2; do
    mkdir -p "prism_embeddings/HISTAI-${dataset}"
    if [ -d "./embedding_output_${dataset}/20x_224px_0px_overlap" ]; then
        cp -r "./embedding_output_${dataset}/20x_224px_0px_overlap" "prism_embeddings/HISTAI-${dataset}/"
    fi
done

# ====================
# Phase 3: Create Training Datasets
# ====================

# Create splits
python -m auxilary_code.training_data_preprocessing.create_splits \
  --input data/histai-instruct.json \
  --output-dir data/splits/ \
  --prism-base-path prism_embeddings/ \
  --n-clusters 32 \
  --seed 42

# Generate HDF5 files
python -m auxilary_code.training_data_preprocessing.preprocessing_pipeline \
  --input data/histai-instruct.json \
  --filter data/splits/train.txt \
  --output data/train.h5 \
  --prism-base-path prism_embeddings/ \
  --n-clusters 15

python -m auxilary_code.training_data_preprocessing.preprocessing_pipeline \
  --input data/histai-instruct.json \
  --filter data/splits/val.txt \
  --output data/val.h5 \
  --prism-base-path prism_embeddings/ \
  --n-clusters 15

python -m auxilary_code.training_data_preprocessing.preprocessing_pipeline \
  --input data/histai-instruct.json \
  --filter data/splits/test.txt \
  --output data/test.h5 \
  --prism-base-path prism_embeddings/ \
  --n-clusters 15

# Optional: Create stratified subsets
python -m auxilary_code.training_data_preprocessing.create_stratified_subsets \
  --train-file data/splits/train.txt \
  --output-dir data/splits/ \
  --seed 42
```

## Expected Output Structure

```
project/
├── data/
│   ├── histai-instruct.json              # Input dataset
│   ├── train.h5, val.h5, test.h5         # Main HDF5 datasets
│   ├── train_2k.h5, train_9k.h5          # Optional subsets
│   ├── splits/
│   │   ├── train.txt, val.txt, test.txt  # Case assignments
│   │   └── train_2k.txt, train_9k.txt    # Subset assignments
│   ├── train_excluded_cases.txt          # Cases without embeddings
│   ├── val_excluded_cases.txt
│   └── test_excluded_cases.txt
├── prism_embeddings/
│   ├── HISTAI-breast/
│   │   └── 20x_224px_0px_overlap/
│   │       └── slide_features_prism/
│   │           ├── 0001_slide_H&E_0.h5
│   │           └── ...
│   ├── HISTAI-skin-b1/
│   │   └── ...
│   ├── HISTAI-colorectal-b1/
│   │   └── ...
│   └── ... (other datasets)
├── histai_downloads/                     # Raw HISTAI WSIs
│   ├── HISTAI-breast/
│   │   └── case_*/slide_*.tiff
│   └── ...
├── processed_histai_breast/              # Intermediate: processed slides
├── embedding_input_breast/               # Intermediate: CSV files
└── embedding_output_breast/              # Intermediate: raw embeddings
```

## Scripts Reference

### tiling_embedding/ - WSI Preprocessing & Embedding

| Script | Purpose |
|--------|---------|
| `add_histai_spacing.sh` | Add spacing metadata to TIFF files |
| `rename_to_case_nr.sh` | Prefix slides with case numbers |
| `embedding/preprocessing/make_slides_csv_trident.py` | Create slide CSV for Trident |
| `embedding/check_missing_embeddings_compared_to_dataset.py` | Validate embedding completion |
| `tissue_segmentation/batch_inference/inference.sh` | External tissue segmentation (optional) |
| `split_extracted_cases.py` | Split mixed case lists by dataset |

### training_data_preprocessing/ - HDF5 Dataset Creation

| Script | Purpose |
|--------|---------|
| `create_splits.py` | Create train/val/test splits with cluster sampling |
| `preprocessing_pipeline.py` | Generate HDF5 datasets with embeddings |
| `create_stratified_subsets.py` | Create smaller proportional training subsets |

## Troubleshooting

### Phase 2: Embedding Generation

**"wholeslidedata not found"**

- Install: `pip install wholeslidedata`

**"No H&E WSI files found"**

- Run `rename_to_case_nr.sh` first
- Verify files are named like `0000_slide_H&E_0.tiff`

**"Slides don't start with digit"**

- The `make_slides_csv_trident.py` script requires renamed files
- Run `rename_to_case_nr.sh` before creating CSV

**Embedding failures**

- Check log files in embedding output directory
- Use validation script to identify which stage failed
- Common issues: insufficient memory, GPU errors, corrupted slides

### Phase 3: HDF5 Creation

**"slide_features_prism not found"**

- Verify Trident completed successfully for ALL datasets
- Check directory structure matches expected format:
  `prism_embeddings/HISTAI-{dataset}/20x_224px_0px_overlap/slide_features_prism/*.h5`
- Ensure all datasets are in the `prism_embeddings/` base directory
- Run validation script (step 2.5) for each dataset

**"Cases don't match between HISTAI-Instruct and embeddings"**

- Run validation script after each embedding batch
- Some cases in HISTAI-Instruct may not have slides available
- The preprocessing pipeline will automatically exclude cases without embeddings
- Check `*_excluded_cases.txt` files for details

**"Wrong dataset name in case_mapping"**

- Verify case_mapping format in histai-instruct.json: `histai/HISTAI-{dataset}/case_{number}`
- Embedding files should be named: `{number}_slide_H&E_0.h5`
- Dataset organization should match: `prism_embeddings/HISTAI-{dataset}/...`

**"Memory error during clustering"**

- Reduce `--n-clusters` parameter (try 10 instead of 15)
- Process subsets separately if working with large datasets
- Ensure sufficient RAM (recommend 32GB+ for full dataset)

**"HDF5 file creation very slow"**

- Normal for large datasets (train.h5 can take several hours)
- Pipeline is resumable - you can stop and restart
- Monitor with: `watch ls -lh data/`

### Multi-Dataset Organization

**"Embeddings from different datasets are mixed up"**

- Ensure each dataset was processed with unique output directories
- Follow naming convention: `embedding_output_{dataset}/`
- Verify organization script correctly identifies dataset names

**"Some datasets missing from prism_embeddings/"**

- Check that embedding generation (step 2.4) completed for all datasets
- Verify output directories exist before running organization script
- Review validation output (step 2.5) for each dataset

## Additional Documentation

For detailed technical documentation:

- **Trident Docker commands**: See `tiling_embedding/embedding/preprocessing/preprocess_with_trident/how_to.md`
- **Script docstrings**: All Python scripts contain detailed docstrings with parameter descriptions
- **HISTAI dataset details**: <https://huggingface.co/collections/histai/histai-whole-slide-images-dataset>
- **HISTAI-Instruct details**: <https://huggingface.co/datasets/SaltySander/HISTAI-Instruct>
- **Trident framework**: <https://github.com/mahmoodlab/MADELEINE>

## Citation

If you use this preprocessing pipeline or the ANTONI-Alpha model, please cite:

```bibtex
@article{antoni2025,
  title={ANTONI-Alpha: Large-Scale Pathology Foundation Model},
  author={Your Name et al.},
  journal={...},
  year={2025}
}
```

Also cite the underlying datasets:

```bibtex
@article{histai2024,
  title={HISTAI: A Foundation Model for Histopathology Image Analysis},
  ...
}

@article{madeleine2024,
  title={MADELEINE: A Multimodal Dataset for Histopathology},
  ...
}
```

## License

[Add license information]

## Contact

For issues, questions, or contributions, please open an issue on GitHub or contact [maintainer email].
