# ANTONI-Alpha Data Preprocessing Pipeline

Complete pipeline for preprocessing HISTAI data for ANTONI-Alpha training: from raw whole-slide images to final HDF5 datasets.

## Overview

This pipeline transforms HISTAI whole-slide images and instruction data into structured training datasets through 3 main phases:

1. **Download HISTAI Data** — Obtain WSI images and HISTAI-Instruct QA pairs
2. **Generate Slide Embeddings** — Process WSIs through TRIDENT to extract PRISM features
3. **Create Training Datasets** — Generate train/val/test HDF5 files with stratified splits

```
HISTAI WSI Downloads (TIFF) ──────┐
                                   ├──> PRISM Embeddings (H5) ─┐
HISTAI-Instruct (JSONL) ──────────┘                            ├──> HDF5 Datasets
                                                                │    (train/val/test)
                                                                └──> Stratified Subsets
```

## Prerequisites

**Python dependencies** for the preprocessing scripts (phases 1–2):
```
pip install -r histai-processing/requirements.txt
```

**Python dependencies** for the training dataset scripts (phase 3):
```
pip install numpy h5py scikit-learn
```

**TRIDENT** for slide embedding (phase 2). Clone and install locally:
```
git clone https://github.com/mahmoodlab/TRIDENT.git
cd TRIDENT
pip install -e .
```

> **Required patch to TRIDENT:** By default, TRIDENT's PRISM encoder only saves the final slide embedding (shape `(1280,)`), but the training data scripts expect the full prototype matrix (shape `(513, 1280)`). After cloning, apply the following change to `trident/slide_encoder_models/load.py` in `PRISMSlideEncoder.forward()`:
>
> ```python
> # Before (original):
> z = self.model.slide_representations(x)
> z = z['image_embedding']
> return z
>
> # After (required):
> z = self.model.slide_representations(x)
> image_embedding = z['image_embedding'].unsqueeze(1)  # (1, 1, 1280)
> image_latents = z['image_latents']                   # (1, 512, 1280)
> z = torch.cat([image_embedding, image_latents], dim=1)  # (1, 513, 1280)
> return z
> ```

> **Note:** The PRISM slide encoder requires `transformers<5.0`. If you have a newer version installed, downgrade it:
> ```
> pip install "transformers<5.0"
> ```

Log in to Hugging Face so TRIDENT can download PRISM model weights automatically:
```
hf auth login
```

---

## Phase 1: Download HISTAI Data

### 1.1 Download HISTAI Whole-Slide Images

All HISTAI datasets are available on Hugging Face and are **gated** — you must request access on the dataset page before downloading.

| Dataset | Link |
|---------|------|
| HISTAI-mixed | [histai/HISTAI-mixed](https://huggingface.co/datasets/histai/HISTAI-mixed) |
| HISTAI-breast | [histai/HISTAI-breast](https://huggingface.co/datasets/histai/HISTAI-breast) |
| HISTAI-skin-b1 | [histai/HISTAI-skin-b1](https://huggingface.co/datasets/histai/HISTAI-skin-b1) |
| HISTAI-skin-b2 | [histai/HISTAI-skin-b2](https://huggingface.co/datasets/histai/HISTAI-skin-b2) |
| HISTAI-thorax | [histai/HISTAI-thorax](https://huggingface.co/datasets/histai/HISTAI-thorax) |
| HISTAI-hematologic | [histai/HISTAI-hematologic](https://huggingface.co/datasets/histai/HISTAI-hematologic) |
| HISTAI-gastrointestinal | [histai/HISTAI-gastrointestinal](https://huggingface.co/datasets/histai/HISTAI-gastrointestinal) |
| HISTAI-colorectal-b1 | [histai/HISTAI-colorectal-b1](https://huggingface.co/datasets/histai/HISTAI-colorectal-b1) |
| HISTAI-colorectal-b2 | [histai/HISTAI-colorectal-b2](https://huggingface.co/datasets/histai/HISTAI-colorectal-b2) |

After access is approved, download using the Hugging Face CLI:
```
hf download histai/HISTAI-breast --repo-type dataset --local-dir /data/raw/HISTAI-breast
```

Repeat for each dataset you want to process. Expected structure after download:
```
/data/raw/
├── HISTAI-breast/
│   ├── case_0000/
│   │   ├── slide_H&E_0.tiff
│   │   └── slide_H&E_1_x40.tiff
│   ├── case_0001/
│   │   └── slide_H&E_0.tiff
│   └── ...
├── HISTAI-skin-b1/
│   └── case_*/slide_*.tiff
└── ... (other datasets)
```

### 1.2 Download HISTAI-Instruct Dataset

```
hf download SaltySander/HISTAI-Instruct --repo-type dataset --local-dir /data/instruct
```

The dataset is a JSONL file that needs to be converted to a JSON array before use:
```
python3 -c "
import json
with open('/data/instruct/histai-instruct.jsonl') as f:
    data = [json.loads(line) for line in f]
with open('/data/instruct/histai-instruct.json', 'w') as f:
    json.dump(data, f)
"
```

> **Note on reproducibility:** The HISTAI-Instruct dataset already includes pre-computed train/val/test splits in the `splits/` directory (`train.txt`, `val.txt`, `test.txt`, `train_2k.txt`, `train_9k.txt`). If you want to reproduce our exact experimental results, use these splits directly in phase 3 instead of re-running `create_splits.py`.

---

## Phase 2: Generate Slide Embeddings

This phase processes raw HISTAI slides to generate PRISM embeddings. **Repeat steps 2.1–2.4 for each HISTAI dataset separately.** For full details and options for each script, see [`histai-processing/README.md`](histai-processing/README.md).

### 2.1 Add Spacing Metadata

HISTAI slides lack spacing metadata. This script adds μm/pixel information based on filename:
- Regular slides: 0.5 μm/px (20x)
- Slides with `x40` in filename: 0.25 μm/px (40x)

```
python3 histai-processing/add_histai_spacing.py \
    -p /data/raw/HISTAI-breast \
    -o /data/processed/HISTAI-breast
```

Output structure:
```
/data/processed/HISTAI-breast/
├── case_0000/
│   └── processed/
│       ├── slide_H&E_0.tiff
│       └── slide_H&E_1_x40.tiff
└── case_0001/
    └── processed/
        └── slide_H&E_0.tiff
```

### 2.2 Rename Files with Case Numbers

Prevents filename conflicts when cases have identically-named slides (e.g. two cases both have `slide_H&E_0.tiff`):

```
# Dry run first to verify
python3 histai-processing/rename_to_case_nr.py /data/processed/HISTAI-breast -n

# Apply renaming
python3 histai-processing/rename_to_case_nr.py /data/processed/HISTAI-breast
```

This renames `slide_H&E_0.tiff` → `0000_slide_H&E_0.tiff` using the case directory number as prefix.

### 2.3 Prepare CSV for Embedding

Creates `slide_batch.csv` listing all slides for TRIDENT to process:

```
python3 histai-processing/make_slides_csv_trident.py \
    HISTAI-breast \
    1 \
    -i /data/processed/ \
    -o /data/processed/
```

The CSV is written to `/data/processed/HISTAI-breast/slide_batch.csv`. Set the second argument > 1 to split into multiple batches.

### 2.4 Generate PRISM Embeddings

Run TRIDENT to produce slide-level PRISM features. Make sure you have applied the required patch described in the Prerequisites section.

```
python3 /path/to/TRIDENT/run_batch_of_slides.py \
    --task all \
    --wsi_dir /data/processed/HISTAI-breast \
    --job_dir /data/output/HISTAI-breast \
    --custom_list_of_wsis /data/processed/HISTAI-breast/slide_batch.csv \
    --slide_encoder prism \
    --patch_size 224 \
    --mag 20 \
    --seg_batch_size 32 \
    --feat_batch_size 32 \
    --segmenter hest \
    --seg_conf_thresh 0.4 \
    --wsi_cache /tmp \
    --cache_batch_size 16
```

> **Note on batch sizes:** `seg_batch_size` and `feat_batch_size` depend on GPU memory. The values above (32) work on a 24GB GPU with other processes running. Increase them if you have more free VRAM.

Output structure:
```
/data/output/HISTAI-breast/
├── _logs_segmentation.txt
├── contours/
├── contours_geojson/
├── thumbnails/
└── 20x_224px_0px_overlap/
    ├── _logs_coords.txt
    ├── _logs_feats_virchow.txt
    ├── _logs_slide_features_prism.txt
    ├── patches/
    ├── features_virchow/
    └── slide_features_prism/
        ├── 0001_slide_H&E_0.h5
        ├── 0042_slide_H&E_0.h5
        └── ...
```

Each HDF5 file contains a `features` dataset of shape **(513, 1280)** — 1 summary embedding + 512 PRISM latents × 1280 dimensions.

**Repeat steps 2.1–2.4 for all datasets** (HISTAI-skin-b1, HISTAI-colorectal-b1, etc.).

### 2.5 Organize Embeddings

After processing all datasets, organize the outputs into the structure expected by phase 3:

```
for dataset in breast skin-b1 skin-b2 thorax hematologic gastrointestinal colorectal-b1 colorectal-b2; do
    mkdir -p /data/prism_embeddings/HISTAI-${dataset}
    cp -r /data/output/HISTAI-${dataset}/20x_224px_0px_overlap \
          /data/prism_embeddings/HISTAI-${dataset}/
done
```

Expected final structure:
```
/data/prism_embeddings/
├── HISTAI-breast/
│   └── 20x_224px_0px_overlap/
│       └── slide_features_prism/
│           ├── 0001_slide_H&E_0.h5
│           └── ...
├── HISTAI-skin-b1/
│   └── 20x_224px_0px_overlap/
│       └── slide_features_prism/
│           └── ...
└── ... (other datasets)
```

---

## Phase 3: Create Training Datasets

This phase combines HISTAI-Instruct with PRISM embeddings to produce HDF5 training datasets. All scripts are in `training_data_preprocessing/`.

### 3.1 Create Train/Val/Test Splits

> **Reproducibility:** The HISTAI-Instruct HF dataset (`SaltySander/HISTAI-Instruct`) already includes splits in `splits/train.txt`, `splits/val.txt`, and `splits/test.txt`. To reproduce our exact results, skip this step and use those files directly in step 3.2.

To generate new splits from scratch using cluster-based stratified sampling:

```
python3 training_data_preprocessing/create_splits.py \
    --input /data/instruct/histai-instruct.json \
    --output-dir /data/splits/ \
    --prism-base-path /data/prism_embeddings/ \
    --n-clusters 32 \
    --seed 42
```

**What it does:**
- Groups cases by HISTAI subset
- Loads PRISM first prototypes (1280-d) for each case
- K-means clustering (k=32) within each subset (falls back to random split if < 100 samples)
- Stratified 80-10-10 split by cluster

**Outputs:** `train.txt`, `val.txt`, `test.txt` — one `histai/HISTAI-{dataset}/case_{id}` per line.

### 3.2 Generate HDF5 Datasets

Run once for each split (train, val, test):

```
python3 training_data_preprocessing/preprocessing_pipeline.py \
    --input /data/instruct/histai-instruct.json \
    --filter /data/splits/train.txt \
    --output /data/train.h5 \
    --prism-base-path /data/prism_embeddings/ \
    --n-clusters 15

python3 training_data_preprocessing/preprocessing_pipeline.py \
    --input /data/instruct/histai-instruct.json \
    --filter /data/splits/val.txt \
    --output /data/val.h5 \
    --prism-base-path /data/prism_embeddings/ \
    --n-clusters 15

python3 training_data_preprocessing/preprocessing_pipeline.py \
    --input /data/instruct/histai-instruct.json \
    --filter /data/splits/test.txt \
    --output /data/test.h5 \
    --prism-base-path /data/prism_embeddings/ \
    --n-clusters 15
```

**What it does:**
- Filters HISTAI-Instruct to cases in the split file
- Validates that PRISM embeddings exist for each case
- K-means clustering by organ type (k=15) using PRISM features
- Writes structured HDF5 with embeddings, text attributes, and cluster assignments

**HDF5 structure:**
```
train.h5/
├── embeddings/
│   ├── HISTAI-breast__case_0001/
│   │   ├── features      # (513, 1280) PRISM array
│   │   ├── cluster_id    # scalar int32
│   │   └── organ         # string
│   └── ...
├── text_attributes/
│   ├── HISTAI-breast__case_0001  # JSON string with full case data
│   └── ...
└── metadata/
    ├── cluster_info       # JSON: cluster statistics by organ
    ├── filtering_stats    # JSON: cases processed/excluded
    ├── pipeline_config    # JSON: run parameters
    └── processing_log     # JSON: execution timeline
```

Note: case keys use double underscore (`HISTAI-breast__case_0001`) for HDF5 path safety.

### 3.3 Create Stratified Subsets (Optional)

Creates smaller training subsets proportional to the original subset distribution:

```
python3 training_data_preprocessing/create_stratified_subsets.py \
    --train-file /data/splits/train.txt \
    --output-dir /data/splits/ \
    --seed 42
```

**Outputs:** `train_2k.txt` (2000 cases) and `train_9k.txt` (9000 cases). To create the corresponding HDF5 files, run step 3.2 with these files as `--filter`.

> **Note:** Pre-computed `train_2k.txt` and `train_9k.txt` are also included in the HISTAI-Instruct HF dataset.

---

## Troubleshooting

**PRISM embeddings are shape `(1280,)` instead of `(513, 1280)`**
- You forgot to apply the required TRIDENT patch. See the Prerequisites section.

**`transformers` version error when loading PRISM**
- Run `pip install "transformers<5.0"`.

**`No H&E WSI files found` / `Slides don't start with digit`**
- Run `rename_to_case_nr.py` (step 2.2) before creating the CSV and before embedding.

**CUDA out of memory during segmentation or feature extraction**
- Lower `--seg_batch_size` and `--feat_batch_size`. Values of 32 work on a 24GB GPU with other processes running.

**`slide_features_prism not found` in phase 3**
- Verify TRIDENT completed successfully for all datasets.
- Check that the directory structure matches: `prism_embeddings/HISTAI-{dataset}/20x_224px_0px_overlap/slide_features_prism/*.h5`

**Cases skipped with "no valid embeddings"**
- Some cases in HISTAI-Instruct may not have slides in the WSI datasets. This is expected. Skipped cases are tracked in `*_excluded_cases.txt`.

**Memory error during clustering**
- Reduce `--n-clusters` (try 10 instead of 15 or 32).

---

## Scripts Reference

### `histai-processing/` — WSI preprocessing and embedding

| Script | Purpose |
|--------|---------|
| `add_histai_spacing.py` | Add μm/px spacing metadata to TIFF files |
| `rename_to_case_nr.py` | Prefix slide filenames with case number |
| `make_slides_csv_trident.py` | Generate `slide_batch.csv` for TRIDENT |

See [`histai-processing/README.md`](histai-processing/README.md) for full usage details.

### `training_data_preprocessing/` — HDF5 dataset creation

| Script | Purpose |
|--------|---------|
| `create_splits.py` | Create train/val/test splits via cluster sampling |
| `preprocessing_pipeline.py` | Generate HDF5 datasets with embeddings and metadata |
| `create_stratified_subsets.py` | Create proportional 2k/9k training subsets |
