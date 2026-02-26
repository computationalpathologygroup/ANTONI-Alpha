Last update: 26-02-2026
Written by: Sebastiaan Ram
Last update by: Sebastiaan Ram

This is a quick how-to for converting HistAI slides from the raw format to its slide embedding. For this example we assume the raw data is stored in a folder we call `/data/raw`. The processed data will be saved in `/data/processed`.

## Available datasets

The following HistAI datasets are available on Hugging Face. All datasets are **gated** — you must request access on the dataset page before downloading.

| Dataset | HF link |
|---------|---------|
| HISTAI-mixed | [histai/HISTAI-mixed](https://huggingface.co/datasets/histai/HISTAI-mixed) |
| HISTAI-breast | [histai/HISTAI-breast](https://huggingface.co/datasets/histai/HISTAI-breast) |
| HISTAI-skin-b2 | [histai/HISTAI-skin-b2](https://huggingface.co/datasets/histai/HISTAI-skin-b2) |
| HISTAI-skin-b1 | [histai/HISTAI-skin-b1](https://huggingface.co/datasets/histai/HISTAI-skin-b1) |
| HISTAI-thorax | [histai/HISTAI-thorax](https://huggingface.co/datasets/histai/HISTAI-thorax) |
| HISTAI-hematologic | [histai/HISTAI-hematologic](https://huggingface.co/datasets/histai/HISTAI-hematologic) |
| HISTAI-gastrointestinal | [histai/HISTAI-gastrointestinal](https://huggingface.co/datasets/histai/HISTAI-gastrointestinal) |
| HISTAI-colorectal-b1 | [histai/HISTAI-colorectal-b1](https://huggingface.co/datasets/histai/HISTAI-colorectal-b1) |
| HISTAI-colorectal-b2 | [histai/HISTAI-colorectal-b2](https://huggingface.co/datasets/histai/HISTAI-colorectal-b2) |

### Downloading a dataset

After your access request has been approved, log in with the Hugging Face CLI and download the dataset to your raw data folder:

```
hf auth login
hf download histai/HISTAI-skin-b1 --repo-type dataset --local-dir /data/raw/HISTAI-skin-b1
```

Replace `histai/HISTAI-skin-b1` and the `--local-dir` path with the dataset you want to download.

## Prerequisites

Install the Python dependencies for the preprocessing scripts:
```
pip install -r requirements.txt
```

For the embedding step (step 4), you need [TRIDENT](https://github.com/mahmoodlab/TRIDENT). Clone the repo and install it locally:
```
git clone https://github.com/mahmoodlab/TRIDENT.git
cd TRIDENT
pip install -e .
```

> **Note:** The PRISM slide encoder requires `transformers<5.0`. If you have a newer version installed, downgrade it:
> ```
> pip install "transformers<5.0"
> ```

> **Required patch to TRIDENT:** By default, TRIDENT's PRISM encoder only saves the final slide embedding (shape `(1280,)`), but downstream scripts expect the full prototype matrix (shape `(513, 1280)` = 1 summary embedding + 512 latents). After cloning TRIDENT, apply the following change to `trident/slide_encoder_models/load.py` in `PRISMSlideEncoder.forward()`:
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

You will also need a Hugging Face account with access to the PRISM model weights. Log in via the CLI so TRIDENT can download the model automatically:
```
hf auth login
```

## Steps

The steps below use `HISTAI-skin-b1` as an example. Replace it with the name of the dataset you downloaded.

1) **Run the `add_histai_spacing.py` script** to add spacing to the HistAI data (they come as general tiffs without spacing information) and move the file to a new location. Provide the `.txt` file for a selection of slides, or use an input folder and output folder. For this example we want to embed all the slides from HISTAI-skin-b1:
```
python3 add_histai_spacing.py -p "/data/raw/HISTAI-skin-b1/" -o "/data/processed/HISTAI-skin-b1/"
```

2) **Rename files** to use their case number instead of original filename. The problem when not doing this is that most slides within each case folder have the same name (e.g. both case_0000 and case_0004 have the file `slide_H&E_0.tiff`), which will conflict during the embedding step. By running the `rename_to_case_nr.py` script, the case number is attached as a prefix to the filename. For example, slide `slide_H&E_0` from case 0000 will be renamed to `0000_slide_H&E_0.tiff`.

You can do a dry run using
```
python3 rename_to_case_nr.py "/data/processed/HISTAI-skin-b1" -n
```

If you're satisfied, run the renaming by removing the `-n` flag

3) **Prepare the embedding process**. Using the `make_slides_csv_trident.py` script, provide a database name (e.g. HISTAI-skin-b1) and the number of batches. If the n_batches = 1, the `slides.csv` file will be placed in the parent folder. When n_batches > 1, different folders will be made for each batch (e.g. `batch_0`, `batch_1`, etc.). This slides file will be used to tell the script which files it should process.

```
python3 make_slides_csv_trident.py \
    HISTAI-skin-b1 \
    1 \
    -i /data/processed/ \
    -o /data/processed/
```

The csv file will then be stored in the `-o + [dataset_name]` folder.

4) **Embed!** Ensure that you have a `slide_batch.csv` file for each batch (generated in step 3). Then run TRIDENT from the cloned repo directory. Below are the parameters we used — feel free to adjust based on your requirements. **Note** that `job_dir` is where the outputs will be stored (e.g. `/data/output/`).

```
python3 /path/to/TRIDENT/run_batch_of_slides.py \
    --task all \
    --wsi_dir "/data/processed/HISTAI-skin-b1" \
    --job_dir "/data/output/HISTAI-skin-b1" \
    --custom_list_of_wsis "/data/processed/HISTAI-skin-b1/slide_batch.csv" \
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

> **Note on batch sizes:** `seg_batch_size` and `feat_batch_size` depend on your GPU memory. The values above (32) work on a 24GB GPU with other processes running. Increase them if you have more free VRAM.
