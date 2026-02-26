Last update: 26-02-2026
Written by: Sebastiaan Ram
Last update by: Sebastiaan Ram

This is a quick how-to for converting HistAI slides from the raw format to its slide embedding. For this example we assume the raw data is stored in a folder we call `/data/raw`. The processed data will be saved in `/data/processed`.

## Prerequisites

Install the Python dependencies for the preprocessing scripts:
```
pip install -r requirements.txt
```

For the embedding step (step 4), you need [TRIDENT](https://github.com/mahmoodlab/TRIDENT). The recommended way is via Docker:
```
git clone https://github.com/mahmoodlab/TRIDENT.git
cd TRIDENT
docker build -t trident:latest .
```

You will also need a Hugging Face token (`HF_TOKEN`) with access to the PRISM model weights. Create a `.env` file containing:
```
HF_TOKEN=<your_token>
```

## Steps

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

4) **Embed!** Ensure that you have the following:
    - For each batch, a `slide_batch.csv` file.
    - (optional) for each image in your `slide_batch.csv`, a corresponding tissue mask. TRIDENT allows you to provide your own tissue masks. However, we generate these ourselves

If you have all of that, you can start embedding! Below you will find the parameters that we used. Feel free to adjust based on your requirements. **Note** that the job_dir is where the outputs will be stored. We can define this as `/data/output/`

Run TRIDENT via Docker, mounting your data directories and passing the `.env` file with your `HF_TOKEN`:
```
docker run --gpus all \
    --env-file=.env \
    -v /data:/data \
    trident:latest python3 /opt/run.py \
    --task all \
    --wsi_dir "/data/processed/HISTAI-skin-b1" \
    --job_dir "/data/output/HISTAI-skin-b1" \
    --custom_list_of_wsis "/data/processed/HISTAI-skin-b1/slide_batch.csv" \
    --slide_encoder prism \
    --patch_size 224 \
    --mag 20 \
    --seg_batch_size 512 \
    --feat_batch_size 512 \
    --segmenter hest \
    --seg_conf_thresh 0.4 \
    --wsi_cache /tmp \
    --cache_batch_size 16
```