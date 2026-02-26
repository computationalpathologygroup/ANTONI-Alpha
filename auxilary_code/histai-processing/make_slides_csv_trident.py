#!/usr/bin/env python3
import argparse
import csv
import sys
import yaml
from pathlib import Path


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate slide batch CSV files for Trident processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  %(prog)s HISTAI-breast 1 -i /data/raw -o /data/output
  %(prog)s HISTAI-breast 4 --input-dir /data/raw --output-dir /data/output
        """
    )
    
    parser.add_argument(
        'dataset_name',
        type=str,
        help='Name of the dataset'
    )
    
    parser.add_argument(
        'nr_of_batches',
        type=int,
        help='Number of batches to split the dataset into (must be >= 1)'
    )
    
    parser.add_argument(
        '-i', '--input-dir',
        type=str,
        required=True,
        help='Input directory containing dataset folders'
    )
    
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        required=True,
        help='Output directory for CSV files'
    )
    
    args = parser.parse_args()
    
    # Validate batch number
    if args.nr_of_batches < 1:
        parser.error("nr_of_batches must be >= 1")
    
    return args


def collect_slides(local_root):
    entries = []
    total_wsi = []

    for case_dir in sorted(local_root.glob("case_*")):
        processed_dir = case_dir / "processed"
        if not processed_dir.exists():
            continue

        # include tif/tiff slides that start with numbers
        for slide_path in processed_dir.glob("*.tif*"):
            # Skip files that contain "mask" in their name or don't start with a number
            if "mask" in slide_path.stem.lower() or not "h&e" in slide_path.stem.lower() or not any(c.isdigit() for c in slide_path.stem[0:1]):
                continue

            total_wsi.append(slide_path)
            entries.append(str(slide_path))

    return entries, total_wsi

def create_batch_folder(entries, batch_number, total_batches, csv_base, dataset):
    # Calculate the start and end indices for this batch
    total_entries = len(entries)
    batch_size = (total_entries + total_batches - 1) // total_batches  # Round up division
    start_idx = batch_number * batch_size  # Now using 0-based index
    end_idx = min(start_idx + batch_size, total_entries)

    # Get entries for this batch
    batch_entries = entries[start_idx:end_idx]

    # Create batch directory and paths
    if total_batches == 1:
        batch_dir = Path(csv_base)
        batch_csv = batch_dir / "slide_batch.csv"
    else:
        batch_dir = Path(csv_base) / f"batch_{batch_number}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        batch_csv = batch_dir / "slide_batch.csv"

    # Find and copy the first yaml config file from the parent directory
    yaml_files = list(Path(csv_base).glob("*.yaml"))
    if yaml_files:
        config_file = yaml_files[0]  # Take the first yaml file found
        config_dest = batch_dir / config_file.name

        # Read the config file
        with open(config_file) as f:
            config = yaml.safe_load(f)

        # Update paths
        config['csv'] = str(batch_csv)
        config['output_dir'] = str(batch_dir)
        config['wandb']['exp_name'] = f"{dataset}_batch_{batch_number}"

        # Write the modified config
        with open(config_dest, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    # Write the CSV file
    with open(batch_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["wsi"])
        for wsi in batch_entries:
            writer.writerow([wsi])

    return len(batch_entries)

def main():
    args = parse_arguments()
    dataset = args.dataset_name
    batch_id = args.nr_of_batches

    local_root = Path(args.input_dir) / dataset
    csv_base = f"{args.output_dir}/{dataset}"

    # Validate dataset directory exists
    if not local_root.exists():
        print(f"Error: Dataset directory not found: {local_root}")
        sys.exit(1)

    entries, total_wsi = collect_slides(local_root)

    # If batch_id is 1, write all to a single file without number
    if batch_id == 1:
        num_written = create_batch_folder(entries, 0, 1, csv_base, dataset)
        print(f"Wrote all {num_written} entries to slide_batch.csv")
    else:
        # Write multiple batch files
        print(f"Distributing {len(entries)} entries across {batch_id} batches...")
        for i in range(batch_id):  # Now using range(batch_id) for 0-based indexing
            num_written = create_batch_folder(entries, i, batch_id, csv_base, dataset)
            batch_dir = "batch_" if i == 0 else f"batch_{i}"
            print(f"Wrote {num_written} entries to {batch_dir}/slide_batch.csv")

    if len(total_wsi) == 0:
        print("Warning: No WSI files found in 'processed/' directories. Have you added the case numbers?")
    else:
        print(f"Found {len(total_wsi)} total WSI files in 'processed/'")

if __name__ == "__main__":
    main()

