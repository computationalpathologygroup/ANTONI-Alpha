#!/usr/bin/env python3

"""
Script to rename slide files by prefixing them with their case number.
Finds all case_* directories and renames files containing 'slide' in their name.
"""

import argparse
import sys
from pathlib import Path


def extract_case_number(case_dir: Path) -> str:
    """Extract case number from case directory name (e.g., 'case_0001' -> '0001')"""
    return case_dir.name.replace('case_', '')


def process_case_directory(case_dir: Path, dry_run: bool = False) -> tuple[int, int]:
    """
    Process a single case directory and rename slide files.
    
    Returns:
        tuple: (files_renamed, files_skipped)
    """
    case_nr = extract_case_number(case_dir)
    files_renamed = 0
    files_skipped = 0
    
    # Find all files containing 'slide' in their name
    slide_files = [f for f in case_dir.rglob('*slide*') if f.is_file()]
    
    for file_path in sorted(slide_files):
        filename = file_path.name
        
        # Skip if the case number is already in the filename
        if filename.startswith(f"{case_nr}_"):
            print(f"Skipping {filename} - already has case number")
            files_skipped += 1
            continue
        
        # Create new filename with case number prefix
        new_filename = f"{case_nr}_{filename}"
        new_filepath = file_path.parent / new_filename
        
        if dry_run:
            print(f"[DRY RUN] Would rename: {file_path} -> {new_filepath}")
        else:
            print(f"Renaming: {file_path} -> {new_filepath}")
            file_path.rename(new_filepath)
        
        files_renamed += 1
    
    return files_renamed, files_skipped


def main():
    parser = argparse.ArgumentParser(
        description="Rename slide files by prefixing them with their case number from case_* directories"
    )
    
    parser.add_argument(
        'parent_directory',
        type=str,
        help='Parent directory containing case_* subdirectories'
    )
    
    parser.add_argument(
        '-n', '--dry-run',
        action='store_true',
        help='Show what would be renamed without actually renaming files'
    )
    
    args = parser.parse_args()
    
    # Validate parent directory
    parent_dir = Path(args.parent_directory)
    if not parent_dir.exists():
        print(f"Error: Directory '{parent_dir}' does not exist", file=sys.stderr)
        sys.exit(1)
    
    if not parent_dir.is_dir():
        print(f"Error: '{parent_dir}' is not a directory", file=sys.stderr)
        sys.exit(1)
    
    # Find all case_* directories
    case_dirs = sorted(parent_dir.glob('case_*'))
    
    if not case_dirs:
        print(f"Warning: No case_* directories found in '{parent_dir}'", file=sys.stderr)
        sys.exit(0)
    
    if args.dry_run:
        print("=" * 60)
        print("DRY RUN MODE - No files will be renamed")
        print("=" * 60)
        print()
    
    # Process each case directory
    total_renamed = 0
    total_skipped = 0
    
    for case_dir in case_dirs:
        if not case_dir.is_dir():
            continue
        
        print(f"\nProcessing: {case_dir.name}")
        renamed, skipped = process_case_directory(case_dir, dry_run=args.dry_run)
        total_renamed += renamed
        total_skipped += skipped
    
    # Print summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total case directories processed: {len(case_dirs)}")
    print(f"Total files renamed: {total_renamed}")
    print(f"Total files skipped: {total_skipped}")
    
    if args.dry_run:
        print("\nThis was a dry run. Run without -n/--dry-run to actually rename files.")


if __name__ == "__main__":
    main()
