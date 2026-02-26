#!/usr/bin/env python3

"""
Script to process HISTAI dataset and add spacing metadata
Structure: /data/raw/HISTAI-skin-b1/case_XXXX/slide_*.tiff
- Regular slides get 0.5 spacing
- Slides with 'x40' in name get 0.25 spacing
- Output goes to a 'processed' folder within each case directory
"""

import argparse
import subprocess
import sys
import os
import shutil
from pathlib import Path
from typing import Optional, Tuple
import tempfile


# ANSI color codes
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color


# Logging functions
def log_info(message: str):
    print(f"{Colors.BLUE}[INFO]{Colors.NC} {message}")


def log_success(message: str):
    print(f"{Colors.GREEN}[SUCCESS]{Colors.NC} {message}")


def log_warning(message: str):
    print(f"{Colors.YELLOW}[WARNING]{Colors.NC} {message}")


def log_error(message: str):
    print(f"{Colors.RED}[ERROR]{Colors.NC} {message}")


class HISIAIProcessor:
    def __init__(self, parent_folder: str, output_base_folder: str, file_list: Optional[str] = None):
        self.parent_folder = Path(parent_folder) if parent_folder else None
        self.output_base_folder = Path(output_base_folder)
        self.file_list = Path(file_list) if file_list else None
        
        # Paths
        self.tmp_folder = Path("/tmp/histai_processing")
        
        # Initialize counters
        self.total_cases = 0
        self.processed_cases = 0
        self.total_files = 0
        self.processed_files = 0
        self.skipped_files = 0
        self.error_files = 0
        
        # Track processed cases
        self.processed_case_names = set()

    def validate_setup(self):
        """Validate that all required paths and scripts exist"""
        if self.parent_folder and not self.parent_folder.exists():
            log_error(f"Parent folder does not exist: {self.parent_folder}")
            sys.exit(1)
        
        # Create output and temp folders
        self.output_base_folder.mkdir(parents=True, exist_ok=True)
        self.tmp_folder.mkdir(parents=True, exist_ok=True)
        
        log_info(f"Using temporary folder: {self.tmp_folder}")
        log_info(f"Output base folder: {self.output_base_folder}")

    def determine_spacing(self, filename: str) -> float:
        """Determine spacing based on filename"""
        if 'x40' in filename:
            return 0.25
        else:
            return 0.5

    def get_case_info(self, file_path: Path) -> Tuple[Path, str]:
        """Get case directory and name from file path"""
        case_dir = file_path.parent
        
        # If the slides are not directly under case_dir, go up one level
        if not case_dir.name.startswith('case_'):
            case_dir = case_dir.parent
        
        case_name = case_dir.name
        return case_dir, case_name

    def process_single_file(self, tiff_file: Path):
        """Process a single TIFF file"""
        if not tiff_file.exists():
            log_warning(f"File not found, skipping: {tiff_file}")
            self.skipped_files += 1
            return
        
        filename = tiff_file.name
        case_dir, case_name = self.get_case_info(tiff_file)
        
        # Create output processed folder for this case
        processed_dir = self.output_base_folder / case_name / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        self.total_files += 1
        
        # Skip if file doesn't start with 'slide_'
        if not filename.startswith('slide_'):
            log_warning(f"Skipping non-slide file: {filename}")
            self.skipped_files += 1
            return
        
        # Determine spacing
        spacing = self.determine_spacing(filename)
        spacing_type = "x40" if spacing == 0.25 else "x20"
        log_info(f"  Processing {spacing_type} slide ({spacing} spacing): {filename}")
        
        # Check if output file already exists
        output_file = processed_dir / filename
        if output_file.exists():
            log_warning(f"  Output file already exists, skipping: {output_file}")
            self.skipped_files += 1
            return
        
        # Create case-specific temp folder
        case_tmp_folder = self.tmp_folder / case_name
        case_tmp_folder.mkdir(parents=True, exist_ok=True)
        
        log_info(f"Processing case: {case_name}")
        log_info(f"  Processing: {filename} with spacing {spacing} μm/px")
        log_info(f"    Starting Python script execution...")
        
        # Run the add_image_spacing.py script
        script_dir = Path(__file__).resolve().parent
        try:
            result = subprocess.run(
                [
                    "python3",
                    str(script_dir / "add_image_spacing.py"),
                    "--input_data", str(tiff_file),
                    "--spacing", str(spacing),
                    "--output_folder", str(processed_dir),
                    "--tmp_folder", str(case_tmp_folder)
                ],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Print Python script output
            for line in result.stdout.splitlines():
                print(f"    [Python] {line}")
            
            log_success(f"  Successfully processed: {filename}")
            self.processed_files += 1
            self.processed_case_names.add(case_name)
            
        except subprocess.CalledProcessError as e:
            # Print error output
            for line in e.stdout.splitlines():
                print(f"    [Python] {line}")
            for line in e.stderr.splitlines():
                print(f"    [Python] {line}")
            
            log_error(f"  Failed to process {filename} (exit code: {e.returncode})")
            self.error_files += 1
            log_info(f"  Continuing to next file...")
        
        finally:
            # Clean up case temp folder
            if case_tmp_folder.exists():
                shutil.rmtree(case_tmp_folder, ignore_errors=True)

    def process_files_in_directory(self, dir_path: Path):
        """Process all slide files in a directory"""
        if not dir_path.exists():
            log_warning(f"Directory not found, skipping: {dir_path}")
            return
        
        log_info(f"Processing directory: {dir_path}")
        
        # Find all slide_*.tiff and slide_*.tif files
        tiff_files = list(dir_path.glob("slide_*.tiff")) + list(dir_path.glob("slide_*.tif"))
        
        for tiff_file in sorted(tiff_files):
            self.process_single_file(tiff_file)

    def process_from_file_list(self):
        """Process files listed in a file list"""
        if not self.file_list.exists():
            log_error(f"File list not found: {self.file_list}")
            sys.exit(1)
        
        log_info(f"Using file list: {self.file_list}")
        
        # Track unique cases for counting
        case_names = set()
        
        with open(self.file_list, 'r') as f:
            for line in f:
                # Trim whitespace and skip blank lines and comments
                file_path_str = line.strip()
                if not file_path_str or file_path_str.startswith('#'):
                    continue
                
                # Handle relative paths
                file_path = Path(file_path_str)
                if not file_path.is_absolute():
                    file_path = Path.cwd() / file_path
                
                # Add case name for counting
                if file_path.exists():
                    _, case_name = self.get_case_info(file_path)
                    case_names.add(case_name)
                
                # Process either a file or directory
                if file_path.is_dir():
                    self.process_files_in_directory(file_path)
                elif file_path.is_file():
                    self.process_single_file(file_path)
                else:
                    log_warning(f"Path not found or invalid: {file_path}")
        
        self.total_cases = len(case_names)
        self.processed_cases = len(self.processed_case_names)

    def process_from_parent_folder(self):
        """Process all case directories in the parent folder"""
        log_info(f"Parent folder: {self.parent_folder}")
        
        # Check if the parent folder itself is a case directory
        if self.parent_folder.name.startswith('case_'):
            # Process this single case directory
            case_dirs = [self.parent_folder]
        else:
            # Find all case_* directories
            case_dirs = sorted(self.parent_folder.glob("case_*"))
        
        for case_dir in case_dirs:
            if not case_dir.is_dir():
                continue
            
            case_name = case_dir.name
            self.total_cases += 1
            
            log_info(f"Processing case: {case_name}")
            
            # Debug: List files in the case directory
            try:
                files = list(case_dir.iterdir())[:10]
                log_info(f"  Files in {case_dir}:")
                for f in files:
                    log_info(f"    {f.name}")
            except Exception as e:
                log_warning(f"  Could not list directory: {e}")
            
            # Find all slide files in the case directory
            tiff_files = (
                list(case_dir.glob("slide_*.tiff")) + 
                list(case_dir.glob("slide_*.tif"))
            )
            
            case_had_files = False
            for tiff_file in sorted(tiff_files):
                self.process_single_file(tiff_file)
                if case_name in self.processed_case_names:
                    case_had_files = True
            
            if case_had_files:
                self.processed_cases += 1
                log_success(f"Completed case: {case_name}")
            else:
                log_warning(f"No files processed for case: {case_name}")
            
            print()  # Add blank line between cases

    def run(self):
        """Main processing loop"""
        self.validate_setup()
        
        log_info("Starting HISTAI dataset processing...")
        
        if self.file_list:
            self.process_from_file_list()
        else:
            self.process_from_parent_folder()
        
        # Clean up main temp folder
        if self.tmp_folder.exists():
            shutil.rmtree(self.tmp_folder, ignore_errors=True)
        
        self.print_summary()

    def print_summary(self):
        """Print processing summary"""
        print()
        print("=" * 40)
        log_info("PROCESSING SUMMARY")
        print("=" * 40)
        print(f"Total cases found: {self.total_cases}")
        print(f"Cases processed: {self.processed_cases}")
        print(f"Total files found: {self.total_files}")
        print(f"Files successfully processed: {self.processed_files}")
        print(f"Files skipped: {self.skipped_files}")
        print(f"Files with errors: {self.error_files}")
        print()
        
        if self.error_files == 0:
            log_success("All processing completed successfully!")
        else:
            log_warning(f"Processing completed with {self.error_files} errors")


def main():
    parser = argparse.ArgumentParser(
        description="Process HISTAI dataset and add spacing metadata",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Expected structure: <input_dir>/case_XXXX/slide_*.tiff
- Regular slides get 0.5 spacing
- Slides with 'x40' in name get 0.25 spacing
- Output goes to a 'processed' folder within each case directory
        """
    )
    
    parser.add_argument(
        '-f', '--file-list',
        type=str,
        help='Path to a .txt file with list of files to process (one per line). Overrides parent-folder scanning when provided.'
    )
    
    parser.add_argument(
        '-p', '--parent-folder',
        type=str,
        default='/data/raw',
        help='Parent folder containing case_* directories (default: /data/raw)'
    )
    
    parser.add_argument(
        '-o', '--output-base',
        type=str,
        default='/data/processed',
        help='Base folder for processed output (default: /data/processed)'
    )
    
    args = parser.parse_args()
    
    # Create processor and run
    processor = HISIAIProcessor(
        parent_folder=args.parent_folder,
        output_base_folder=args.output_base,
        file_list=args.file_list
    )
    
    try:
        processor.run()
    except KeyboardInterrupt:
        log_warning("\nProcessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        log_error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
