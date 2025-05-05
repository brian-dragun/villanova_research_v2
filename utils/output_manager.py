"""
Output Manager Utility

This module provides utilities for efficiently managing, organizing, and cleaning
output files generated during LLM analysis runs.

Features:
- Organized output structure by date and experiment type
- Automatic compression of older results
- Summary tracking of all experiments
- Cleanup utilities for managing disk space
"""

import os
import json
import math
import shutil
import tarfile
import datetime
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('output_manager')

try:
    from colorama import Fore, Style
    COLOR_ENABLED = True
    
    def success_print(message):
        """Print success messages in green."""
        print(Fore.GREEN + f"✅ {message}" + Style.RESET_ALL)
        
    def warning_print(message):
        """Print warning messages in yellow."""
        print(Fore.YELLOW + f"⚠️ {message}" + Style.RESET_ALL)
        
    def error_print(message):
        """Print error messages in red."""
        print(Fore.RED + f"❌ {message}" + Style.RESET_ALL)
        
except ImportError:
    COLOR_ENABLED = False
    
    def success_print(message):
        print(f"✅ {message}")
        
    def warning_print(message):
        print(f"⚠️ {message}")
        
    def error_print(message):
        print(f"❌ {message}")

class OutputManager:
    """Manager for organizing and maintaining output files from LLM analysis runs."""
    
    def __init__(self, base_output_dir: Union[str, Path], 
                 max_recent_runs: int = 5,
                 auto_compress_older: bool = True):
        """
        Initialize the output manager.
        
        Args:
            base_output_dir: Base directory for all outputs
            max_recent_runs: Maximum number of recent runs to keep uncompressed
            auto_compress_older: Automatically compress older runs
        """
        self.base_dir = Path(base_output_dir)
        self.max_recent_runs = max_recent_runs
        self.auto_compress_older = auto_compress_older
        self.index_file = self.base_dir / "output_index.json"
        
        # Create base directory if it doesn't exist
        os.makedirs(self.base_dir, exist_ok=True)
        
        # Load or create output index
        self.output_index = self._load_or_create_index()

    def create_output_dir(self, 
                         experiment_type: str, 
                         model_name: str = None,
                         run_id: str = None) -> Path:
        """
        Create a new output directory with organized structure.
        
        Args:
            experiment_type: Type of experiment (e.g., 'sensitivity', 'ablation', etc.)
            model_name: Name of the model being analyzed
            run_id: Optional run ID (will be generated if None)
            
        Returns:
            Path to the created output directory
        """
        # Generate timestamp and run ID
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if run_id is None:
            run_id = f"run_{timestamp}"
            
        # Create directories
        date_dir = datetime.datetime.now().strftime("%Y-%m-%d")
        
        if model_name:
            # Create path with model name included
            output_dir = self.base_dir / date_dir / experiment_type / model_name / run_id
        else:
            # Create path without model name
            output_dir = self.base_dir / date_dir / experiment_type / run_id
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Update index
        self._update_index(date_dir, experiment_type, model_name, run_id, output_dir)
        
        success_print(f"Created output directory: {output_dir}")
        return output_dir
        
    def compress_old_runs(self, days_threshold: int = 7) -> List[str]:
        """
        Compress older output directories to save space.
        
        Args:
            days_threshold: Compress runs older than this many days
            
        Returns:
            List of compressed archives created
        """
        archives_created = []
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days_threshold)
        
        # Find dated directories older than threshold
        for date_dir in self.base_dir.glob("????-??-??"):
            try:
                dir_date = datetime.datetime.strptime(date_dir.name, "%Y-%m-%d")
                if dir_date < cutoff_date:
                    # Check if already compressed
                    archive_path = self.base_dir / f"{date_dir.name}_outputs.tar.gz"
                    if archive_path.exists():
                        continue
                        
                    # Compress the directory
                    with tarfile.open(archive_path, "w:gz") as tar:
                        tar.add(date_dir, arcname=date_dir.name)
                    
                    archives_created.append(str(archive_path))
                    
                    # Remove the original directory after successful compression
                    shutil.rmtree(date_dir)
                    
                    success_print(f"Compressed old outputs: {archive_path}")
            except (ValueError, tarfile.TarError) as e:
                error_print(f"Error processing directory {date_dir}: {str(e)}")
                
        return archives_created
                
    def clean_outputs(self, 
                     keep_days: int = 30, 
                     keep_compressed_days: int = 90,
                     interactive: bool = True) -> Dict[str, int]:
        """
        Clean up old output files to free disk space.
        
        Args:
            keep_days: Keep uncompressed results newer than this many days
            keep_compressed_days: Keep compressed archives newer than this many days
            interactive: If True, ask for confirmation before deleting
            
        Returns:
            Dict with deleted file counts and freed space
        """
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=keep_days)
        compressed_cutoff = datetime.datetime.now() - datetime.timedelta(days=keep_compressed_days)
        
        to_delete = []
        
        # Find uncompressed dated directories to delete
        for date_dir in self.base_dir.glob("????-??-??"):
            try:
                dir_date = datetime.datetime.strptime(date_dir.name, "%Y-%m-%d")
                if dir_date < cutoff_date:
                    to_delete.append(date_dir)
            except ValueError:
                pass
                
        # Find compressed archives to delete
        for archive in self.base_dir.glob("????-??-??_outputs.tar.gz"):
            try:
                # Extract date from filename
                date_str = archive.name.split("_")[0]
                archive_date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
                if archive_date < compressed_cutoff:
                    to_delete.append(archive)
            except (ValueError, IndexError):
                pass
                
        if not to_delete:
            success_print("No old output files to clean up.")
            return {"deleted_count": 0, "freed_space_bytes": 0}
            
        # Calculate space that will be freed
        total_size = sum(
            os.path.getsize(f) if os.path.isfile(f) else get_dir_size(f)
            for f in to_delete
        )
        
        # Ask for confirmation if interactive
        if interactive:
            print("\nThe following items will be deleted:")
            for item in to_delete:
                print(f"- {item}")
            print(f"\nThis will free approximately {format_size(total_size)}")
            
            confirm = input("\nDelete these items? (y/N): ").lower().strip()
            if confirm != 'y':
                warning_print("Cleanup cancelled.")
                return {"deleted_count": 0, "freed_space_bytes": 0}
                
        # Perform deletion
        deleted_count = 0
        for item in to_delete:
            try:
                if os.path.isfile(item):
                    os.remove(item)
                else:
                    shutil.rmtree(item)
                deleted_count += 1
            except (OSError, shutil.Error) as e:
                error_print(f"Error deleting {item}: {str(e)}")
                
        # Update index after cleaning
        self._clean_index()
        
        success_print(f"Cleaned up {deleted_count} items, freed {format_size(total_size)}")
        return {"deleted_count": deleted_count, "freed_space_bytes": total_size}
        
    def list_recent_outputs(self, limit: int = 10) -> List[Dict]:
        """
        List recent output directories.
        
        Args:
            limit: Maximum number of recent outputs to list
            
        Returns:
            List of output directory information
        """
        if not self.output_index or "outputs" not in self.output_index:
            return []
            
        # Sort outputs by creation timestamp (most recent first)
        sorted_outputs = sorted(
            self.output_index["outputs"], 
            key=lambda x: x.get("timestamp", ""),
            reverse=True
        )
        
        return sorted_outputs[:limit]
        
    def extract_archive(self, archive_path: Union[str, Path], 
                       extract_dir: Optional[Union[str, Path]] = None) -> Optional[Path]:
        """
        Extract a compressed output archive.
        
        Args:
            archive_path: Path to the compressed archive
            extract_dir: Directory to extract to (default: base_dir)
            
        Returns:
            Path to the extracted directory or None if failed
        """
        archive_path = Path(archive_path)
        if not archive_path.exists() or not tarfile.is_tarfile(archive_path):
            error_print(f"Invalid archive: {archive_path}")
            return None
            
        if extract_dir is None:
            extract_dir = self.base_dir
        else:
            extract_dir = Path(extract_dir)
            os.makedirs(extract_dir, exist_ok=True)
            
        try:
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(path=extract_dir)
                
            # Determine extracted directory name
            dir_name = archive_path.stem
            if dir_name.endswith("_outputs"):
                dir_name = dir_name[:-8]  # Remove "_outputs" suffix
                
            extracted_path = extract_dir / dir_name
            success_print(f"Extracted archive to: {extracted_path}")
            return extracted_path
            
        except tarfile.TarError as e:
            error_print(f"Error extracting archive: {str(e)}")
            return None
            
    def get_disk_usage(self) -> Dict:
        """
        Get disk usage information for output directories.
        
        Returns:
            Dict with disk usage information
        """
        total_size = get_dir_size(self.base_dir)
        uncompressed_size = 0
        compressed_size = 0
        
        # Calculate sizes
        for item in os.listdir(self.base_dir):
            path = os.path.join(self.base_dir, item)
            if os.path.isfile(path) and path.endswith(".tar.gz"):
                compressed_size += os.path.getsize(path)
            elif os.path.isdir(path) and item not in [".git", "__pycache__"]:
                uncompressed_size += get_dir_size(path)
                
        return {
            "total_bytes": total_size,
            "total_formatted": format_size(total_size),
            "uncompressed_bytes": uncompressed_size,
            "uncompressed_formatted": format_size(uncompressed_size),
            "compressed_bytes": compressed_size,
            "compressed_formatted": format_size(compressed_size)
        }
        
    def _load_or_create_index(self) -> Dict:
        """Load existing index or create new one."""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                warning_print(f"Error loading index file, creating new one")
                
        # Create new index
        index = {
            "last_updated": datetime.datetime.now().isoformat(),
            "outputs": []
        }
        
        # Save and return
        with open(self.index_file, 'w') as f:
            json.dump(index, f, indent=2)
            
        return index
        
    def _update_index(self, date_dir, experiment_type, model_name, run_id, output_dir) -> None:
        """Update the index with a new output directory."""
        entry = {
            "date": date_dir,
            "experiment_type": experiment_type,
            "model_name": model_name,
            "run_id": run_id,
            "path": str(output_dir),
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        self.output_index["outputs"].append(entry)
        self.output_index["last_updated"] = datetime.datetime.now().isoformat()
        
        # Save updated index
        with open(self.index_file, 'w') as f:
            json.dump(self.output_index, f, indent=2)
            
    def _clean_index(self) -> None:
        """Clean index by removing entries for deleted directories."""
        if not self.output_index or "outputs" not in self.output_index:
            return
            
        # Filter out entries for non-existent paths
        self.output_index["outputs"] = [
            entry for entry in self.output_index["outputs"]
            if os.path.exists(entry["path"])
        ]
        
        self.output_index["last_updated"] = datetime.datetime.now().isoformat()
        
        # Save updated index
        with open(self.index_file, 'w') as f:
            json.dump(self.output_index, f, indent=2)
            
    def generate_summary_report(self, output_file: Optional[str] = None) -> str:
        """
        Generate a summary report of all outputs.
        
        Args:
            output_file: Path to save the report (if None, just returns the report text)
            
        Returns:
            Report text
        """
        report = []
        report.append("# LLM Analysis Output Summary Report")
        report.append(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Add disk usage stats
        usage = self.get_disk_usage()
        report.append("## Disk Usage")
        report.append(f"- Total: {usage['total_formatted']}")
        report.append(f"- Uncompressed: {usage['uncompressed_formatted']}")
        report.append(f"- Compressed: {usage['compressed_formatted']}\n")
        
        # Add recent runs
        recent = self.list_recent_outputs(limit=20)
        report.append("## Recent Runs")
        
        if not recent:
            report.append("No runs recorded yet.\n")
        else:
            for i, run in enumerate(recent):
                report.append(f"{i+1}. {run.get('date')} - {run.get('experiment_type')}")
                if run.get('model_name'):
                    report.append(f"   Model: {run.get('model_name')}")
                report.append(f"   ID: {run.get('run_id')}")
                report.append(f"   Path: {run.get('path')}")
                report.append("")
                
        # Add experiment type statistics
        if recent:
            exp_types = {}
            for run in self.output_index.get("outputs", []):
                exp = run.get("experiment_type", "unknown")
                exp_types[exp] = exp_types.get(exp, 0) + 1
                
            report.append("## Experiment Statistics")
            for exp, count in sorted(exp_types.items(), key=lambda x: x[1], reverse=True):
                report.append(f"- {exp}: {count} runs")
        
        # Join report sections
        report_text = "\n".join(report)
        
        # Save to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            success_print(f"Summary report saved to {output_file}")
            
        return report_text

# Utility functions

def get_dir_size(path: Union[str, Path]) -> int:
    """Get the total size of a directory in bytes."""
    total_size = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size

def format_size(size_bytes: int) -> str:
    """Format size in bytes to human-readable format."""
    if size_bytes == 0:
        return "0 B"
    size_name = ("B", "KB", "MB", "GB", "TB")
    i = int(math.log(size_bytes, 1024))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"