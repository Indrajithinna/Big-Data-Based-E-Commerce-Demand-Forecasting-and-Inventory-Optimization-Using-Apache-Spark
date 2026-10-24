# =============================================================================
# Module 2: Big Data Storage
# Script: hdfs_operations.py
# =============================================================================
# PURPOSE:
#   Provides HDFS file operations for the Big Data pipeline.
#   In production, data would be stored on HDFS (Hadoop Distributed File System)
#   for distributed storage across a cluster. In local/demo mode, we use the
#   local filesystem but structure operations identically.
#
# BIS701 RELEVANCE:
#   - Hadoop / HDFS → Distributed file storage foundation
#   - Demonstrates how data is organized for Spark processing
#   - Shows the storage layer of a Big Data architecture
#
# NOTE: This module works in two modes:
#   1. HDFS mode: When Hadoop is installed and HDFS is running
#   2. Local mode: Falls back to local filesystem (for demo/development)
# =============================================================================

import os
import sys
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config_loader import load_config, ensure_directories


class HDFSHandler:
    """
    Handler for HDFS / local filesystem operations.
    
    In a production Big Data cluster, data is stored on HDFS for distributed
    access by Spark workers. For local development and demo, we use the
    local filesystem with the same directory structure.
    
    The key principle demonstrated is: data is organized in a structured
    directory hierarchy (raw → processed → results) that mirrors how
    data would be organized on HDFS in a production environment.
    """
    
    def __init__(self, config=None, use_hdfs=False):
        """
        Initialize storage handler.
        
        Args:
            config: Configuration dictionary.
            use_hdfs: If True, use HDFS commands. If False, use local FS.
        """
        if config is None:
            config = load_config()
        
        self.config = config
        self.use_hdfs = use_hdfs
        self.paths = config["paths"]
        
        # Create directory structure
        ensure_directories(config)
    
    def list_files(self, directory):
        """
        List files in a directory.
        
        Args:
            directory: Path to directory.
        
        Returns:
            list: List of file paths.
        """
        if self.use_hdfs:
            # In HDFS mode, would use: hdfs dfs -ls <directory>
            # For now, fall back to local
            pass
        
        if not os.path.exists(directory):
            return []
        
        files = []
        for f in os.listdir(directory):
            full_path = os.path.join(directory, f)
            if os.path.isfile(full_path):
                size_mb = os.path.getsize(full_path) / (1024 * 1024)
                files.append({
                    "name": f,
                    "path": full_path,
                    "size_mb": round(size_mb, 2)
                })
        return files
    
    def file_exists(self, file_path):
        """Check if a file exists in storage."""
        return os.path.exists(file_path)
    
    def copy_to_storage(self, source, destination):
        """
        Copy a file to the storage layer.
        In HDFS mode, this would use 'hdfs dfs -put'.
        
        Args:
            source: Local source file path.
            destination: Storage destination path.
        """
        if self.use_hdfs:
            # hdfs dfs -put <source> <destination>
            os.system(f"hdfs dfs -put -f {source} {destination}")
        else:
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            shutil.copy2(source, destination)
        
        print(f"[Storage] Copied {source} → {destination}")
    
    def delete_path(self, path):
        """
        Delete a file or directory from storage.
        
        Args:
            path: Path to delete.
        """
        if self.use_hdfs:
            os.system(f"hdfs dfs -rm -r {path}")
        else:
            if os.path.isdir(path):
                shutil.rmtree(path)
            elif os.path.isfile(path):
                os.remove(path)
        
        print(f"[Storage] Deleted: {path}")
    
    def get_storage_info(self):
        """
        Print information about the storage layout.
        
        Returns:
            dict: Storage statistics.
        """
        info = {
            "mode": "HDFS" if self.use_hdfs else "Local Filesystem",
            "directories": {}
        }
        
        for name, path in self.paths.items():
            if os.path.isdir(path):
                files = self.list_files(path)
                total_size = sum(f["size_mb"] for f in files)
                info["directories"][name] = {
                    "path": path,
                    "files": len(files),
                    "total_size_mb": round(total_size, 2)
                }
        
        return info
    
    def print_storage_summary(self):
        """Print a formatted summary of the storage layout."""
        info = self.get_storage_info()
        print(f"\n{'='*60}")
        print(f"Storage Summary (Mode: {info['mode']})")
        print(f"{'='*60}")
        for name, details in info["directories"].items():
            print(f"  {name}:")
            print(f"    Path: {details['path']}")
            print(f"    Files: {details['files']}")
            print(f"    Size: {details['total_size_mb']:.2f} MB")
        print(f"{'='*60}")


if __name__ == "__main__":
    print("=" * 60)
    print("Module 2: HDFS / Storage Operations")
    print("=" * 60)
    
    handler = HDFSHandler()
    handler.print_storage_summary()
