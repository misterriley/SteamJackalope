#!/usr/bin/env python3
"""Measure memory footprint of just loading the DataManager (without model)."""
import os
import sys
import gc
import psutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def get_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def force_gc():
    gc.collect()
    import time
    time.sleep(0.5)  # Give OS time to release

def main():
    # Record initial Python process memory
    force_gc()
    baseline = get_memory_mb()
    print(f"Initial baseline (Python startup): {baseline:.2f} MB")
    
    # Now import and load DataManager
    print("\nImporting DataManager...")
    from app.server import DataManager
    
    after_import = get_memory_mb()
    print(f"After import (classes defined): {after_import:.2f} MB (+{after_import - baseline:.2f} MB)")
    
    print("\nCreating DataManager instance...")
    dm = DataManager()
    after_instance = get_memory_mb()
    print(f"After instance creation: {after_instance:.2f} MB (+{after_instance - baseline:.2f} MB)")
    
    print("\nCalling load_data()...")
    dm.load_data()
    force_gc()
    after_load = get_memory_mb()
    print(f"After load_data() completes: {after_load:.2f} MB (+{after_load - baseline:.2f} MB)")
    
    # Print metadata info
    if dm.metadata is not None:
        mem_usage = dm.metadata.memory_usage(deep=True).sum() / (1024 * 1024)
        print(f"\nMetadata DataFrame size: {mem_usage:.2f} MB")
        print(f"Metadata shape: {dm.metadata.shape}")
        print(f"Metadata dtypes: {dm.metadata.dtypes.value_counts().to_dict()}")
    
    # Print sizes of memory-mapped arrays (they're mmapped, not fully loaded)
    if dm.embeddings_desc_norm is not None:
        print(f"\nembeddings_desc_norm shape: {dm.embeddings_desc_norm.shape}, dtype: {dm.embeddings_desc_norm.dtype}")
        print(f"  (memory-mapped, not fully loaded into RAM)")
    
    if dm.embeddings_structural_norm is not None:
        print(f"embeddings_structural_norm shape: {dm.embeddings_structural_norm.shape}, dtype: {dm.embeddings_structural_norm.dtype}")
    
    if dm.tag_vectors is not None:
        print(f"tag_vectors shape: {dm.tag_vectors.shape}, dtype: {dm.tag_vectors.dtype}")
    
    if dm.quality_grid is not None:
        print(f"quality_grid shape: {dm.quality_grid.shape}, dtype: {dm.quality_grid.dtype}")
    
    print("\n" + "="*60)
    print(f"TOTAL MEMORY INCREASE from load_data(): {after_load - baseline:.2f} MB")
    print("="*60)
    
    # Keep alive for manual inspection
    print("\nProcess will sleep for 30 seconds for manual inspection...")
    import time
    time.sleep(30)

if __name__ == "__main__":
    main()