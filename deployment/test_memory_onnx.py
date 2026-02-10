#!/usr/bin/env python3
"""Test memory footprint using ONNX backend."""
import os
import sys
import gc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def get_memory_mb():
    import psutil
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def main():
    from sentence_transformers import SentenceTransformer
    
    baseline = get_memory_mb()
    print(f"BASELINE: {baseline:.2f} MB")
    
    # Load ONNX model
    print("Loading ONNX model...")
    model = SentenceTransformer(
        'all-MiniLM-L6-v2',
        backend='onnx',
        model_kwargs={"file_name": "onnx/model_quint8_avx2.onnx"}
    )
    
    after_load = get_memory_mb()
    print(f"AFTER LOAD: {after_load:.2f} MB (+{after_load - baseline:.2f} MB)")
    
    # Run inference
    test_sentences = [
        "This is a test sentence for memory measurement.",
        "Another test sentence to ensure model is actually loaded.",
        "Video games are fun and entertaining.",
        "I enjoy playing strategy games with deep mechanics.",
        "The graphics and atmosphere are amazing."
    ] * 20  # 100 sentences
    
    embeddings = model.encode(test_sentences, convert_to_numpy=True)
    after_inference = get_memory_mb()
    print(f"AFTER INFERENCE: {after_inference:.2f} MB (+{after_inference - baseline:.2f} MB)")
    print(f"Embedding shape: {embeddings.shape}, dtype: {embeddings.dtype}")
    
    # Keep process alive for measurement
    print("\n" + "="*60)
    print(f"ONNX TOTAL INCREASE: {after_inference - baseline:.2f} MB")
    print("="*60)
    print("(Process will sleep for 30 seconds for manual memory inspection)")
    
    import time
    time.sleep(30)

if __name__ == "__main__":
    main()