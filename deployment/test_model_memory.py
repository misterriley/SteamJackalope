#!/usr/bin/env python3
"""
Memory footprint comparison between full SentenceTransformer (PyTorch) and ONNX backend.
Measures baseline, after model load, and after inference memory usage.
"""

import os
import sys
import psutil
import gc
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def get_memory_mb():
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def force_gc():
    """Force garbage collection to get stable memory readings."""
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def test_backend(backend_name, model_kwargs=None):
    """Test memory usage for a specific SentenceTransformer backend."""
    print(f"\n{'='*60}")
    print(f"Testing: {backend_name}")
    print('='*60)
    
    force_gc()
    baseline = get_memory_mb()
    print(f"Baseline memory: {baseline:.2f} MB")
    
    # Import here to ensure fresh state
    from sentence_transformers import SentenceTransformer
    
    model_start = get_memory_mb()
    print(f"Before model load: {model_start:.2f} MB")
    
    # Load model
    if backend_name == 'onnx':
        model = SentenceTransformer(
            'all-MiniLM-L6-v2',
            backend='onnx',
            model_kwargs=model_kwargs or {"file_name": "onnx/model_quint8_avx2.onnx"}
        )
    else:
        # Default PyTorch backend
        model = SentenceTransformer('all-MiniLM-L6-v2')
    
    force_gc()
    after_load = get_memory_mb()
    model_load_increase = after_load - baseline
    print(f"After model load: {after_load:.2f} MB")
    print(f"Model load increase: {model_load_increase:.2f} MB")
    
    # Run inference
    test_sentences = [
        "This is a test sentence for memory measurement.",
        "Another test sentence to ensure model is actually loaded.",
        "Video games are fun and entertaining.",
        "I enjoy playing strategy games with deep mechanics.",
        "The graphics and atmosphere are amazing."
    ] * 10  # 50 sentences total
    
    inference_start = get_memory_mb()
    embeddings = model.encode(test_sentences, convert_to_numpy=True)
    force_gc()
    after_inference = get_memory_mb()
    inference_increase = after_inference - baseline
    
    print(f"After inference: {after_inference:.2f} MB")
    print(f"Total increase from baseline: {inference_increase:.2f} MB")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Dtype: {embeddings.dtype}")
    
    # Clean up
    del model
    del embeddings
    force_gc()
    final = get_memory_mb()
    print(f"After cleanup: {final:.2f} MB")
    print(f"Memory released: {after_inference - final:.2f} MB")
    
    return {
        'backend': backend_name,
        'baseline': baseline,
        'model_load_mb': model_load_increase,
        'total_inference_mb': inference_increase,
        'embedding_shape': str(embeddings.shape) if 'embeddings' in locals() else 'N/A',
        'embedding_dtype': str(embeddings.dtype) if 'embeddings' in locals() else 'N/A'
    }

def main():
    print("="*60)
    print("SentenceTransformer Memory Footprint Comparison")
    print("="*60)
    
    results = []
    
    # Test 1: ONNX backend (quantized)
    try:
        results.append(test_backend('onnx', {"file_name": "onnx/model_quint8_avx2.onnx"}))
    except Exception as e:
        print(f"ERROR testing ONNX: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Full PyTorch model
    try:
        results.append(test_backend('pytorch'))
    except Exception as e:
        print(f"ERROR testing PyTorch: {e}")
        import traceback
        traceback.print_exc()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for r in results:
        print(f"\n{r['backend']}:")
        print(f"  Model load increase: {r['model_load_mb']:.2f} MB")
        print(f"  Total inference increase: {r['total_inference_mb']:.2f} MB")
    
    if len(results) == 2:
        diff = results[1]['total_inference_mb'] - results[0]['total_inference_mb']
        print(f"\nMemory difference (PyTorch - ONNX): {diff:.2f} MB")
        print(f"ONNX is {diff/results[1]['total_inference_mb']*100:.1f}% smaller than PyTorch" if diff > 0 else "PyTorch is smaller")
    
    return results

if __name__ == "__main__":
    main()