#!/usr/bin/env python3
"""
Measure memory footprint of running backend and frontend.
This script is called by measure_memory.bat after services are started.
"""

import os
import sys
import time
import psutil
import requests
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def get_app_memory_mb():
    """Get total memory of backend and frontend processes."""
    total = 0
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_info']):
        try:
            if 'python' in proc.info['name'].lower():
                cmdline = ' '.join(proc.info['cmdline'] if proc.info['cmdline'] else [])
                if 'app.server' in cmdline or 'streamlit' in cmdline:
                    total += proc.info['memory_info'].rss / 1024 / 1024
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return total

def wait_for_server(url, timeout=30):
    """Wait for server to be responsive."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(url, timeout=2)
            if resp.status_code == 200:
                return True
        except:
            time.sleep(1)
    return False

def main():
    print("="*60)
    print("Memory Measurement")
    print("="*60)
    
    # Wait for backend to be ready
    print("\nChecking backend health...")
    if wait_for_server("http://127.0.0.1:8000/games"):
        print("✓ Backend is responsive")
    else:
        print("✗ Backend not responding")
        return
    
    # Baseline: Services running but model not loaded yet
    time.sleep(2)
    baseline = get_app_memory_mb()
    print(f"\nBaseline (services running, no model): {baseline:.2f} MB")
    
    # Trigger model load with inference
    print("\nTriggering model load via inference request...")
    try:
        response = requests.post("http://127.0.0.1:8000/recommend", json={
            "alpha": 0.5,
            "beta": 0.5,
            "quality_pref": 0.0,
            "age_pref": 0.0,
            "pop_pref": 0.0,
            "disc_pref": 0.0,
            "length_pref": 0.0,
            "difficulty_pref": 0.0,
            "remove_vr": True,
            "english_only": True,
            "remove_nsfw": True,
            "remove_utilities": True,
            "remove_unreleased": True,
            "top_k": 10,
            "prompt": "A relaxing strategy game with deep mechanics",  # Loads model
            "seed_games": [],
            "genres": []
        }, timeout=30)
        
        if response.status_code == 200:
            print("✓ Inference request completed successfully")
        else:
            print(f"✗ Request failed: {response.status_code}")
    except Exception as e:
        print(f"✗ Request error: {e}")
        return
    
    # Wait for model to settle
    print("\nWaiting for memory to stabilize...")
    time.sleep(5)
    
    # Measure peak memory after inference
    peak = get_app_memory_mb()
    print(f"\nPeak memory (after inference): {peak:.2f} MB")
    print(f"Inference impact: +{peak - baseline:.2f} MB")
    
    # Summary
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Backend + Frontend (no model): {baseline:.1f} MB")
    print(f"Backend + Frontend (with model): {peak:.1f} MB")
    print(f"Total memory footprint: {peak:.1f} MB")
    print("="*60)
    
    # Write results to file for later viewing
    with open('memory_results.txt', 'w') as f:
        f.write(f"Baseline (no model): {baseline:.2f} MB\n")
        f.write(f"Peak (with model): {peak:.2f} MB\n")
        f.write(f"Inference impact: {peak - baseline:.2f} MB\n")
        f.write(f"Total footprint: {peak:.2f} MB\n")
    
    print("\nResults saved to memory_results.txt")

if __name__ == "__main__":
    main()