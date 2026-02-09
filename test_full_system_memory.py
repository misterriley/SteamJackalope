#!/usr/bin/env python3
"""
Measure memory footprint of the full system:
- FastAPI backend (app/server.py)
- Streamlit frontend (app/app.py)
Both running simultaneously, with and without model loaded.
"""

import os
import sys
import time
import psutil
import subprocess
from pathlib import Path
import requests

sys.path.insert(0, str(Path(__file__).parent))

def get_total_memory_mb():
    """Get total memory of all python processes in current working directory."""
    total = 0
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_info']):
        try:
            if 'python' in proc.info['name'].lower():
                # Check if it's running our app
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
    print("Full System Memory Test: Backend + Frontend")
    print("="*60)
    
    # Kill any existing processes
    print("\nCleaning up existing processes...")
    subprocess.run("taskkill /F /IM python.exe 2>nul", shell=True)
    time.sleep(2)
    
    # Baseline: Just this script
    baseline = get_total_memory_mb()
    print(f"\nBaseline (this script only): {baseline:.2f} MB")
    
    # Start FastAPI backend
    print("\nStarting FastAPI backend...")
    backend_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.server:app", "--host", "127.0.0.1", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(Path(__file__).parent)
    )
    
    # Wait for backend to be ready
    print("Waiting for backend to start...")
    if wait_for_server("http://127.0.0.1:8000/games"):
        print("Backend is ready!")
    else:
        print("WARNING: Backend may not have started properly")
    
    time.sleep(3)  # Give it extra time to fully initialize
    after_backend = get_total_memory_mb()
    backend_mem = after_backend - baseline
    print(f"Backend memory: {backend_mem:.2f} MB")
    
    # Start Streamlit frontend
    print("\nStarting Streamlit frontend...")
    frontend_proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app/app.py", "--server.port", "8501"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(Path(__file__).parent)
    )
    
    print("Waiting for frontend to start...")
    time.sleep(5)  # Streamlit takes a bit to initialize
    after_frontend = get_total_memory_mb()
    frontend_mem = after_frontend - after_backend
    total_mem = after_frontend - baseline
    
    print(f"Frontend memory: {frontend_mem:.2f} MB")
    print(f"Total system memory (backend + frontend): {total_mem:.2f} MB")
    
    # Test lazy loading: model not loaded yet, just hit an endpoint that doesn't use it
    print("\nTesting lazy loading (no prompt)...")
    try:
        requests.post("http://127.0.0.1:8000/recommend", json={
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
            "top_k": 5,
            "prompt": "",  # No prompt = no model load
            "seed_games": [],
            "genres": []
        }, timeout=10)
        print("Request completed (no model load)")
    except Exception as e:
        print(f"Request failed: {e}")
    
    time.sleep(2)
    after_request_no_model = get_total_memory_mb()
    print(f"Memory after request (no prompt): {after_request_no_model - baseline:.2f} MB total")
    
    # Now trigger model load with a prompt
    print("\nTriggering model load with prompt...")
    try:
        requests.post("http://127.0.0.1:8000/recommend", json={
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
            "top_k": 5,
            "prompt": "action adventure game",  # This loads the model
            "seed_games": [],
            "genres": []
        }, timeout=30)
        print("Request with prompt completed")
    except Exception as e:
        print(f"Request failed: {e}")
    
    time.sleep(3)  # Wait for model to be fully loaded and cached
    after_request_with_model = get_total_memory_mb()
    print(f"Memory after request (with prompt): {after_request_with_model - baseline:.2f} MB total")
    model_load_impact = after_request_with_model - after_request_no_model
    
    print(f"\nModel load impact: +{model_load_impact:.2f} MB")
    
    # Summary
    print("\n" + "="*60)
    print("MEMORY SUMMARY")
    print("="*60)
    print(f"Backend only (no model): ~{backend_mem:.1f} MB")
    print(f"Frontend only: ~{frontend_mem:.1f} MB")
    print(f"Total without model: ~{total_mem:.1f} MB")
    print(f"Total with model loaded: ~{after_request_with_model - baseline:.1f} MB")
    print(f"Model footprint: ~{model_load_impact:.1f} MB")
    
    # Cleanup
    print("\nCleaning up...")
    backend_proc.terminate()
    frontend_proc.terminate()
    time.sleep(2)
    subprocess.run("taskkill /F /IM python.exe 2>nul", shell=True)
    
    print("\nTest complete!")

if __name__ == "__main__":
    main()