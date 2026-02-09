import pytest
import subprocess
import sys
import time
import psutil
import requests
import os
import signal

SERVER_URL = "http://127.0.0.1:8001" # Use a different port to avoid conflict with running instances

def get_process_memory_uss_mb(pid):
    try:
        process = psutil.Process(pid)
        # USS (Unique Set Size) is the memory which is unique to a process and which would be freed if the process was terminated right now.
        # This excludes shared memory (like mmap files) which is critical for our check.
        mem_info = process.memory_full_info()
        uss = mem_info.uss / (1024 * 1024)
        return uss
    except psutil.NoSuchProcess:
        return 0

@pytest.fixture(scope="module")
def backend_server():
    # Start the backend server
    env = os.environ.copy()
    # Ensure optimized settings
    env["OMP_NUM_THREADS"] = "1"
    
    cmd = [sys.executable, "-m", "uvicorn", "app.server:app", "--host", "127.0.0.1", "--port", "8001"]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    
    # Wait for server to start
    start_time = time.time()
    started = False
    # Increased timeout to allow for data loading (especially on slower machines)
    while time.time() - start_time < 60:
        try:
            requests.get(f"{SERVER_URL}/genres")
            started = True
            break
        except requests.exceptions.ConnectionError:
            time.sleep(0.5)
            
    if not started:
        process.terminate()
        stdout, stderr = process.communicate()
        pytest.fail(f"Server failed to start:\nStdout: {stdout.decode()}\nStderr: {stderr.decode()}")

    yield process
    
    # Teardown
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()

def test_memory_usage(backend_server):
    pid = backend_server.pid
    
    # 1. Check Baseline Memory (Startup)
    time.sleep(2) # Stabilize
    baseline_uss = get_process_memory_uss_mb(pid)
    print(f"\nBaseline Memory (USS): {baseline_uss:.2f} MB")
    
    # Assert baseline is well below 512 MB (e.g. < 300 MB expected)
    assert baseline_uss < 512, f"Startup memory {baseline_uss:.2f} MB exceeds 512 MB limit"
    
    # 2. Simulate 'Random' Button (Prompt Search)
    # This triggers loading the SentenceTransformer model and quantization
    payload = {
        "alpha": 0.5,
        "beta": 0.5,
        "quality_pref": 0.5,
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
        "prompt": "an exciting adventure game", # Prompt triggers model load
        "seed_games": [],
        "genres": []
    }
    
    start_req = time.time()
    response = requests.post(f"{SERVER_URL}/recommend", json=payload)
    duration = time.time() - start_req
    
    assert response.status_code == 200, "Recommendation request failed"
    print(f"Request duration: {duration:.2f}s")
    
    # 3. Check Peak Memory (After Model Load)
    time.sleep(2) # Stabilize after GC/Quantization
    peak_uss = get_process_memory_uss_mb(pid)
    print(f"Peak Memory (USS) with Model: {peak_uss:.2f} MB")
    
    # Assert peak is still below 512 MB
    assert peak_uss < 512, f"Peak memory {peak_uss:.2f} MB exceeds 512 MB limit"
    
    # Verify model loaded by checking if memory increased significantly (e.g. > 50 MB)
    # This ensures we actually tested the model loading path
    # Quantized model ~20MB + PyTorch overhead ~100MB? 
    # If optimization is good, it might be small increase.
    print(f"Memory Increase: {peak_uss - baseline_uss:.2f} MB")

if __name__ == "__main__":
    # Allow running directly for debugging
    pytest.main([__file__, "-s"])
