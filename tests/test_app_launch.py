import subprocess
import time
import requests
import socket
import pytest
import os

def is_port_open(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

@pytest.mark.skipif(os.environ.get("GITHUB_ACTIONS") == "true", reason="Skipping UI test in CI environment")
def test_app_launch():
    """
    Test that the streamlit app can launch and respond to requests.
    """
    port = 8501
    # Ensure port is not already in use
    if is_port_open(port):
        pytest.skip(f"Port {port} is already in use, cannot start test server")

    # Start streamlit app
    process = subprocess.Popen(
        ["streamlit", "run", "app/app.py", "--server.port", str(port), "--server.headless", "true"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    try:
        # Wait for app to start
        max_retries = 30
        success = False
        for i in range(max_retries):
            try:
                response = requests.get(f"http://localhost:{port}")
                if response.status_code == 200:
                    success = True
                    break
            except requests.exceptions.ConnectionError:
                pass
            time.sleep(1)
        
        assert success, "Streamlit app failed to start and respond with 200 OK"
        
    finally:
        # Terminate the process
        process.terminate()
        process.wait()
