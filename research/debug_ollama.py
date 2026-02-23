import requests
import json

def debug_ollama():
    print("Checking Ollama availability...")
    try:
        # 1. Check basic tags endpoint
        r = requests.get("http://localhost:11434/api/tags")
        if r.status_code == 200:
            print("Successfully connected to Ollama!")
            models = [m['name'] for m in r.json().get('models', [])]
            print(f"Available models: {models}")
        else:
            print(f"Failed to get models: {r.status_code}")
            
        # 2. Check version
        r = requests.get("http://localhost:11434/api/version")
        if r.status_code == 200:
            print(f"Ollama Version: {r.json().get('version')}")
            
    except Exception as e:
        print(f"Error connecting: {e}")

if __name__ == "__main__":
    debug_ollama()
