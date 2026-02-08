import pytest
from fastapi.testclient import TestClient
import os
import sys

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.server import app, data_manager

@pytest.fixture
def client():
    # Ensure data is loaded
    if data_manager.metadata is None:
        data_manager.load_data()
    return TestClient(app)

def test_metadata_endpoint(client):
    # First get a valid game name
    resp = client.get("/games")
    games = resp.json()
    test_game = games[0]
    
    # Test metadata retrieval
    payload = {"names": [test_game]}
    response = client.post("/metadata", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["name"] == test_game
    assert "appid" in data[0]
    assert "tags" in data[0]
    assert "genres" in data[0]

def test_metadata_empty(client):
    payload = {"names": []}
    response = client.post("/metadata", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 0

def test_metadata_invalid_game(client):
    payload = {"names": ["NonExistentGame12345"]}
    response = client.post("/metadata", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 0
