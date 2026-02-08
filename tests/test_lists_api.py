import pytest
from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
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

def test_get_games(client):
    response = client.get("/games")
    assert response.status_code == 200
    games = response.json()
    assert isinstance(games, list)
    assert len(games) > 0

def test_lists_quality(client):
    response = client.get("/lists/quality?discovery_pref=0.0")
    assert response.status_code == 200
    data = response.json()
    assert "top" in data
    assert "bottom" in data
    assert len(data["top"]) <= 50
    assert "quality_score" in data["top"][0]

def test_lists_length(client):
    response = client.get("/lists/length")
    assert response.status_code == 200
    data = response.json()
    assert "top" in data
    assert "bottom" in data
    assert "playtime" in data["top"][0]

def test_lists_popularity(client):
    response = client.get("/lists/popularity")
    assert response.status_code == 200
    data = response.json()
    assert "top" in data
    assert "bottom" in data
    assert "total_reviews" in data["top"][0]

def test_lists_age(client):
    response = client.get("/lists/age")
    assert response.status_code == 200
    data = response.json()
    assert "top" in data
    assert "bottom" in data
    assert "release_date" in data["top"][0]

def test_lists_difficulty(client):
    response = client.get("/lists/difficulty")
    assert response.status_code == 200
    data = response.json()
    assert "top" in data
    assert "bottom" in data
    assert "difficulty_predicted" in data["top"][0]
    assert "tag_impacts" in data

def test_invalid_category(client):
    response = client.get("/lists/invalid")
    assert response.status_code == 400
