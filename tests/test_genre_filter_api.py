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

def test_genres_endpoint(client):
    response = client.get("/genres")
    assert response.status_code == 200
    genres = response.json()
    assert isinstance(genres, list)
    assert len(genres) > 0
    assert "Action" in genres or "RPG" in genres or "Indie" in genres

def test_recommend_with_genre_filter(client):
    # Test filtering by a specific genre
    genre_to_test = "RPG"
    payload = {
        "alpha": 1.0,
        "beta": 0.0,
        "quality_pref": 0.5,
        "age_pref": 0.0,
        "pop_pref": 0.0,
        "disc_pref": 0.0,
        "length_pref": 0.0,
        "difficulty_pref": 0.0,
        "price_pref": 0.0,
        "remove_vr": True,
        "english_only": True,
        "remove_nsfw": True,
        "remove_utilities": True,
        "remove_unreleased": True,
        "top_k": 10,
        "prompt": "adventure",
        "seed_games": [],
        "genres": [genre_to_test]
    }
    response = client.post("/recommend", json=payload)
    assert response.status_code == 200
    results = response.json()
    
    # Check that all results have the requested genre
    for game in results:
        # The backend returns 'genres' as it is in metadata (string or list)
        # But for the filter we use 'genres_list' internally.
        # Let's verify based on what the API returns.
        genres = game["genres"]
        if isinstance(genres, str):
            assert genre_to_test.lower() in genres.lower()
        else:
            assert any(genre_to_test.lower() == g.lower() for g in genres)

def test_recommend_with_multiple_genres(client):
    # Test filtering by multiple genres (OR logic)
    genres_to_test = ["RPG", "Strategy"]
    payload = {
        "alpha": 1.0,
        "beta": 0.0,
        "quality_pref": 0.5,
        "age_pref": 0.0,
        "pop_pref": 0.0,
        "disc_pref": 0.0,
        "length_pref": 0.0,
        "difficulty_pref": 0.0,
        "price_pref": 0.0,
        "remove_vr": True,
        "english_only": True,
        "remove_nsfw": True,
        "remove_utilities": True,
        "remove_unreleased": True,
        "top_k": 10,
        "prompt": "battle",
        "seed_games": [],
        "genres": genres_to_test
    }
    response = client.post("/recommend", json=payload)
    assert response.status_code == 200
    results = response.json()
    
    for game in results:
        genres = game["genres"]
        if isinstance(genres, str):
            match = any(g.lower() in genres.lower() for g in genres_to_test)
        else:
            match = any(g.lower() in [x.lower() for x in genres] for g in genres_to_test)
        assert match
