"""
API endpoint tests.
"""

import pytest
from fastapi.testclient import TestClient

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.main import app


client = TestClient(app)


def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "models_loaded" in data
    assert "device" in data


def test_list_models():
    """Test models listing endpoint."""
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert "id" in data[0]
    assert "owned_by" in data[0]


def test_list_preset_speakers():
    """Test preset speakers endpoint."""
    response = client.get("/v1/speakers")
    assert response.status_code == 200
    data = response.json()
    assert "speakers" in data
    assert "total" in data
    assert data["total"] > 0


def test_list_voices():
    """Test voices listing endpoint."""
    response = client.get("/v1/voices")
    assert response.status_code == 200
    data = response.json()
    assert "voices" in data
    assert "total" in data


def test_root_endpoint():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "version" in data
