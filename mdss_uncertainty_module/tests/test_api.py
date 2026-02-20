"""Tests for FastAPI endpoints."""

import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from mdss_uncertainty.api import app

client = TestClient(app)


def create_dummy_image():
    """Create dummy X-ray image bytes."""
    img = Image.new("L", (224, 224), color=128)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


class TestHealthEndpoint:
    """Test /health endpoint."""
    
    def test_health_returns_ok(self):
        """Test health check."""
        response = client.get("/health")
        
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert response.json()["model_loaded"] is True


class TestAnalyzeEndpoint:
    """Test /analyze endpoint."""
    
    def test_analyze_without_image(self):
        """Test missing file returns 422."""
        response = client.post("/analyze")
        assert response.status_code == 422
        
    def test_analyze_returns_required_keys(self):
        """Test response has all required fields."""
        img_bytes = create_dummy_image()
        response = client.post(
            "/analyze",
            files={"file": ("test.png", img_bytes, "image/png")}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "filename" in data
        assert "uncertainty" in data
        assert "confidence" in data
        assert "risk_level" in data
        assert "oversight_action" in data
        assert "article_14_status" in data
        
    def test_analyze_uncertainty_in_range(self):
        """Test uncertainty is 0-1."""
        img_bytes = create_dummy_image()
        response = client.post(
            "/analyze",
            files={"file": ("test.png", img_bytes, "image/png")}
        )
        
        data = response.json()
        assert 0 <= data["uncertainty"] <= 1
        assert 0 <= data["confidence"] <= 1
        
    def test_analyze_risk_level_valid(self):
        """Test risk level is valid."""
        img_bytes = create_dummy_image()
        response = client.post(
            "/analyze",
            files={"file": ("test.png", img_bytes, "image/png")}
        )
        
        data = response.json()
        assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]


class TestGenerateDossier:
    """Test /generate-dossier endpoint."""
    
    def test_dossier_without_aggregate_returns_503(self):
        """Test 503 when batch not run."""
        img_bytes = create_dummy_image()
        response = client.post(
            "/generate-dossier",
            files={"file": ("test.png", img_bytes, "image/png")}
        )
        
        assert response.status_code == 503
        assert "batch processor" in response.json()["detail"].lower()
