"""Tests for UncertaintyEngine."""

import numpy as np
import pytest
import torch

from mdss_uncertainty.engine import UncertaintyEngine


@pytest.fixture
def engine():
    """Create engine fixture."""
    return UncertaintyEngine()


@pytest.fixture
def dummy_tensor():
    """Create dummy image tensor."""
    return torch.randn(1, 1, 224, 224)


class TestUncertaintyEngine:
    """Test UncertaintyEngine functionality."""
    
    def test_engine_instantiation(self, engine):
        """Test engine loads model."""
        assert engine.model is not None
        assert engine.pneumonia_idx >= 0
        
    def test_monte_carlo_returns_valid_uncertainty(self, engine, dummy_tensor):
        """Test uncertainty is in [0, 1] range."""
        result = engine.monte_carlo_uncertainty(dummy_tensor, n_samples=5)
        
        assert 0 <= result["epistemic_uncertainty"] <= 1
        assert 0 <= result["mean_confidence"] <= 1
        assert result["ci_lower"] <= result["ci_upper"]
        assert "all_pathologies" in result
        
    def test_monte_carlo_with_3d_input(self, engine):
        """Test with [1, 224, 224] input."""
        tensor = torch.randn(1, 224, 224)
        result = engine.monte_carlo_uncertainty(tensor, n_samples=3)
        
        assert isinstance(result["epistemic_uncertainty"], float)
        
    def test_oversight_policy_low(self, engine):
        """Test LOW risk classification."""
        policy = engine.oversight_policy(uncertainty=0.05, confidence=0.92)
        
        assert policy["risk_level"] == "LOW"
        assert "Proceed" in policy["oversight_action"]
        
    def test_oversight_policy_medium(self, engine):
        """Test MEDIUM risk classification."""
        policy = engine.oversight_policy(uncertainty=0.15, confidence=0.78)
        
        assert policy["risk_level"] == "MEDIUM"
        assert "double reading" in policy["oversight_action"]
        
    def test_oversight_policy_high(self, engine):
        """Test HIGH risk classification."""
        policy = engine.oversight_policy(uncertainty=0.35, confidence=0.55)
        
        assert policy["risk_level"] == "HIGH"
        assert "Mandatory" in policy["oversight_action"]
        assert "Article 14" in policy["article_14_interpretation"]
        
    def test_sanity_check_caps_high_confidence(self, engine):
        """Test confidence > 1 is capped."""
        result = {"mean_confidence": 1.5, "epistemic_uncertainty": 0.1}
        checked = engine.basic_sanity_check(result)
        
        assert checked["mean_confidence"] == 1.0
        assert "NUMERICAL_ANOMALY" in str(checked.get("flags", []))
        
    def test_sanity_check_fixes_nan(self, engine):
        """Test NaN is fixed."""
        result = {"mean_confidence": np.nan, "epistemic_uncertainty": 0.1}
        checked = engine.basic_sanity_check(result)
        
        assert checked["mean_confidence"] == 0.5
        assert not np.isnan(checked["mean_confidence"])
