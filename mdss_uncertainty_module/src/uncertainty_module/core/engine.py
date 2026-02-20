"""Core Uncertainty Oversight Engine for medical AI compliance."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F


class RiskLevel(Enum):
    """Risk levels according to EU AI Act Article 15."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class UncertaintyMetrics:
    """Container for uncertainty metrics."""
    total_uncertainty: float
    aleatoric_uncertainty: float
    epistemic_uncertainty: float
    confidence_score: float
    risk_level: RiskLevel


@dataclass
class OversightResult:
    """Result of oversight check."""
    image_id: str
    prediction: str
    prediction_probability: float
    uncertainty: UncertaintyMetrics
    requires_human_review: bool
    override_reasons: list[str]
    confidence_interval: tuple[float, float]


class UncertaintyOversightEngine:
    """Physics-based uncertainty quantification engine.
    
    Args:
        confidence_threshold: Minimum confidence for autonomous operation
        uncertainty_threshold_low: Threshold for medium risk
        uncertainty_threshold_high: Threshold for high risk
        mc_dropout_samples: Number of Monte Carlo samples
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.7,
        uncertainty_threshold_low: float = 0.10,
        uncertainty_threshold_high: float = 0.25,
        mc_dropout_samples: int = 30,
    ):
        self.confidence_threshold = confidence_threshold
        self.uncertainty_threshold_low = uncertainty_threshold_low
        self.uncertainty_threshold_high = uncertainty_threshold_high
        self.mc_dropout_samples = mc_dropout_samples
        self._model: Optional[torch.nn.Module] = None
        self._class_names: list[str] = []
        
    def register_model(
        self,
        model: torch.nn.Module,
        class_names: list[str],
    ) -> None:
        """Register a PyTorch model for uncertainty estimation."""
        self._model = model
        self._model.eval()
        self._class_names = class_names
        
    def analyze(
        self,
        image: torch.Tensor,
        image_id: str = "unknown",
    ) -> OversightResult:
        """Analyze an image and return oversight decision."""
        if self._model is None:
            raise RuntimeError("No model registered. Call register_model() first.")
            
        # Get MC dropout predictions
        probs = self._mc_dropout_predict(image)
        mean_probs = probs.mean(axis=0)
        
        # Calculate uncertainty metrics
        uncertainty = self._calculate_uncertainty(probs, mean_probs)
        
        # Determine oversight need
        requires_review, reasons = self._determine_oversight_need(uncertainty)
        
        # Get prediction
        pred_idx = int(mean_probs.argmax())
        pred_prob = float(mean_probs[pred_idx])
        
        # Calculate confidence interval
        ci_lower, ci_upper = self._calculate_confidence_interval(probs, pred_idx)
        
        return OversightResult(
            image_id=image_id,
            prediction=self._class_names[pred_idx] if pred_idx < len(self._class_names) else str(pred_idx),
            prediction_probability=pred_prob,
            uncertainty=uncertainty,
            requires_human_review=requires_review,
            override_reasons=reasons,
            confidence_interval=(ci_lower, ci_upper),
        )
        
    def _mc_dropout_predict(self, image: torch.Tensor) -> np.ndarray:
        """Run Monte Carlo dropout predictions."""
        image = image.unsqueeze(0) if image.dim() == 3 else image
        
        predictions = []
        with torch.no_grad():
            for _ in range(self.mc_dropout_samples):
                self._model.train()  # Enable dropout
                output = self._model(image)
                self._model.eval()
                probs = F.softmax(output, dim=1)
                predictions.append(probs.cpu().numpy())
                
        return np.array(predictions).squeeze()
        
    def _calculate_uncertainty(
        self,
        probs: np.ndarray,
        mean_probs: np.ndarray,
    ) -> UncertaintyMetrics:
        """Calculate uncertainty metrics from MC samples."""
        # Total uncertainty: predictive entropy
        entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-10))
        max_entropy = np.log(len(mean_probs))
        total_unc = entropy / max_entropy if max_entropy > 0 else 0
        
        # Aleatoric: expected entropy
        sample_entropies = -np.sum(probs * np.log(probs + 1e-10), axis=1)
        aleatoric = float(sample_entropies.mean()) / max_entropy if max_entropy > 0 else 0
        
        # Epistemic: mutual information
        epistemic = max(0, total_unc - aleatoric)
        
        # Confidence
        confidence = float(mean_probs.max())
        
        # Risk level
        risk = self._determine_risk_level(total_unc, confidence)
        
        return UncertaintyMetrics(
            total_uncertainty=total_unc,
            aleatoric_uncertainty=aleatoric,
            epistemic_uncertainty=epistemic,
            confidence_score=confidence,
            risk_level=risk,
        )
        
    def _determine_risk_level(self, uncertainty: float, confidence: float) -> RiskLevel:
        """Determine risk level based on uncertainty and confidence."""
        if uncertainty > self.uncertainty_threshold_high or confidence < 0.5:
            return RiskLevel.CRITICAL
        elif uncertainty > self.uncertainty_threshold_low or confidence < self.confidence_threshold:
            return RiskLevel.HIGH
        elif uncertainty > 0.05 or confidence < 0.85:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW
        
    def _determine_oversight_need(self, uncertainty: UncertaintyMetrics) -> tuple[bool, list[str]]:
        """Determine if human oversight is required."""
        reasons = []
        
        if uncertainty.confidence_score < self.confidence_threshold:
            reasons.append(f"Confidence {uncertainty.confidence_score:.2f} below threshold {self.confidence_threshold}")
            
        if uncertainty.total_uncertainty > self.uncertainty_threshold_high:
            reasons.append(f"Uncertainty {uncertainty.total_uncertainty:.2f} exceeds high threshold")
        elif uncertainty.total_uncertainty > self.uncertainty_threshold_low:
            reasons.append(f"Uncertainty {uncertainty.total_uncertainty:.2f} exceeds medium threshold")
            
        if uncertainty.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
            reasons.append(f"Risk level: {uncertainty.risk_level.value}")
            
        return len(reasons) > 0, reasons
        
    def _calculate_confidence_interval(
        self,
        probs: np.ndarray,
        pred_idx: int,
        confidence: float = 0.95,
    ) -> tuple[float, float]:
        """Calculate confidence interval for prediction probability."""
        pred_probs = probs[:, pred_idx]
        mean = pred_probs.mean()
        std = pred_probs.std()
        
        margin = 1.96 * std  # 95% CI
        return (max(0, mean - margin), min(1, mean + margin))
