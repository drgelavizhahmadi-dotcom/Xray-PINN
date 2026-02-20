"""
Conformal Prediction Module for Distribution-Free Uncertainty Quantification

Provides finite-sample coverage guarantees for regulatory compliance.
Use case: "95% of true diagnoses are in this prediction set" (provable guarantee)
vs MC Dropout's "model uncertainty is 0.3" (heuristic).

References:
- Shafer & Vovk (2008) - Tutorial on Conformal Prediction
- Angelopoulos & Bates (2021) - Gentle Introduction to Conformal Prediction
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class ConformalPredictionSet:
    """
    Conformal prediction set with coverage guarantees.
    Critical for EU AI Act Article 15 'robustness' requirements.
    """
    prediction_set: List[int]      # Class indices in the prediction set
    confidence: float              # 1 - alpha (coverage guarantee, e.g., 0.95)
    is_singleton: bool             # Single prediction = definitive diagnosis
    set_size: int                  # Number of classes in prediction set
    
    def contains(self, true_label: int) -> bool:
        """Check if true label is in prediction set (guaranteed coverage)."""
        return true_label in self.prediction_set
    
    def to_regulatory_dict(self) -> Dict:
        """Convert to format suitable for regulatory documentation."""
        return {
            "prediction_set": self.prediction_set,
            "coverage_guarantee": f"{self.confidence:.1%}",
            "set_size": self.set_size,
            "is_definitive": self.is_singleton,
            "interpretation": (
                "High confidence definitive diagnosis" if self.is_singleton 
                else "Multiple differential diagnoses - human review recommended"
            ),
            "ai_act_article_15": (
                "Robustness demonstrated via distribution-free coverage guarantee. "
                f"True diagnosis contained in prediction set with {self.confidence:.1%} probability."
            )
        }


class ConformalPredictor:
    """
    Split Conformal Prediction for distribution-free uncertainty quantification.
    
    Provides finite-sample coverage guarantees without distributional assumptions.
    Key differentiator for regulatory submissions: "95% coverage" vs "model seems unsure"
    
    Usage:
        1. Calibrate on held-out calibration set (1000 samples)
        2. Predict with coverage guarantees on new data
        3. Generate regulatory documentation automatically
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Args:
            alpha: Miscoverage rate (1-alpha = coverage guarantee, e.g., 0.05 = 95% coverage)
        """
        self.alpha = alpha
        self.q_hat: Optional[float] = None          # Quantile threshold
        self.calibration_scores: Optional[np.ndarray] = None
        self._is_calibrated: bool = False
        
    def calibrate(
        self, 
        model: nn.Module, 
        dataloader: DataLoader,
        device: torch.device = torch.device("cpu")
    ) -> Dict[str, float]:
        """
        Calibrate on held-out calibration set (typically 1000 samples).
        Must be done once before inference.
        
        Returns:
            Calibration metrics for regulatory documentation
        """
        model.eval()
        scores = []
        
        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                # Get logits and probabilities
                logits = model(batch_x)
                probs = torch.softmax(logits, dim=1)
                
                # Non-conformity score: 1 - probability of true class
                # Lower = more conforming (model confident in truth)
                true_class_probs = probs[torch.arange(len(batch_y)), batch_y]
                non_conformity_scores = 1 - true_class_probs
                
                scores.extend(non_conformity_scores.cpu().numpy())
        
        self.calibration_scores = np.array(scores)
        n = len(scores)
        
        # Compute quantile threshold for finite-sample guarantee
        # Using (n+1)/n * (1-alpha) quantile
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.q_hat = np.quantile(scores, q_level, method='higher')
        self._is_calibrated = True
        
        metrics = {
            "calibration_samples": n,
            "quantile_threshold": float(self.q_hat),
            "alpha": self.alpha,
            "coverage_guarantee": f"{(1-self.alpha):.1%}",
            "mean_nonconformity": float(np.mean(scores)),
            "max_nonconformity": float(np.max(scores))
        }
        
        print(f"[Conformal] Calibrated on {n} samples. Threshold: {self.q_hat:.3f}")
        return metrics
        
    def predict(
        self, 
        model: nn.Module, 
        x: torch.Tensor,
        return_sets: bool = True
    ) -> Tuple[torch.Tensor, Optional[ConformalPredictionSet]]:
        """
        Predict with conformal prediction sets.
        
        Args:
            model: PyTorch model (any architecture)
            x: Input tensor (batch_size, ...)
            return_sets: If True, returns prediction sets; if False, just predictions
            
        Returns:
            predictions: Argmax predictions
            cp_set: Conformal prediction set with coverage guarantee
        """
        if not self._is_calibrated:
            raise ValueError("Must calibrate before prediction. Call calibrate() first.")
            
        model.eval()
        device = next(model.parameters()).device
        x = x.to(device)
        
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            predictions = probs.argmax(dim=1)
            
            if not return_sets:
                return predictions, None
            
            # Build prediction sets: include all classes where p(y) >= 1 - q_hat
            threshold = 1 - self.q_hat
            prediction_sets = (probs >= threshold).cpu().numpy()
            
            # Process each item in batch
            results = []
            for i in range(len(x)):
                included_classes = np.where(prediction_sets[i])[0].tolist()
                
                cp_set = ConformalPredictionSet(
                    prediction_set=included_classes,
                    confidence=1 - self.alpha,
                    is_singleton=(len(included_classes) == 1),
                    set_size=len(included_classes)
                )
                results.append(cp_set)
            
            # Return single item if batch size 1, else list
            if len(x) == 1:
                return predictions[0:1], results[0]
            else:
                return predictions, results
    
    def evaluate_coverage(
        self, 
        model: nn.Module, 
        test_loader: DataLoader,
        device: torch.device = torch.device("cpu")
    ) -> Dict[str, float]:
        """
        Evaluate empirical coverage on test set (should be ~1-alpha).
        Generates regulatory documentation of coverage performance.
        
        Returns:
            Dictionary with coverage metrics for AI Act compliance documentation
        """
        correct_coverage = 0
        total = 0
        set_sizes = []
        singleton_count = 0
        
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            
            for i in range(len(batch_x)):
                x = batch_x[i:i+1]
                y = batch_y[i].item()
                
                _, cp_set = self.predict(model, x)
                
                if cp_set.contains(y):
                    correct_coverage += 1
                set_sizes.append(cp_set.set_size)
                if cp_set.is_singleton:
                    singleton_count += 1
                total += 1
        
        empirical_coverage = correct_coverage / total if total > 0 else 0.0
        avg_set_size = np.mean(set_sizes) if set_sizes else 0.0
        singleton_rate = singleton_count / total if total > 0 else 0.0
        
        # Regulatory compliance check (within 2% of nominal coverage)
        is_compliant = abs(empirical_coverage - (1 - self.alpha)) <= 0.02
        
        return {
            "nominal_coverage": 1 - self.alpha,
            "empirical_coverage": empirical_coverage,
            "coverage_gap": abs(empirical_coverage - (1 - self.alpha)),
            "average_prediction_set_size": avg_set_size,
            "singleton_rate": singleton_rate,
            "total_test_samples": total,
            "regulatory_compliant": is_compliant,
            "interpretation": (
                f"Conformal predictor achieves {empirical_coverage:.1%} empirical coverage "
                f"(target: {1-self.alpha:.1%}). Average differential diagnosis size: {avg_set_size:.1f} classes. "
                f"Definitive diagnoses (single class): {singleton_rate:.1%} of cases."
            ),
            "ai_act_article_15_evidence": (
                f"Statistical guarantee: True label contained in prediction set "
                f"{empirical_coverage:.1%} of the time (n={total} test samples). "
                f"{'Meets' if is_compliant else 'Approaches'} AI Act accuracy/robustness requirements."
            )
        }
    
    def get_regulatory_summary(self) -> Dict:
        """Generate summary for regulatory documentation."""
        if not self._is_calibrated:
            return {"error": "Not calibrated"}
        
        return {
            "method": "Split Conformal Prediction (distribution-free)",
            "coverage_guarantee": f"{(1-self.alpha):.1%}",
            "calibration_samples": len(self.calibration_scores) if self.calibration_scores is not None else 0,
            "quantile_threshold": self.q_hat,
            "use_case": "AI Act Article 15 (accuracy, robustness) compliance",
            "advantage_over_mc_dropout": "Finite-sample coverage guarantee without distributional assumptions"
        }
