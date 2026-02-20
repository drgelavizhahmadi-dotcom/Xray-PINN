"""Conformal Prediction for uncertainty calibration.

Implements conformal prediction methods to provide statistically valid
confidence sets with coverage guarantees for medical AI predictions.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class ConformalSet:
    """Result from conformal prediction."""
    prediction_set: list[str]
    coverage_guarantee: float
    p_values: dict[str, float]
    calibrated: bool


class ConformalPredictor:
    """Split conformal prediction for medical AI.
    
    Provides prediction sets with guaranteed coverage for risk assessment.
    Uses holdout calibration set to determine conformal scores.
    
    Args:
        coverage: Target coverage level (1-alpha), e.g., 0.90 for 90%
        calibration_size: Number of samples for calibration split
    """"
    
    def __init__(
        self,
        coverage: float = 0.90,
        calibration_size: int = 1000,
    ):
        self.coverage = coverage
        self.calibration_size = calibration_size
        self._quantile: Optional[float] = None
        self._calibration_scores: Optional[np.ndarray] = None
        self._class_names: list[str] = []
        
    def fit(
        self,
        probs_cal: np.ndarray,
        labels_cal: np.ndarray,
        class_names: list[str],
    ) -> "ConformalPredictor":
        """Fit conformal predictor on calibration set.
        
        Args:
            probs_cal: Calibration probabilities [n_samples, n_classes]
            labels_cal: True labels (indices) [n_samples]
            class_names: List of class names
        """
        self._class_names = class_names
        
        # Calculate non-conformity scores: 1 - probability of true class
        n_samples = len(labels_cal)
        scores = np.array([
            1.0 - probs_cal[i, labels_cal[i]]
            for i in range(n_samples)
        ])
        
        # Quantile for desired coverage (with finite-sample correction)
        alpha = 1.0 - self.coverage
        quantile_level = np.ceil((n_samples + 1) * (1 - alpha)) / n_samples
        quantile_level = min(quantile_level, 1.0)
        
        self._quantile = np.quantile(scores, quantile_level)
        self._calibration_scores = scores
        
        return self
        
    def predict(
        self,
        probs: np.ndarray,
        return_p_values: bool = False,
    ) -> ConformalSet | list[ConformalSet]:
        """Generate conformal prediction sets.
        
        Args:
            probs: Predicted probabilities [n_samples, n_classes] or [n_classes]
            return_p_values: Whether to return p-values for each class
            
        Returns:
            ConformalSet or list of ConformalSets
        """
        if self._quantile is None:
            raise RuntimeError("Predictor not fitted. Call fit() first.")
            
        single_sample = probs.ndim == 1
        if single_sample:
            probs = probs.reshape(1, -1)
            
        results = []
        for prob in probs:
            # Include classes where score <= quantile
            # Score for class j: 1 - prob[j]
            scores = 1.0 - prob
            inclusion_mask = scores <= self._quantile
            
            pred_set = [
                self._class_names[i]
                for i, include in enumerate(inclusion_mask)
                if include and i < len(self._class_names)
            ]
            
            p_values = None
            if return_p_values:
                # P-value for each class
                p_values = {
                    self._class_names[i]: float(
                        (self._calibration_scores >= scores[i]).mean()
                    )
                    for i in range(len(prob))
                    if i < len(self._class_names)
                }
                
            results.append(ConformalSet(
                prediction_set=pred_set,
                coverage_guarantee=self.coverage,
                p_values=p_values or {},
                calibrated=True,
            ))
            
        return results[0] if single_sample else results
        
    def evaluate_coverage(
        self,
        probs_test: np.ndarray,
        labels_test: np.ndarray,
    ) -> dict:
        """Evaluate empirical coverage on test set.
        
        Returns:
            Dictionary with coverage metrics
        """
        sets = self.predict(probs_test)
        
        correct = 0
        set_sizes = []
        
        for i, cset in enumerate(sets):
            true_label = self._class_names[labels_test[i]]
            if true_label in cset.prediction_set:
                correct += 1
            set_sizes.append(len(cset.prediction_set))
            
        empirical_coverage = correct / len(labels_test)
        
        return {
            "target_coverage": self.coverage,
            "empirical_coverage": empirical_coverage,
            "mean_set_size": np.mean(set_sizes),
            "median_set_size": np.median(set_sizes),
            "empty_sets": sum(1 for s in set_sizes if s == 0),
        }


class AdaptiveConformalPredictor(ConformalPredictor):
    """Conformal predictor with adaptive thresholding.
    
    Adjusts quantile based on difficulty of examples for better efficiency.
    """
    
    def __init__(
        self,
        coverage: float = 0.90,
        calibration_size: int = 1000,
        stratified: bool = True,
    ):
        super().__init__(coverage, calibration_size)
        self.stratified = stratified
        self._quantiles_by_confidence: Optional[dict] = None
        
    def fit(
        self,
        probs_cal: np.ndarray,
        labels_cal: np.ndarray,
        class_names: list[str],
    ) -> "AdaptiveConformalPredictor":
        """Fit with stratification by confidence level."""
        super().fit(probs_cal, labels_cal, class_names)
        
        if not self.stratified:
            return self
            
        # Stratify by max probability (confidence)
        max_probs = probs_cal.max(axis=1)
        bins = np.percentile(max_probs, [33, 66])
        
        self._quantiles_by_confidence = {}
        for bin_idx, (low, high) in enumerate([
            (0, bins[0]), (bins[0], bins[1]), (bins[1], 1.0)
        ]):
            mask = (max_probs >= low) & (max_probs < high)
            if mask.sum() > 10:
                scores_bin = self._calibration_scores[mask]
                alpha = 1.0 - self.coverage
                n = len(scores_bin)
                q_level = np.ceil((n + 1) * (1 - alpha)) / n
                q_level = min(q_level, 1.0)
                self._quantiles_by_confidence[bin_idx] = np.quantile(scores_bin, q_level)
                
        return self
        
    def predict(
        self,
        probs: np.ndarray,
        return_p_values: bool = False,
    ) -> ConformalSet | list[ConformalSet]:
        """Predict with adaptive threshold based on confidence."""
        if self._quantiles_by_confidence is None:
            return super().predict(probs, return_p_values)
            
        single_sample = probs.ndim == 1
        if single_sample:
            probs = probs.reshape(1, -1)
            
        results = []
        for prob in probs:
            max_prob = prob.max()
            
            # Select appropriate quantile
            if max_prob < 0.33:
                quantile = self._quantiles_by_confidence.get(0, self._quantile)
            elif max_prob < 0.66:
                quantile = self._quantiles_by_confidence.get(1, self._quantile)
            else:
                quantile = self._quantiles_by_confidence.get(2, self._quantile)
                
            scores = 1.0 - prob
            inclusion_mask = scores <= quantile
            
            pred_set = [
                self._class_names[i]
                for i, include in enumerate(inclusion_mask)
                if include and i < len(self._class_names)
            ]
            
            p_values = None
            if return_p_values:
                p_values = {
                    self._class_names[i]: float(
                        (self._calibration_scores >= scores[i]).mean()
                    )
                    for i in range(len(prob))
                    if i < len(self._class_names)
                }
                
            results.append(ConformalSet(
                prediction_set=pred_set,
                coverage_guarantee=self.coverage,
                p_values=p_values or {},
                calibrated=True,
            ))
            
        return results[0] if single_sample else results


def calibrate_from_dataset(
    predictor: ConformalPredictor,
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cpu",
) -> ConformalPredictor:
    """Utility to calibrate predictor from a PyTorch DataLoader.
    
    Args:
        predictor: Unfitted conformal predictor
        model: Trained PyTorch model
        dataloader: Calibration data loader
        device: Device for inference
        
    Returns:
        Fitted conformal predictor
    """
    model.eval()
    
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            images, labels = batch[0].to(device), batch[1]
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            
            all_probs.append(probs)
            all_labels.append(labels.numpy() if isinstance(labels, torch.Tensor) else labels)
            
    probs_cal = np.vstack(all_probs)
    labels_cal = np.concatenate(all_labels)
    
    class_names = [f"class_{i}" for i in range(probs_cal.shape[1])]
    
    return predictor.fit(probs_cal, labels_cal, class_names)


if __name__ == "__main__":
    # Test conformal predictor
    print("Testing ConformalPredictor...")
    
    np.random.seed(42)
    n_classes = 14
    n_cal = 500
    n_test = 100
    
    # Generate synthetic calibration data
    probs_cal = np.random.dirichlet(np.ones(n_classes), n_cal)
    labels_cal = np.random.randint(0, n_classes, n_cal)
    
    # Fit predictor
    class_names = [f"Pathology_{i}" for i in range(n_classes)]
    predictor = ConformalPredictor(coverage=0.90)
    predictor.fit(probs_cal, labels_cal, class_names)
    
    # Test prediction
    probs_test = np.random.dirichlet(np.ones(n_classes), 1)[0]
    result = predictor.predict(probs_test, return_p_values=True)
    
    print(f"Prediction set: {result.prediction_set}")
    print(f"Coverage guarantee: {result.coverage_guarantee}")
    print(f"P-values: {result.p_values}")
    
    # Evaluate coverage
    probs_test_batch = np.random.dirichlet(np.ones(n_classes), n_test)
    labels_test = np.random.randint(0, n_classes, n_test)
    metrics = predictor.evaluate_coverage(probs_test_batch, labels_test)
    
    print(f"\nCoverage metrics:")
    print(f"  Target: {metrics['target_coverage']}")
    print(f"  Empirical: {metrics['empirical_coverage']:.3f}")
    print(f"  Mean set size: {metrics['mean_set_size']:.2f}")
