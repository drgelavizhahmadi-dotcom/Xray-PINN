"""
Musculoskeletal Physics Constraints
For MURA and LERA datasets (extremity X-rays)
"""

import numpy as np
from typing import List, Dict


class ExtremityPhysics:
    """
    Musculoskeletal physics constraints for extremity X-rays.
    Much simpler than chest - mainly binary logic with confidence-based filtering.
    """
    
    def __init__(self):
        # MURA/Extremity labels
        self.LABELS = ['Normal', 'Abnormal']
        
        # Hierarchy: Soft tissue changes often accompany fractures
        self.hierarchy = {
            'swelling': ['fracture'],  # If swelling detected, check for fracture
            'effusion': ['fracture'],
        }
        
    def apply(self, prediction_set: List[int], probabilities: np.ndarray, threshold: float = 0.5) -> List[int]:
        """
        Apply physics constraints to MURA/extremity prediction set.
        
        MURA is essentially binary, but we create meaningful prediction sets:
        - Singleton [Normal] or [Abnormal] = High confidence, definitive
        - Set [Normal, Abnormal] = Uncertain (needs human review)
        
        Physics rule: If Abnormal probability > threshold AND 
        confidence is high, remove Normal from the set (can't be both normal and abnormal).
        
        Args:
            prediction_set: List of class indices from conformal predictor
            probabilities: Probability distribution over classes
            threshold: Confidence threshold for decision
            
        Returns:
            Filtered prediction set
        """
        if len(prediction_set) == 1:
            return prediction_set
        
        # MURA binary: 0=Normal, 1=Abnormal
        normal_idx = 0
        abnormal_idx = 1
        
        if normal_idx in prediction_set and abnormal_idx in prediction_set:
            prob_normal = probabilities[normal_idx]
            prob_abnormal = probabilities[abnormal_idx]
            
            # If one is clearly higher, remove the other
            margin = 0.2  # 20% margin for definitive diagnosis
            
            if prob_abnormal > prob_normal + margin:
                # Strong evidence for abnormality
                return [abnormal_idx]
            elif prob_normal > prob_abnormal + margin:
                # Strong evidence for normality
                return [normal_idx]
            # Else keep both (uncertain - needs radiologist review)
        
        return prediction_set
    
    def calculate_uncertainty_score(self, probabilities: np.ndarray) -> float:
        """
        Calculate uncertainty score for extremity X-ray.
        
        Returns:
            Uncertainty score (0 = certain, 1 = completely uncertain)
        """
        # Binary entropy
        p = probabilities[1] if len(probabilities) > 1 else 0.5
        
        # Clamp to avoid log(0)
        p = np.clip(p, 1e-10, 1 - 1e-10)
        
        # Binary entropy: -[p*log(p) + (1-p)*log(1-p)]
        entropy = -(p * np.log(p) + (1 - p) * np.log(1 - p))
        
        # Normalize by max entropy (ln(2))
        return float(entropy / np.log(2))
    
    def needs_human_review(self, prediction_set: List[int], uncertainty_score: float) -> bool:
        """
        Determine if case needs human radiologist review.
        
        Args:
            prediction_set: Current prediction set
            uncertainty_score: Calculated uncertainty score
            
        Returns:
            True if human review recommended
        """
        # Review needed if:
        # 1. Prediction set contains both normal and abnormal (uncertain)
        # 2. Uncertainty score is high (>0.5)
        
        if len(prediction_set) > 1:
            return True
        
        if uncertainty_score > 0.5:
            return True
        
        return False
    
    def to_regulatory_dict(self, prediction_set: List[int], probabilities: np.ndarray) -> Dict:
        """
        Convert to regulatory documentation format.
        """
        uncertainty_score = self.calculate_uncertainty_score(probabilities)
        needs_review = self.needs_human_review(prediction_set, uncertainty_score)
        
        if len(prediction_set) == 1:
            diagnosis = self.LABELS[prediction_set[0]]
            confidence = "High" if not needs_review else "Medium"
        else:
            diagnosis = "Uncertain (Normal vs Abnormal)"
            confidence = "Low - Requires Human Review"
        
        return {
            "prediction_set": [self.LABELS[i] for i in prediction_set],
            "diagnosis": diagnosis,
            "confidence": confidence,
            "uncertainty_score": round(uncertainty_score, 3),
            "requires_human_review": needs_review,
            "ai_act_article_14": "Human oversight triggered" if needs_review else "Automated decision approved",
            "interpretation": (
                "Definitive diagnosis" if len(prediction_set) == 1 and not needs_review
                else "Differential diagnosis - radiologist review recommended"
            )
        }


class PhysicsEnhancedExtremityPredictor:
    """
    Combines Conformal Prediction with extremity physics constraints.
    """
    
    def __init__(self, conformal_predictor):
        self.cp = conformal_predictor
        self.physics = ExtremityPhysics()
        
    def predict_with_physics(self, model, x, return_baseline=False):
        """
        Predict with both statistical and physics constraints.
        
        Returns:
            physics_set: Filtered prediction set
            metrics: Dict with uncertainty scores and review recommendations
        """
        # Get baseline conformal prediction
        pred, baseline_set = self.cp.predict(model, x)
        
        if baseline_set is None:
            return None, None
        
        # Get probabilities from model
        import torch
        import torch.nn.functional as F
        
        model.eval()
        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        
        # Apply physics constraints
        physics_set = self.physics.apply(
            baseline_set.prediction_set,
            probs
        )
        
        # Calculate metrics
        uncertainty_score = self.physics.calculate_uncertainty_score(probs)
        needs_review = self.physics.needs_human_review(physics_set, uncertainty_score)
        
        metrics = {
            'original_set_size': baseline_set.set_size,
            'physics_set_size': len(physics_set),
            'uncertainty_score': uncertainty_score,
            'requires_human_review': needs_review,
            'prob_normal': float(probs[0]),
            'prob_abnormal': float(probs[1]) if len(probs) > 1 else 0.0
        }
        
        if return_baseline:
            return physics_set, baseline_set.prediction_set, metrics
        else:
            return physics_set, metrics


if __name__ == "__main__":
    # Test
    print("Testing ExtremityPhysics...")
    
    physics = ExtremityPhysics()
    
    # Test case 1: High confidence normal
    probs1 = np.array([0.95, 0.05])
    set1 = [0, 1]  # Both normal and abnormal in prediction set
    result1 = physics.apply(set1, probs1)
    print(f"Test 1 - High confidence normal: {result1} (should be [0])")
    
    # Test case 2: High confidence abnormal
    probs2 = np.array([0.1, 0.9])
    set2 = [0, 1]
    result2 = physics.apply(set2, probs2)
    print(f"Test 2 - High confidence abnormal: {result2} (should be [1])")
    
    # Test case 3: Uncertain
    probs3 = np.array([0.45, 0.55])
    set3 = [0, 1]
    result3 = physics.apply(set3, probs3)
    print(f"Test 3 - Uncertain: {result3} (should be [0, 1])")
    
    # Test uncertainty scores
    print(f"\nUncertainty scores:")
    print(f"  Certain (0.95): {physics.calculate_uncertainty_score(np.array([0.05, 0.95])):.3f}")
    print(f"  Uncertain (0.5): {physics.calculate_uncertainty_score(np.array([0.5, 0.5])):.3f}")
    
    print("\nExtremityPhysics tests passed!")
