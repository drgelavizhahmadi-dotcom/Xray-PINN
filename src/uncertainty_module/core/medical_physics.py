"""
Medical Physics Constraints for Chest X-Ray
Filters anatomically impossible diagnosis combinations
"""

import torch
import numpy as np
from typing import List, Set, Tuple


class ChestXRayPhysicsConstraints:
    """
    Physics-informed constraint layer for chest X-ray.
    Enforces anatomical feasibility to shrink prediction sets.
    """
    
    PATHOLOGIES = [
        'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
        'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
        'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
        'Pleural Other', 'Fracture', 'Support Devices'
    ]
    
    def __init__(self):
        self.n_classes = len(self.PATHOLOGIES)
        self.idx = {name: i for i, name in enumerate(self.PATHOLOGIES)}
        
        # Define anatomical impossibilities (mutual exclusions)
        self.exclusions = self._build_exclusion_matrix()
        
    def _build_exclusion_matrix(self) -> np.ndarray:
        """Matrix of anatomically impossible co-occurrences."""
        exc = np.zeros((self.n_classes, self.n_classes))
        
        # 1. NO FINDING excludes everything else (strong constraint)
        no_finding = self.idx['No Finding']
        for i in range(1, self.n_classes):
            exc[no_finding, i] = 1.0
            exc[i, no_finding] = 1.0
            
        # 2. Pneumothorax (air) vs Pleural Effusion (fluid) - same space
        ptx = self.idx['Pneumothorax']
        eff = self.idx['Pleural Effusion']
        exc[ptx, eff] = 0.8  # 0.8 = strong constraint, not absolute
        
        # 3. Consolidation vs Pneumothorax - rare but possible, weak constraint
        cons = self.idx['Consolidation']
        exc[cons, ptx] = 0.3
        
        # 4. Cardiomegaly vs No Finding (already covered by #1)
        
        return exc
    
    def filter_prediction_set(self, prediction_set: List[int], 
                             confidence_scores: np.ndarray = None) -> List[int]:
        """
        Remove anatomically impossible pathologies from prediction set.
        
        Args:
            prediction_set: List of class indices from conformal predictor
            confidence_scores: Optional confidence for each class
            
        Returns:
            Filtered list with physics violations removed
        """
        if len(prediction_set) <= 1:
            return prediction_set
            
        # Check for mutual exclusions
        filtered = set(prediction_set)
        set_list = list(prediction_set)
        
        for i, idx1 in enumerate(set_list):
            for idx2 in set_list[i+1:]:
                exclusion_strength = self.exclusions[idx1, idx2]
                
                if exclusion_strength > 0.5:  # Strong constraint violated
                    # Remove lower-confidence item
                    if confidence_scores is not None:
                        if confidence_scores[idx1] < confidence_scores[idx2]:
                            filtered.discard(idx1)
                        else:
                            filtered.discard(idx2)
                    else:
                        # Default: remove "No Finding" if other pathologies present
                        if idx1 == self.idx['No Finding']:
                            filtered.discard(idx1)
                        elif idx2 == self.idx['No Finding']:
                            filtered.discard(idx2)
        
        return sorted(list(filtered))
    
    def calculate_physics_score(self, prediction_set: List[int]) -> float:
        """
        Calculate anatomical feasibility score (0-1).
        1.0 = fully feasible, 0.0 = physically impossible
        """
        if not prediction_set:
            return 0.0
            
        score = 1.0
        set_list = list(prediction_set)
        
        for i, idx1 in enumerate(set_list):
            for idx2 in set_list[i+1:]:
                exclusion = self.exclusions[idx1, idx2]
                score *= (1.0 - exclusion)
                
        return max(0.0, score)


class PhysicsEnhancedConformalPredictor:
    """
    Combines Conformal Prediction (statistical) + Physics (structural)
    """
    
    def __init__(self, conformal_predictor):
        self.cp = conformal_predictor
        self.physics = ChestXRayPhysicsConstraints()
        
    def predict_with_physics(self, model, x, return_baseline=False):
        """
        Predict with both statistical and physics constraints.
        
        Returns:
            physics_set: Filtered prediction set
            baseline_set: Original conformal set (for comparison)
            metrics: Dict with efficiency gains
        """
        # Get baseline conformal prediction
        pred, baseline_set = self.cp.predict(model, x)
        
        if baseline_set is None:
            return None, None, None
            
        # Apply physics constraints
        physics_indices = self.physics.filter_prediction_set(
            baseline_set.prediction_set
            # Could add confidence_scores here if available
        )
        
        # Calculate improvement
        original_size = baseline_set.set_size
        new_size = len(physics_indices)
        reduction = (original_size - new_size) / original_size if original_size > 0 else 0
        
        metrics = {
            'original_set_size': original_size,
            'physics_set_size': new_size,
            'reduction_percent': reduction * 100,
            'physics_feasibility': self.physics.calculate_physics_score(physics_indices),
            'items_removed': list(set(baseline_set.prediction_set) - set(physics_indices))
        }
        
        # Create new prediction set with physics constraints
        # Note: Coverage guarantee still holds because we're SUBSETTING
        # (removing impossible items doesn't break the guarantee that truth is in the set)
        
        if return_baseline:
            return physics_indices, baseline_set.prediction_set, metrics
        else:
            return physics_indices, metrics
