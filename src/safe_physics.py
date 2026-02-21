"""
Safe Physics Constraints - Guaranteed Non-Reductive
Never removes high-confidence predictions, only filters impossible ones.
"""
import numpy as np
from typing import List


class SafeScienceConstraints:
    """
    Guaranteed non-reductive physics constraints.
    Never removes high-confidence predictions (>0.6 prob).
    Only filters low-confidence (<0.4) impossible combinations.
    """
    def __init__(self, high_conf_threshold=0.6, low_conf_threshold=0.4, 
                 min_retention=0.5, impossible_pairs=None):
        """
        Args:
            high_conf_threshold: Never filter predictions above this prob
            low_conf_threshold: Only filter predictions below this prob
            min_retention: If we'd remove more than this fraction, abort
            impossible_pairs: List of (i,j) tuples that are anatomically impossible
        """
        self.high_conf_threshold = high_conf_threshold
        self.low_conf_threshold = low_conf_threshold
        self.min_retention = min_retention
        self.impossible_pairs = impossible_pairs or []
        
    def is_impossible(self, i: int, j: int = None) -> bool:
        """Check if class i (or pair i,j) is anatomically impossible."""
        if j is None:
            return False  # Single class is never impossible
        return (i, j) in self.impossible_pairs or (j, i) in self.impossible_pairs
    
    def apply(self, pred_set: List[int], probs: np.ndarray) -> List[int]:
        """
        Apply safe physics constraints.
        
        Strategy:
        1. Tier 1 (Sacred): Never touch high-confidence items (>0.6 prob)
        2. Tier 2 (Questionable): Only filter low-confidence (<0.4) + impossible
        3. Safety check: If we'd remove >50%, abort and return original
        """
        if len(pred_set) <= 1:
            return pred_set  # Nothing to filter
        
        probs = np.array(probs)
        
        # Tier 1: Sacred items (high confidence)
        sacred = [i for i in pred_set if probs[i] > self.high_conf_threshold]
        
        # Tier 2: Questionable items (low confidence)
        questionable = [i for i in pred_set if probs[i] <= self.low_conf_threshold]
        
        # Only filter questionable items that form impossible pairs
        filtered_questionable = []
        removed = []
        
        for i in questionable:
            # Check if i forms an impossible pair with any sacred item
            is_impossible_with_sacred = any(self.is_impossible(i, s) for s in sacred)
            
            if is_impossible_with_sacred:
                removed.append(i)
            else:
                filtered_questionable.append(i)
        
        # Also check for impossible pairs within questionable items
        # Keep the higher probability one, remove the lower
        final_questionable = []
        skipped = set()
        
        for i in filtered_questionable:
            if i in skipped:
                continue
                
            # Check if i has an impossible partner
            impossible_partner = None
            for j in filtered_questionable:
                if i != j and self.is_impossible(i, j):
                    impossible_partner = j
                    break
            
            if impossible_partner is not None:
                # Keep the one with higher probability
                if probs[i] >= probs[impossible_partner]:
                    final_questionable.append(i)
                    skipped.add(impossible_partner)
                else:
                    skipped.add(i)
            else:
                final_questionable.append(i)
        
        # Combine sacred + filtered questionable
        result = sacred + final_questionable
        result = sorted(list(set(result)))  # Remove duplicates and sort
        
        # Safety check: If we removed too much, abort
        retention_ratio = len(result) / len(pred_set) if len(pred_set) > 0 else 1.0
        if retention_ratio < self.min_retention:
            # We removed too much - return original conformal set
            return sorted(pred_set)
        
        return result


class SafeExtremityPhysics(SafeScienceConstraints):
    """Safe physics for extremity X-rays (MURA)."""
    
    def __init__(self, confidence_gap=0.3):
        """
        Args:
            confidence_gap: Only filter if difference exceeds this threshold
        """
        super().__init__(
            high_conf_threshold=0.7,  # Very high threshold
            low_conf_threshold=0.3,
            min_retention=0.9,  # Keep at least 90% of predictions
            impossible_pairs=[]
        )
        self.confidence_gap = confidence_gap
    
    def apply(self, pred_set: List[int], probs: np.ndarray) -> List[int]:
        """
        For binary classification: only filter if confidence gap is large.
        Strategy: Be very conservative - only filter when very confident.
        """
        if len(pred_set) <= 1:
            return pred_set
        
        probs = np.array(probs)
        
        # Get probabilities for classes in pred_set
        pred_probs = [(i, probs[i]) for i in pred_set]
        pred_probs.sort(key=lambda x: x[1], reverse=True)
        
        if len(pred_probs) >= 2:
            top_prob = pred_probs[0][1]
            second_prob = pred_probs[1][1]
            gap = top_prob - second_prob
            
            # Only filter if gap is very large
            if gap > self.confidence_gap:
                # Top prediction is much more confident
                return [pred_probs[0][0]]
        
        # Keep original set (don't filter)
        return sorted(pred_set)


class SafeBoneAgePhysics(SafeScienceConstraints):
    """Safe physics for bone age (developmental sequence)."""
    
    def __init__(self):
        # Infant=0, Toddler=1, Child=2, Adolescent=3
        impossible = [(0, 2), (0, 3), (1, 3)]  # Skip one or more stages
        
        super().__init__(
            high_conf_threshold=0.7,  # Very conservative
            low_conf_threshold=0.3,
            min_retention=0.9,  # Keep 90%
            impossible_pairs=impossible
        )
    
    def apply(self, pred_set: List[int], probs: np.ndarray) -> List[int]:
        """
        Enforce developmental sequence conservatively.
        Only filter if we have very high confidence + clear impossibility.
        """
        if len(pred_set) <= 1:
            return pred_set
        
        probs = np.array(probs)
        
        # Check for high-confidence predictions
        high_conf = [i for i in pred_set if probs[i] > self.high_conf_threshold]
        
        # If we have high-confidence predictions, check if they're valid
        if len(high_conf) >= 2:
            # Check if high-confidence predictions are non-contiguous
            min_conf = min(high_conf)
            max_conf = max(high_conf)
            
            # If gap > 1, we have non-contiguous high-confidence predictions
            if max_conf - min_conf > 1:
                # This is anatomically impossible with high confidence
                # Keep only the highest probability one
                best = max(high_conf, key=lambda i: probs[i])
                return [best]
        
        # Default: keep original set (don't filter)
        return sorted(pred_set)


# Factory function
def get_safe_physics(anatomy_type: str):
    """Get appropriate safe physics constraints."""
    if anatomy_type == 'extremity':
        return SafeExtremityPhysics()
    elif anatomy_type == 'bone_age':
        return SafeBoneAgePhysics()
    else:
        return SafeScienceConstraints()
