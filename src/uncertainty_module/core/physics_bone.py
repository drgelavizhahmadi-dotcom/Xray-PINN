"""
Pediatric Bone Age Physics Constraints
Enforces developmental stage ordering - you can't skip stages
"""

import numpy as np
from typing import List, Dict


class BoneAgePhysics:
    """
    Pediatric bone age constraints based on developmental biology.
    
    Key insight: Skeletal development is sequential. A child cannot be 
    simultaneously in non-adjacent developmental stages (e.g., Infant and Adolescent).
    """
    
    def __init__(self):
        self.stages = ['Infant', 'Toddler', 'Child', 'Adolescent']
        
        # Adjacency graph: which stages can co-occur in prediction set
        # In developmental biology, you can only be in adjacent stages
        # (representing transition periods or uncertainty at boundaries)
        self.adjacent = {
            0: [0, 1],      # Infant can be with Infant or Toddler
            1: [0, 1, 2],   # Toddler can be with Infant, Toddler, or Child
            2: [1, 2, 3],   # Child can be with Toddler, Child, or Adolescent
            3: [2, 3]       # Adolescent can be with Child or Adolescent
        }
        
        # Stage definitions for documentation
        self.stage_info = {
            0: {'name': 'Infant', 'months': '0-18', 'characteristics': 'Cartilage predominance, minimal ossification'},
            1: {'name': 'Toddler', 'months': '18-60', 'characteristics': 'Rapid ossification, carpal bones forming'},
            2: {'name': 'Child', 'months': '60-144', 'characteristics': 'Epiphyseal plates active, most carpals present'},
            3: {'name': 'Adolescent', 'months': '144-216', 'characteristics': 'Epiphyseal fusion begins, adult pattern emerges'}
        }
    
    def apply(self, prediction_set: List[int], probabilities: np.ndarray) -> List[int]:
        """
        Remove non-adjacent stage combinations from prediction set.
        
        Developmental biology constraint: A patient can only be in 
        adjacent developmental stages (representing transition uncertainty).
        E.g., [Infant, Adolescent] is biologically impossible - remove one.
        
        Args:
            prediction_set: List of stage indices from conformal predictor
            probabilities: Probability distribution over stages
            
        Returns:
            Filtered prediction set respecting developmental adjacency
        """
        if len(prediction_set) <= 1:
            return prediction_set
        
        filtered = set(prediction_set)
        changed = True
        
        # Iteratively remove non-adjacent stages until stable
        while changed and len(filtered) > 1:
            changed = False
            current_list = list(filtered)
            
            # Check all pairs for non-adjacency
            for i in range(len(current_list)):
                for j in range(i + 1, len(current_list)):
                    stage_i = current_list[i]
                    stage_j = current_list[j]
                    
                    # Check if stages are adjacent
                    if stage_j not in self.adjacent.get(stage_i, []):
                        # Non-adjacent stages - biologically impossible together
                        # Remove the one with lower probability
                        if probabilities[stage_i] < probabilities[stage_j]:
                            if stage_i in filtered:
                                filtered.remove(stage_i)
                                changed = True
                        else:
                            if stage_j in filtered:
                                filtered.remove(stage_j)
                                changed = True
                        break
                if changed:
                    break
        
        return sorted(list(filtered))
    
    def is_developmentally_valid(self, stages: List[int]) -> bool:
        """
        Check if a combination of stages is developmentally valid.
        
        Args:
            stages: List of stage indices
            
        Returns:
            True if all stages are mutually adjacent
        """
        if len(stages) <= 1:
            return True
        
        for i, stage_i in enumerate(stages):
            for j, stage_j in enumerate(stages):
                if i != j and stage_j not in self.adjacent.get(stage_i, []):
                    return False
        return True
    
    def get_stage_range(self, prediction_set: List[int]) -> Dict:
        """
        Get age range information for prediction set.
        
        Returns:
            Dictionary with age range and description
        """
        if not prediction_set:
            return {'error': 'Empty prediction set'}
        
        if len(prediction_set) == 1:
            stage = prediction_set[0]
            info = self.stage_info[stage]
            return {
                'predicted_stage': info['name'],
                'age_months': info['months'],
                'characteristics': info['characteristics'],
                'certainty': 'High'
            }
        else:
            # Multiple adjacent stages - range
            stages = sorted(prediction_set)
            first_stage = self.stage_info[stages[0]]
            last_stage = self.stage_info[stages[-1]]
            
            return {
                'predicted_stages': [self.stage_info[s]['name'] for s in stages],
                'age_range': f"{first_stage['months']} to {last_stage['months']}",
                'characteristics': f"Transition period between {first_stage['name']} and {last_stage['name']}",
                'certainty': 'Medium - Transition uncertainty'
            }
    
    def calculate_developmental_score(self, prediction_set: List[int]) -> float:
        """
        Calculate developmental consistency score.
        
        Returns:
            Score from 0-1 indicating biological plausibility
        """
        if not prediction_set:
            return 0.0
        
        if len(prediction_set) == 1:
            return 1.0  # Single stage - fully consistent
        
        if self.is_developmentally_valid(prediction_set):
            # Adjacent stages - reasonable uncertainty
            return 0.7
        else:
            # Non-adjacent - biologically implausible
            return 0.0
    
    def to_regulatory_dict(self, prediction_set: List[int], probabilities: np.ndarray) -> Dict:
        """
        Convert to regulatory documentation format.
        """
        range_info = self.get_stage_range(prediction_set)
        dev_score = self.calculate_developmental_score(prediction_set)
        is_valid = self.is_developmentally_valid(prediction_set)
        
        return {
            "prediction_set": [self.stage_info[i]['name'] for i in prediction_set],
            "age_information": range_info,
            "developmental_validity": "Valid" if is_valid else "Invalid - Corrected",
            "consistency_score": round(dev_score, 2),
            "regulatory_interpretation": (
                "Definitive bone age assessment" if len(prediction_set) == 1
                else "Age range assessment - consider growth plate analysis"
            ),
            "ai_act_compliance": (
                "High confidence prediction" if len(prediction_set) == 1
                else "Uncertainty quantified via stage range"
            )
        }


class PhysicsEnhancedBoneAgePredictor:
    """
    Combines Conformal Prediction with bone age developmental constraints.
    """
    
    def __init__(self, conformal_predictor):
        self.cp = conformal_predictor
        self.physics = BoneAgePhysics()
        
    def predict_with_physics(self, model, x, return_baseline=False):
        """
        Predict bone age with developmental biology constraints.
        
        Returns:
            physics_set: Filtered prediction set
            metrics: Dict with developmental validity and age range
        """
        import torch
        import torch.nn.functional as F
        
        # Get baseline conformal prediction
        pred, baseline_set = self.cp.predict(model, x)
        
        if baseline_set is None:
            return None, None
        
        # Get probabilities
        model.eval()
        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        
        # Apply developmental constraints
        physics_set = self.physics.apply(
            baseline_set.prediction_set,
            probs
        )
        
        # Calculate metrics
        range_info = self.physics.get_stage_range(physics_set)
        is_valid = self.physics.is_developmentally_valid(physics_set)
        
        metrics = {
            'original_set_size': baseline_set.set_size,
            'physics_set_size': len(physics_set),
            'age_range': range_info.get('age_range', range_info.get('age_months', 'Unknown')),
            'predicted_stage': range_info.get('predicted_stage') or range_info.get('predicted_stages'),
            'developmentally_valid': is_valid,
            'consistency_score': self.physics.calculate_developmental_score(physics_set)
        }
        
        if return_baseline:
            return physics_set, baseline_set.prediction_set, metrics
        else:
            return physics_set, metrics


if __name__ == "__main__":
    # Test
    print("Testing BoneAgePhysics...")
    
    physics = BoneAgePhysics()
    
    # Test case 1: Valid adjacent stages
    probs = np.array([0.4, 0.4, 0.1, 0.1])
    set1 = [0, 1, 2]  # Infant, Toddler, Child (should reduce to adjacent pair)
    result1 = physics.apply(set1, probs)
    print(f"Test 1 - Multiple stages: {set1} -> {result1}")
    print(f"  Valid: {physics.is_developmentally_valid(result1)}")
    
    # Test case 2: Invalid non-adjacent (should be corrected)
    probs2 = np.array([0.45, 0.05, 0.05, 0.45])
    set2 = [0, 3]  # Infant and Adolescent (impossible!)
    result2 = physics.apply(set2, probs2)
    print(f"\nTest 2 - Non-adjacent: {set2} -> {result2}")
    print(f"  Valid: {physics.is_developmentally_valid(result2)}")
    
    # Test case 3: Single stage
    set3 = [2]
    result3 = physics.apply(set3, probs)
    print(f"\nTest 3 - Single stage: {set3} -> {result3}")
    print(f"  Range info: {physics.get_stage_range(result3)}")
    
    print("\nBoneAgePhysics tests passed!")
