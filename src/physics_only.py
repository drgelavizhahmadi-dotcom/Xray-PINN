"""
Physics-Only Baseline
Apply physics constraints to standard argmax predictions without conformal calibration.
Shows physics helps even without conformal prediction.
"""
import torch
import torch.nn as nn
from typing import List, Tuple
import numpy as np


class PhysicsOnly:
    """
    Apply physics constraints to baseline argmax predictions.
    Shows physics helps even without conformal calibration.
    """
    def __init__(self, physics_layer):
        self.physics = physics_layer
        self.name = "Baseline_Physics"
        self._is_calibrated = True  # No calibration needed
    
    def predict(self, model: nn.Module, image: torch.Tensor) -> Tuple[List[int], torch.Tensor]:
        """
        Standard forward pass with physics filtering.
        
        Args:
            model: PyTorch model
            image: Input image tensor
            
        Returns:
            filtered_set: Physics-filtered prediction set
            probs: Class probabilities
        """
        model.eval()
        device = next(model.parameters()).device
        
        with torch.no_grad():
            image = image.unsqueeze(0).to(device)
            logits = model(image)
            probs = torch.softmax(logits, dim=1).cpu().squeeze()
            
            # Get top-k predictions (k=2 for binary, k=3 for multi-class)
            num_classes = len(probs)
            k = min(3, num_classes)
            topk = probs.topk(k)
            pred_set = topk.indices.tolist()
            
            # Apply physics filtering
            filtered_set = self.physics.apply(pred_set, probs.numpy())
            
            return filtered_set, probs
    
    def calibrate(self, *args, **kwargs):
        """No-op: physics-only doesn't need calibration."""
        pass
    
    def evaluate(self, model: nn.Module, dataloader, device='cuda') -> dict:
        """
        Evaluate physics-only baseline.
        
        Returns:
            metrics: Dictionary with accuracy, coverage, set_size
        """
        model.eval()
        correct = 0
        total = 0
        coverage_count = 0
        total_set_size = 0
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                
                # Physics filtering
                for i in range(len(images)):
                    # Get top-k
                    num_classes = probs[i].shape[0]
                    k = min(3, num_classes)
                    topk = probs[i].topk(k)
                    pred_set = topk.indices.cpu().tolist()
                    
                    # Apply physics
                    filtered_set = self.physics.apply(pred_set, probs[i].cpu().numpy())
                    
                    # Coverage: is true label in set?
                    if labels[i].item() in filtered_set:
                        coverage_count += 1
                    
                    total_set_size += len(filtered_set)
        
        accuracy = correct / total if total > 0 else 0
        coverage = coverage_count / total if total > 0 else 0
        avg_set_size = total_set_size / total if total > 0 else 0
        
        return {
            'accuracy': accuracy,
            'coverage': coverage,
            'avg_set_size': avg_set_size,
            'method': 'Physics_Only'
        }


class ArgmaxBaseline:
    """Standard argmax baseline (no physics, no conformal)."""
    
    def __init__(self):
        self.name = "Argmax_Baseline"
    
    def predict(self, model: nn.Module, image: torch.Tensor) -> Tuple[List[int], torch.Tensor]:
        """Standard argmax prediction."""
        model.eval()
        device = next(model.parameters()).device
        
        with torch.no_grad():
            image = image.unsqueeze(0).to(device)
            logits = model(image)
            probs = torch.softmax(logits, dim=1).cpu().squeeze()
            
            # Single argmax prediction
            pred = probs.argmax().item()
            return [pred], probs
    
    def evaluate(self, model: nn.Module, dataloader, device='cuda') -> dict:
        """Evaluate standard argmax baseline."""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                _, predicted = outputs.max(1)
                
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        accuracy = correct / total if total > 0 else 0
        
        return {
            'accuracy': accuracy,
            'coverage': accuracy,  # Singleton set, coverage = accuracy
            'avg_set_size': 1.0,
            'method': 'Argmax'
        }


# Factory function
def get_baseline(method: str, physics_layer=None):
    """
    Get baseline method.
    
    Args:
        method: 'argmax' or 'physics_only'
        physics_layer: Physics layer (required for physics_only)
    """
    if method == 'argmax':
        return ArgmaxBaseline()
    elif method == 'physics_only':
        if physics_layer is None:
            raise ValueError("physics_layer required for physics_only method")
        return PhysicsOnly(physics_layer)
    else:
        raise ValueError(f"Unknown method: {method}")
