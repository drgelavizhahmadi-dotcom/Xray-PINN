"""
MC Dropout with Physics Constraints
Fair comparison: MC Dropout gets physics layer too.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple


class MCDropoutBaseline:
    """
    MC Dropout baseline for uncertainty quantification.
    Uses stochastic forward passes through dropout layers.
    """
    def __init__(self, n_samples=50, dropout_rate=0.1):
        self.n_samples = n_samples
        self.dropout_rate = dropout_rate
        self.name = "MC_Dropout"
    
    def predict(self, model: nn.Module, image: torch.Tensor, device='cuda') -> Tuple[List[int], np.ndarray]:
        """
        MC Dropout prediction with multiple forward passes.
        
        Args:
            model: PyTorch model with dropout layers
            image: Input image tensor
            device: Device to run on
            
        Returns:
            pred_set: Prediction set based on uncertainty
            mean_probs: Mean probabilities across MC samples
        """
        model.train()  # Enable dropout
        device = next(model.parameters()).device
        image = image.unsqueeze(0).to(device)
        
        # Collect predictions from multiple forward passes
        predictions = []
        probs_list = []
        
        with torch.no_grad():
            for _ in range(self.n_samples):
                logits = model(image)
                probs = torch.softmax(logits, dim=1).cpu().squeeze()
                pred = probs.argmax().item()
                
                predictions.append(pred)
                probs_list.append(probs.numpy())
        
        model.eval()  # Return to eval mode
        
        # Calculate mean probabilities
        mean_probs = np.mean(probs_list, axis=0)
        
        # Create prediction set based on uncertainty
        # Include classes with prob > threshold
        threshold = 0.3  # Include if >30% mean probability
        pred_set = [i for i, p in enumerate(mean_probs) if p > threshold]
        
        # Always include top prediction
        top_pred = mean_probs.argmax()
        if top_pred not in pred_set:
            pred_set.append(top_pred)
        
        pred_set.sort()
        return pred_set, mean_probs
    
    def enable_dropout(self, model):
        """Enable dropout layers during inference."""
        for module in model.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                module.train()
    
    def evaluate(self, model: nn.Module, dataloader, device='cuda') -> dict:
        """Evaluate MC Dropout baseline."""
        correct = 0
        total = 0
        coverage_count = 0
        total_set_size = 0
        
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Standard prediction for accuracy
            model.eval()
            with torch.no_grad():
                outputs = model(images)
                _, predicted = outputs.max(1)
                correct += (predicted == labels).sum().item()
            
            # MC Dropout for coverage and set size
            for i in range(len(images)):
                pred_set, _ = self.predict(model, images[i], device)
                
                if labels[i].item() in pred_set:
                    coverage_count += 1
                total_set_size += len(pred_set)
            
            total += labels.size(0)
        
        accuracy = correct / total if total > 0 else 0
        coverage = coverage_count / total if total > 0 else 0
        avg_set_size = total_set_size / total if total > 0 else 0
        
        return {
            'accuracy': accuracy,
            'coverage': coverage,
            'avg_set_size': avg_set_size,
            'method': 'MC_Dropout'
        }


class MCDropoutWithPhysics(MCDropoutBaseline):
    """
    MC Dropout gets the physics layer too—fair fight.
    """
    def __init__(self, physics_layer, n_samples=50):
        super().__init__(n_samples=n_samples)
        self.physics = physics_layer
        self.name = "MC_Dropout_Physics"
    
    def predict(self, model: nn.Module, image: torch.Tensor, device='cuda') -> Tuple[List[int], np.ndarray]:
        # Get MC Dropout predictions
        pred_set, probs = super().predict(model, image, device)
        
        # Apply physics constraints
        filtered_set = self.physics.apply(pred_set, probs)
        
        return filtered_set, probs
    
    def evaluate(self, model: nn.Module, dataloader, device='cuda') -> dict:
        """Evaluate MC Dropout with physics."""
        correct = 0
        total = 0
        coverage_count = 0
        total_set_size = 0
        
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Standard prediction for accuracy
            model.eval()
            with torch.no_grad():
                outputs = model(images)
                _, predicted = outputs.max(1)
                correct += (predicted == labels).sum().item()
            
            # MC Dropout + Physics for coverage and set size
            for i in range(len(images)):
                pred_set, _ = self.predict(model, images[i], device)
                
                if labels[i].item() in pred_set:
                    coverage_count += 1
                total_set_size += len(pred_set)
            
            total += labels.size(0)
        
        accuracy = correct / total if total > 0 else 0
        coverage = coverage_count / total if total > 0 else 0
        avg_set_size = total_set_size / total if total > 0 else 0
        
        return {
            'accuracy': accuracy,
            'coverage': coverage,
            'avg_set_size': avg_set_size,
            'method': 'MC_Dropout_Physics'
        }


# Factory function
def get_mc_dropout(method: str, physics_layer=None, n_samples=50):
    """
    Get MC Dropout method.
    
    Args:
        method: 'mc_dropout' or 'mc_dropout_physics'
        physics_layer: Physics layer (required for mc_dropout_physics)
        n_samples: Number of MC samples
    """
    if method == 'mc_dropout':
        return MCDropoutBaseline(n_samples=n_samples)
    elif method == 'mc_dropout_physics':
        if physics_layer is None:
            raise ValueError("physics_layer required for mc_dropout_physics")
        return MCDropoutWithPhysics(physics_layer, n_samples=n_samples)
    else:
        raise ValueError(f"Unknown method: {method}")
