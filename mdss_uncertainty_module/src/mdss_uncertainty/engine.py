"""Physics core: Bayesian MC Dropout uncertainty quantification."""

import numpy as np
import torch
import torch.nn.functional as F
import torchxrayvision as xrv


class UncertaintyEngine:
    """Bayesian uncertainty engine using MC Dropout."""
    
    def __init__(self):
        self.model = xrv.models.DenseNet(weights="densenet121-res224-all")
        self.model.eval()
        self.pathologies = self.model.pathologies
        self.pneumonia_idx = self.pathologies.index("Pneumonia") if "Pneumonia" in self.pathologies else 0
        
    def monte_carlo_uncertainty(self, img_tensor: torch.Tensor, n_samples: int = 30) -> dict:
        """MC Dropout inference. Input: [1, 1, 224, 224] or [1, 224, 224]."""
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)
        
        # xrv expects [batch, channels, height, width]
        if img_tensor.shape[1] != 3 and img_tensor.shape[1] != 1:
            img_tensor = img_tensor.permute(0, 1, 2, 3) if img_tensor.dim() == 4 else img_tensor.unsqueeze(1)
            
        # Ensure correct size
        if img_tensor.shape[2:] != (224, 224):
            img_tensor = F.interpolate(img_tensor, size=(224, 224), mode="bilinear", align_corners=False)
            
        # xrv DenseNet expects 1 channel (grayscale), keep as is
            
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                self.model.train()  # Enable dropout
                out = self.model(img_tensor)
                # xrv returns dict or tensor depending on version
                if isinstance(out, dict):
                    probs = torch.sigmoid(out["out"])
                else:
                    probs = torch.sigmoid(out)
                predictions.append(probs.cpu().numpy())
                
        predictions = np.array(predictions)  # [n_samples, batch, num_classes]
        predictions = predictions.squeeze(1)  # Remove batch dim if present
        
        # Mean prediction across MC samples
        mean_probs = predictions.mean(axis=0)
        
        # Epistemic uncertainty: variance of mean prediction
        epistemic = predictions.var(axis=0).mean()
        
        # Confidence for pneumonia target
        pneumonia_conf = float(mean_probs[self.pneumonia_idx])
        
        # 95% CI using percentiles
        pneumonia_samples = predictions[:, self.pneumonia_idx]
        ci_lower = float(np.percentile(pneumonia_samples, 2.5))
        ci_upper = float(np.percentile(pneumonia_samples, 97.5))
        
        # All pathologies dict
        all_pathologies = {p: float(mean_probs[i]) for i, p in enumerate(self.pathologies)}
        
        result = {
            "mean_confidence": pneumonia_conf,
            "epistemic_uncertainty": float(epistemic),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "all_pathologies": all_pathologies,
        }
        
        return self.basic_sanity_check(result)
        
    def oversight_policy(self, uncertainty: float, confidence: float) -> dict:
        """Map uncertainty/confidence to risk level and oversight action."""
        if uncertainty < 0.10 and confidence > 0.80:
            risk_level = "LOW"
            action = "Proceed with automated read"
            interpretation = "No Article 14 notification required"
        elif uncertainty < 0.25 and confidence > 0.60:
            risk_level = "MEDIUM"
            action = "Encourage double reading"
            interpretation = "Radiologist notification recommended"
        else:
            risk_level = "HIGH"
            action = "Mandatory radiologist review"
            interpretation = "Article 14 human oversight required"
            
        return {
            "risk_level": risk_level,
            "oversight_action": action,
            "article_14_interpretation": interpretation,
        }
        
    def basic_sanity_check(self, result: dict) -> dict:
        """Check for numerical anomalies."""
        flags = []
        
        if np.isnan(result["mean_confidence"]) or np.isnan(result["epistemic_uncertainty"]):
            flags.append("NUMERICAL_ANOMALY: NaN detected")
            result["mean_confidence"] = 0.5 if np.isnan(result["mean_confidence"]) else result["mean_confidence"]
            result["epistemic_uncertainty"] = 0.0 if np.isnan(result["epistemic_uncertainty"]) else result["epistemic_uncertainty"]
            
        if result["epistemic_uncertainty"] < 0:
            flags.append("NUMERICAL_ANOMALY: Negative uncertainty")
            result["epistemic_uncertainty"] = 0.0
            
        if result["mean_confidence"] > 1.0:
            flags.append("NUMERICAL_ANOMALY: Confidence > 1")
            result["mean_confidence"] = 1.0
            
        if flags:
            result["flags"] = flags
            
        return result


if __name__ == "__main__":
    # Test with dummy tensor
    print("Testing UncertaintyEngine...")
    engine = UncertaintyEngine()
    dummy_img = torch.randn(1, 1, 224, 224)
    
    result = engine.monte_carlo_uncertainty(dummy_img, n_samples=10)
    print(f"Mean confidence: {result['mean_confidence']:.4f}")
    print(f"Epistemic uncertainty: {result['epistemic_uncertainty']:.4f}")
    print(f"95% CI: [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
    
    policy = engine.oversight_policy(result['epistemic_uncertainty'], result['mean_confidence'])
    print(f"Risk level: {policy['risk_level']}")
    print(f"Action: {policy['oversight_action']}")
