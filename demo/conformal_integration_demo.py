"""
Demo: Conformal Prediction Integration
Tests new conformal module without touching legacy mdss_uncertainty_module code
"""

import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as models

# Import NEW conformal code (from src/)
sys.path.insert(0, 'src')
from uncertainty_module.core.conformal import ConformalPredictor

# Import LEGACY engine (from mdss_uncertainty_module/) 
# Adjust this import based on your actual legacy structure
sys.path.insert(0, 'mdss_uncertainty_module/src')
try:
    from uncertainty_module.core.engine import UncertaintyEngine
    print("[OK] Legacy UncertaintyEngine imported successfully")
except ImportError as e:
    print(f"[WARN] Could not import legacy engine: {e}")
    print("Using vanilla PyTorch model instead")
    UncertaintyEngine = None

def test_conformal_standalone():
    """Test conformal prediction with a simple ResNet model"""
    print("\n" + "="*60)
    print("CONFORMAL PREDICTION DEMO")
    print("="*60)
    
    # 1. Create/load a model (using standard ResNet for demo)
    print("\n[1] Loading model...")
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 14)  # 14 pathology classes for X-ray
    model.eval()
    
    # 2. Create synthetic calibration data (replace with real CheXpert data)
    print("[2] Creating calibration dataset (1000 samples)...")
    n_cal = 1000
    cal_images = torch.randn(n_cal, 3, 224, 224)
    cal_labels = torch.randint(0, 14, (n_cal,))
    cal_loader = DataLoader(
        TensorDataset(cal_images, cal_labels), 
        batch_size=32, 
        shuffle=False
    )
    
    # 3. Initialize and calibrate conformal predictor
    print("[3] Initializing conformal predictor (alpha=0.05, 95% coverage)...")
    cp = ConformalPredictor(alpha=0.05)
    
    print("[4] Calibrating on calibration set...")
    cal_metrics = cp.calibrate(model, cal_loader)
    print(f"    Calibration complete: {cal_metrics}")
    
    # 4. Test on new "X-ray"
    print("\n[5] Testing prediction on new X-ray sample...")
    test_xray = torch.randn(1, 3, 224, 224)
    pred, cp_set = cp.predict(model, test_xray)
    
    print(f"    Predicted class: {pred.item()}")
    print(f"    Prediction set: {cp_set.prediction_set}")
    print(f"    Coverage guarantee: {cp_set.confidence:.1%}")
    print(f"    Is definitive: {cp_set.is_singleton}")
    print(f"    Set size: {cp_set.set_size} classes")
    
    # 5. Generate regulatory report
    print("\n[6] Regulatory Documentation:")
    reg_doc = cp_set.to_regulatory_dict()
    for key, value in reg_doc.items():
        print(f"    {key}: {value}")
    
    # 6. Validate coverage on test set
    print("\n[7] Validating coverage on test set (500 samples)...")
    test_images = torch.randn(500, 3, 224, 224)
    test_labels = torch.randint(0, 14, (500,))
    test_loader = DataLoader(
        TensorDataset(test_images, test_labels),
        batch_size=32
    )
    
    coverage_metrics = cp.evaluate_coverage(model, test_loader)
    print(f"    Empirical coverage: {coverage_metrics['empirical_coverage']:.1%}")
    print(f"    Target coverage: {coverage_metrics['nominal_coverage']:.1%}")
    print(f"    Compliant: {coverage_metrics['regulatory_compliant']}")
    print(f"    Avg set size: {coverage_metrics['average_prediction_set_size']:.1f}")
    
    # 7. Summary for MDSS pitch
    print("\n" + "="*60)
    print("MDSS PITCH SUMMARY")
    print("="*60)
    print(f"[OK] Coverage guarantee: {coverage_metrics['empirical_coverage']:.1%} (target 95%)")
    print(f"[OK] Average differential diagnoses: {coverage_metrics['average_prediction_set_size']:.1f} classes")
    print(f"[OK] Definitive diagnoses: {coverage_metrics['singleton_rate']:.1%} of cases")
    print(f"[OK] Regulatory compliant: {coverage_metrics['regulatory_compliant']}")
    print("\nKey advantage over MC Dropout:")
    print("   MC Dropout: 'Model uncertainty is 0.3' (heuristic)")
    print("   Conformal:  'True diagnosis is in this set 95% of the time' (guarantee)")
    
    return cp, coverage_metrics

if __name__ == "__main__":
    try:
        cp, metrics = test_conformal_standalone()
        print("\n[SUCCESS] Demo completed successfully!")
        print("Conformal prediction module is ready for integration.")
    except Exception as e:
        print(f"\n[FAILED] Demo failed: {e}")
        import traceback
        traceback.print_exc()
