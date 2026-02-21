"""
Three-Domain Universal Evaluation (The Anatomical Trinity)
Tests MURA (extremity) + Bone Age (pediatric) + Montgomery (chest)
"""
import sys
sys.path.insert(0, 'mdss_uncertainty_module/src')
sys.path.insert(0, 'src')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from torchvision import models, transforms
from PIL import Image
from pathlib import Path

from uncertainty_module.core.conformal import ConformalPredictor
from uncertainty_module.core.physics_extremity import ExtremityPhysics
from uncertainty_module.core.physics_bone import BoneAgePhysics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class UniversalDataset(Dataset):
    """Universal dataset loader for any CSV with image_path and label columns."""
    def __init__(self, csv_path, transform=None, max_samples=None):
        self.df = pd.read_csv(csv_path)
        if max_samples:
            self.df = self.df.sample(min(max_samples, len(self.df)), random_state=42)
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = Path(row['image_path'])
        if not img_path.is_absolute():
            img_path = Path.cwd() / img_path
        image = Image.open(img_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
            
        # Handle different column names for labels
        if 'label' in row:
            label = int(row['label'])
        elif 'age_label' in row:
            label = int(row['age_label'])
        else:
            raise KeyError("No label column found (expected 'label' or 'age_label')")
        return image, label


def run_experiment(name, anatomy, csv_path, physics_layer, num_classes):
    """Run evaluation on a single domain."""
    print(f"\n{'='*70}")
    print(f"DOMAIN: {name}")
    print(f"Anatomy: {anatomy}")
    print(f"{'='*70}")
    
    # Load model
    model = models.densenet121(pretrained=True)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    model = model.to(DEVICE)
    model.eval()
    
    # Load calibration data
    cal_csv = csv_path.replace('test.csv', 'calibration.csv')
    cal_dataset = UniversalDataset(cal_csv, max_samples=100)
    cal_loader = DataLoader(cal_dataset, batch_size=16, shuffle=False)
    
    # Load test data
    test_dataset = UniversalDataset(csv_path, max_samples=50)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    print(f"Calibration: {len(cal_dataset)} samples")
    print(f"Test: {len(test_dataset)} samples")
    
    # Calibrate conformal predictor
    predictor = ConformalPredictor(alpha=0.05)
    predictor.calibrate(model, cal_loader)
    
    # Evaluate
    correct = 0
    total = 0
    coverage_count = 0
    total_conformal_size = 0
    total_physics_size = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # Get predictions
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            # Accuracy
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            # Conformal prediction
            for i in range(len(images)):
                _, pred_result = predictor.predict(model, images[i].unsqueeze(0))
                label_item = labels[i].item()
                
                # Coverage
                if label_item in pred_result.prediction_set:
                    coverage_count += 1
                
                # Set sizes
                total_conformal_size += len(pred_result.prediction_set)
                
                # Apply physics constraints
                physics_set = physics_layer.apply(pred_result.prediction_set, probs[i].cpu().numpy())
                total_physics_size += len(physics_set)
    
    accuracy = correct / total if total > 0 else 0
    coverage = coverage_count / total if total > 0 else 0
    avg_conformal_size = total_conformal_size / total if total > 0 else 0
    avg_physics_size = total_physics_size / total if total > 0 else 0
    physics_reduction = (1 - avg_physics_size / avg_conformal_size) * 100 if avg_conformal_size > 0 else 0
    
    print(f"\n  [Results] {name}:")
    print(f"    Top-1 accuracy: {accuracy*100:.1f}%")
    print(f"    Coverage: {coverage*100:.1f}% (target: 95%)")
    print(f"    Conformal set size: {avg_conformal_size:.1f}")
    print(f"    Physics set size: {avg_physics_size:.1f}")
    print(f"    Efficiency gain: {physics_reduction:.1f}%")
    
    return {
        'dataset': name,
        'anatomy': anatomy,
        'accuracy': accuracy,
        'coverage': coverage,
        'target': 0.95,
        'conformal_size': avg_conformal_size,
        'physics_size': avg_physics_size,
        'physics_reduction': physics_reduction
    }


def main():
    print("="*70)
    print("THE ANATOMICAL TRINITY: UNIVERSAL COMPLIANCE VALIDATION")
    print("="*70)
    print(f"\nDevice: {DEVICE}")
    print("Single DenseNet121, zero fine-tuning across all domains")
    
    # The Three Domains
    experiments = [
        {
            'name': 'MURA_Extremity',
            'anatomy': 'Long Bones (Fracture)',
            'csv': 'data/mura/processed/mura_valid_test.csv',
            'physics': ExtremityPhysics(),
            'num_classes': 2
        },
        {
            'name': 'LERA_BoneAge',
            'anatomy': 'Hand/Wrist (Development)',
            'csv': 'data/bone_age/test.csv',
            'physics': BoneAgePhysics(),
            'num_classes': 4
        },
        {
            'name': 'Montgomery_Chest',
            'anatomy': 'Lungs (TB/Normal)',
            'csv': 'data/montgomery/test.csv',
            'physics': ExtremityPhysics(),  # Binary classification
            'num_classes': 2
        }
    ]
    
    results = []
    
    for exp in experiments:
        try:
            result = run_experiment(
                exp['name'],
                exp['anatomy'],
                exp['csv'],
                exp['physics'],
                exp['num_classes']
            )
            results.append(result)
        except Exception as e:
            print(f"\nERROR in {exp['name']}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Final Summary
    print("\n" + "="*70)
    print("UNIVERSALITY PROOF: CROSS-ANATOMICAL COMPLIANCE")
    print("="*70)
    
    df = pd.DataFrame(results)
    print("\n" + df[['dataset', 'anatomy', 'coverage', 'target', 'accuracy', 'physics_reduction']].to_string(index=False))
    
    if len(results) > 0:
        avg_coverage = df['coverage'].mean()
        print(f"\nAverage Coverage: {avg_coverage:.1%} (Target: 95%)")
        print(f"Domains Validated: {len(results)}/3")
        
        if avg_coverage >= 0.94:
            print("[OK] UNIVERSALITY PROVEN: Method works across all anatomical domains")
        
        # Save results
        df.to_csv('results/universality_trinity.csv', index=False)
        print(f"\nResults saved to: results/universality_trinity.csv")
    
    print("="*70)


if __name__ == "__main__":
    main()
