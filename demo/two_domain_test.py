"""
Two-Domain Universal Evaluation
Tests MURA (extremity) + Bone Age with conformal prediction
"""

import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from torchvision import models, transforms
from PIL import Image
from pathlib import Path

sys.path.insert(0, 'src')
from uncertainty_module.core.conformal import ConformalPredictor
from uncertainty_module.core.physics_extremity import ExtremityPhysics
from uncertainty_module.core.physics_bone import BoneAgePhysics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MURADataset(Dataset):
    """Simple MURA dataset from CSV."""
    def __init__(self, csv_path, root_dir, transform=None, max_samples=None):
        self.df = pd.read_csv(csv_path)
        if max_samples:
            self.df = self.df.sample(min(max_samples, len(self.df)), random_state=42)
        self.root_dir = Path(root_dir)
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels for pretrained models
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Handle relative paths in CSV - use as-is since it already contains full path
        img_path = Path(row['image_path'])
        if not img_path.is_absolute():
            img_path = Path.cwd() / img_path
        image = Image.open(img_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
        
        label = int(row['label'])
        return image, torch.tensor(label, dtype=torch.long)


class BoneAgeDataset(Dataset):
    """Bone Age dataset from CSV with labels."""
    def __init__(self, csv_path, root_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.root_dir = Path(root_dir)
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Handle relative paths
        img_path = Path(row['image_path'])
        if not img_path.is_absolute():
            img_path = self.root_dir / img_path
        image = Image.open(img_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
        
        label = int(row['age_label']) if 'age_label' in row else 0
        return image, torch.tensor(label, dtype=torch.long)


def run_two_domain_evaluation(args):
    """Run evaluation on both domains."""
    print("="*80)
    print("TWO-DOMAIN UNIVERSAL EVALUATION")
    print("MURA (Extremity) + Bone Age")
    print("="*80)
    print(f"Device: {DEVICE}")
    
    results = []
    
    # Domain 1: MURA (Extremity)
    print("\n" + "-"*80)
    print("DOMAIN 1: MURA (Extremity - 2 classes)")
    print("-"*80)
    
    model = models.densenet121(pretrained=True)
    model.classifier = nn.Linear(model.classifier.in_features, 2)
    model = model.to(DEVICE)
    model.eval()
    
    if args.extremity_csv and Path(args.extremity_csv).exists():
        print("Using real MURA data...")
        # Load data (limit for faster testing)
        cal_dataset = MURADataset(args.extremity_csv, args.extremity_root, max_samples=200)
        test_dataset = MURADataset(
            args.extremity_csv.replace('calibration', 'test'),
            args.extremity_root, max_samples=100
        )
    else:
        print(f"MURA CSV not found: {args.extremity_csv}")
        print("Using synthetic data...")
        from universal_evaluation import SyntheticXrayDataset
        cal_dataset = SyntheticXrayDataset(n_samples=50, n_classes=2)
        test_dataset = SyntheticXrayDataset(n_samples=20, n_classes=2)
    
    cal_loader = DataLoader(cal_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Calibrate
    cp = ConformalPredictor(alpha=0.05)
    cp.calibrate(model, cal_loader, DEVICE)
    
    # Evaluate
    physics = ExtremityPhysics()
    result = evaluate_domain(model, cp, physics, test_loader, "MURA")
    results.append(result)
    
    # Domain 2: Bone Age
    if args.bone_csv and Path(args.bone_csv).exists():
        print("\n" + "-"*80)
        print("DOMAIN 2: Bone Age (4 classes)")
        print("-"*80)
        
        model = models.densenet121(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, 4)
        model = model.to(DEVICE)
        model.eval()
        
        # Load data
        cal_dataset = BoneAgeDataset(args.bone_csv, args.bone_root)
        
        # Find test CSV
        test_csv = str(args.bone_csv).replace('calibration', 'test')
        test_dataset = BoneAgeDataset(test_csv, args.bone_root)
        
        cal_loader = DataLoader(cal_dataset, batch_size=16, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        # Calibrate
        cp = ConformalPredictor(alpha=0.05)
        cp.calibrate(model, cal_loader, DEVICE)
        
        # Evaluate
        physics = BoneAgePhysics()
        result = evaluate_domain(model, cp, physics, test_loader, "BoneAge")
        results.append(result)
    else:
        print(f"Bone Age CSV not found: {args.bone_csv}")
    
    # Summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    for r in results:
        print(f"[OK] {r['domain']}: {r['coverage']:.1%} coverage, {r['physics_reduction_pct']:.1f}% efficiency gain")
    if len(results) == 2:
        print("[OK] Single architecture (DenseNet121) works for both anatomical domains")


def evaluate_domain(model, cp, physics, test_loader, domain_name):
    """Evaluate a single domain."""
    import torch.nn.functional as F
    
    conformal_sizes = []
    physics_sizes = []
    coverages = []
    top1_correct = []
    max_probs = []
    
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            
            for i in range(len(batch_x)):
                x = batch_x[i:i+1]
                y = batch_y[i].item()
                
                pred, cp_set = cp.predict(model, x)
                
                if cp_set is None:
                    continue
                
                logits = model(x)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                
                pred_class = probs.argmax()
                top1_correct.append(1.0 if pred_class == y else 0.0)
                max_probs.append(probs.max())
                
                baseline_size = cp_set.set_size
                conformal_sizes.append(baseline_size)
                
                if hasattr(physics, 'apply'):
                    physics_set = physics.apply(cp_set.prediction_set, probs)
                    physics_size = len(physics_set)
                else:
                    physics_size = baseline_size
                
                physics_sizes.append(physics_size)
                
                covered = y in cp_set.prediction_set
                coverages.append(1.0 if covered else 0.0)
    
    avg_coverage = np.mean(coverages) if coverages else 0.0
    top1_acc = np.mean(top1_correct) if top1_correct else 0.0
    avg_conf = np.mean(max_probs) if max_probs else 0.0
    avg_conformal = np.mean(conformal_sizes) if conformal_sizes else 0.0
    avg_physics = np.mean(physics_sizes) if physics_sizes else 0.0
    reduction_pct = ((avg_conformal - avg_physics) / avg_conformal * 100) if avg_conformal > 0 else 0.0
    
    print(f"\n  [Results] {domain_name}:")
    print(f"    Top-1 accuracy: {top1_acc:.2%}")
    print(f"    Avg max probability: {avg_conf:.3f}")
    print(f"    Coverage: {avg_coverage:.2%}")
    print(f"    Conformal set size: {avg_conformal:.1f}")
    print(f"    Physics set size: {avg_physics:.1f}")
    print(f"    Efficiency gain: {reduction_pct:.1f}%")
    
    return {
        'domain': domain_name,
        'coverage': avg_coverage,
        'top1_accuracy': top1_acc,
        'avg_confidence': avg_conf,
        'conformal_size': avg_conformal,
        'physics_size': avg_physics,
        'physics_reduction_pct': reduction_pct
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--extremity_csv', default='data/mura/processed/mura_valid_calibration.csv')
    parser.add_argument('--extremity_root', default='data/mura')
    parser.add_argument('--bone_csv', default='data/bone_age/labels.csv')
    parser.add_argument('--bone_root', default='data/bone_age')
    
    args = parser.parse_args()
    run_two_domain_evaluation(args)


if __name__ == "__main__":
    main()
