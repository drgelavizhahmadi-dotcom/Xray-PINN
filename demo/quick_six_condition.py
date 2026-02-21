"""
Quick 6-Condition Comparison (without MC Dropout for speed)
Focus on the key comparisons: Baseline, Physics-Only, Conformal, Conformal+Physics
"""
import sys
sys.path.insert(0, 'mdss_uncertainty_module/src')
sys.path.insert(0, 'src')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from torchvision import models

from uncertainty_module.core.conformal import ConformalPredictor
from uncertainty_module.core.physics_extremity import ExtremityPhysics
from uncertainty_module.core.physics_bone import BoneAgePhysics
from three_domain_trinity import UniversalDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_condition(model, dataloader, condition_name, conformal=None, physics=None):
    """Evaluate a single condition."""
    model.eval()
    correct = 0
    total = 0
    coverage_count = 0
    total_set_size = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            for i in range(len(images)):
                label_item = labels[i].item()
                
                if condition_name == 'Baseline':
                    pred_set = [predicted[i].item()]
                    
                elif condition_name == 'Physics-Only':
                    num_classes = probs[i].shape[0]
                    k = min(3, num_classes)
                    topk = probs[i].topk(k)
                    pred_set = topk.indices.cpu().tolist()
                    if physics:
                        pred_set = physics.apply(pred_set, probs[i].cpu().numpy())
                    
                elif condition_name == 'Conformal':
                    _, cp_result = conformal.predict(model, images[i].unsqueeze(0))
                    pred_set = cp_result.prediction_set
                    
                elif condition_name == 'Conformal+Physics':
                    _, cp_result = conformal.predict(model, images[i].unsqueeze(0))
                    pred_set = cp_result.prediction_set
                    if physics:
                        filtered = physics.apply(pred_set, probs[i].cpu().numpy())
                        pred_set = filtered if len(filtered) > 0 else pred_set
                
                if label_item in pred_set:
                    coverage_count += 1
                total_set_size += len(pred_set)
    
    accuracy = correct / total if total > 0 else 0
    coverage = coverage_count / total if total > 0 else 0
    avg_set_size = total_set_size / total if total > 0 else 0
    
    return {
        'condition': condition_name,
        'accuracy': accuracy,
        'coverage': coverage,
        'set_size': avg_set_size
    }


def run_comparison():
    print("="*70)
    print("4-CONDITION COMPARISON (Key Methods)")
    print("="*70)
    print("\nConditions:")
    print("  1. Baseline: Standard argmax")
    print("  2. Physics-Only: Physics constraints on argmax")
    print("  3. Conformal: Split conformal prediction")
    print("  4. Conformal+Physics: Full method")
    
    datasets = {
        'MURA': {
            'csv_cal': 'data/mura/processed/mura_valid_calibration.csv',
            'csv_test': 'data/mura/processed/mura_valid_test.csv',
            'physics': ExtremityPhysics(),
            'classes': 2
        },
        'LERA': {
            'csv_cal': 'data/bone_age/calibration.csv',
            'csv_test': 'data/bone_age/test.csv',
            'physics': BoneAgePhysics(),
            'classes': 4
        },
        'Montgomery': {
            'csv_cal': 'data/montgomery/calibration.csv',
            'csv_test': 'data/montgomery/test.csv',
            'physics': ExtremityPhysics(),
            'classes': 2
        }
    }
    
    conditions = ['Baseline', 'Physics-Only', 'Conformal', 'Conformal+Physics']
    all_results = []
    
    for dataset_name, config in datasets.items():
        print(f"\n{'='*70}")
        print(f"DATASET: {dataset_name}")
        print(f"{'='*70}")
        
        # Load model
        model = models.densenet121(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, config['classes'])
        model = model.to(DEVICE)
        model.eval()
        
        # Load data
        cal_dataset = UniversalDataset(config['csv_cal'], max_samples=50)
        cal_loader = DataLoader(cal_dataset, batch_size=16, shuffle=False)
        
        test_dataset = UniversalDataset(config['csv_test'], max_samples=30)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        print(f"Calibration: {len(cal_dataset)}, Test: {len(test_dataset)}")
        
        # Calibrate conformal
        conformal = ConformalPredictor(alpha=0.05)
        conformal.calibrate(model, cal_loader)
        
        physics = config['physics']
        
        # Evaluate all conditions
        for condition in conditions:
            result = evaluate_condition(
                model, test_loader, condition,
                conformal if 'Conformal' in condition else None,
                physics if 'Physics' in condition else None
            )
            result['dataset'] = dataset_name
            all_results.append(result)
            print(f"  {condition:20s}: {result['coverage']*100:5.1f}% coverage, {result['set_size']:.2f} set size")
    
    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    
    df = pd.DataFrame(all_results)
    
    # Pivot table
    pivot = df.pivot_table(
        values='coverage',
        index='dataset',
        columns='condition',
        aggfunc='mean'
    ).round(3)
    
    print("\nCoverage by Dataset and Condition:")
    print(pivot.to_string())
    
    # Physics effect
    print("\nPhysics Effect (improvement over baseline):")
    for dataset in df['dataset'].unique():
        base = df[(df['dataset'] == dataset) & (df['condition'] == 'Baseline')]['coverage'].values[0]
        phys = df[(df['dataset'] == dataset) & (df['condition'] == 'Physics-Only')]['coverage'].values[0]
        conf = df[(df['dataset'] == dataset) & (df['condition'] == 'Conformal')]['coverage'].values[0]
        both = df[(df['dataset'] == dataset) & (df['condition'] == 'Conformal+Physics')]['coverage'].values[0]
        
        print(f"  {dataset:12s}: Baseline {base:.1%} → Physics {phys:.1%} (+{phys-base:.1%})")
        print(f"                Baseline {base:.1%} → Conformal {conf:.1%} (+{conf-base:.1%})")
        print(f"                Physics {phys:.1%} + Conformal {conf:.1%} → Both {both:.1%}")
    
    # Set size efficiency
    print("\nSet Size Efficiency:")
    for dataset in df['dataset'].unique():
        conf_size = df[(df['dataset'] == dataset) & (df['condition'] == 'Conformal')]['set_size'].values[0]
        both_size = df[(df['dataset'] == dataset) & (df['condition'] == 'Conformal+Physics')]['set_size'].values[0]
        reduction = (1 - both_size / conf_size) * 100 if conf_size > 0 else 0
        print(f"  {dataset:12s}: Conformal {conf_size:.2f} → Conformal+Physics {both_size:.2f} ({reduction:.1f}% reduction)")
    
    # Save
    df.to_csv('results/quick_comparison.csv', index=False)
    print(f"\nResults saved to: results/quick_comparison.csv")
    
    return df


if __name__ == "__main__":
    run_comparison()
