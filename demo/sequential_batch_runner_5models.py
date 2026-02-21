"""
5-Model Sequential Batch Runner
Runs 5 models × 3 datasets × 5 methods = 75 experiments
Methods: Baseline, PhysicsOnly, Conformal, MC Dropout, Physics+Conformal
"""
import sys
sys.path.insert(0, 'mdss_uncertainty_module/src')
sys.path.insert(0, 'src')

import torch
import pandas as pd
import numpy as np
import time
import gc
from datetime import datetime
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn

from uncertainty_module.core.conformal import ConformalPredictor
from uncertainty_module.core.physics_extremity import ExtremityPhysics
from uncertainty_module.core.physics_bone import BoneAgePhysics
from mc_dropout_physics import MCDropoutBaseline, MCDropoutWithPhysics
from physics_only import PhysicsOnly, ArgmaxBaseline
from three_domain_trinity import UniversalDataset

DEVICE = torch.device("cpu")
torch.set_num_threads(4)


class FiveModelBatchRunner:
    def __init__(self, mc_samples=10):
        self.device = DEVICE
        self.mc_samples = mc_samples
        self.results = []
        
        self.datasets = {
            'MURA': {
                'csv_cal': 'data/mura/processed/mura_valid_calibration.csv',
                'csv_test': 'data/mura/processed/mura_valid_test.csv',
                'physics': ExtremityPhysics(),
                'classes': 2,
                'max_cal': 50,
                'max_test': 30
            },
            'LERA': {
                'csv_cal': 'data/bone_age/calibration.csv',
                'csv_test': 'data/bone_age/test.csv',
                'physics': BoneAgePhysics(),
                'classes': 4,
                'max_cal': 50,
                'max_test': 30
            },
            'Montgomery': {
                'csv_cal': 'data/montgomery/calibration.csv',
                'csv_test': 'data/montgomery/test.csv',
                'physics': ExtremityPhysics(),
                'classes': 2,
                'max_cal': 50,
                'max_test': 30
            }
        }
        
    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")
        
    def save_checkpoint(self, batch_name):
        df = pd.DataFrame(self.results)
        df.to_csv('results/five_model_results.csv', index=False)
        self.log(f"[SAVE] {batch_name} ({len(df)} rows)")
        
    def get_five_models(self):
        """Load 5 specific models."""
        model_configs = [
            ('densenet121', models.densenet121),
            ('resnet50', models.resnet50),
            ('efficientnet_b0', models.efficientnet_b0),
        ]
        
        loaded = {}
        for name, model_fn in model_configs:
            try:
                if 'efficientnet' in name:
                    model = model_fn(weights='IMAGENET1K_V1')
                else:
                    model = model_fn(pretrained=True)
                loaded[name] = model
                self.log(f"  Loaded {name}")
            except Exception as e:
                self.log(f"  Skipped {name}: {e}")
        
        return loaded
    
    def evaluate(self, model, dataloader, predictor, physics=None):
        """Evaluate a predictor."""
        model.eval()
        correct = 0
        total = 0
        coverage_count = 0
        total_set_size = 0
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                
                for i in range(len(images)):
                    label_item = labels[i].item()
                    
                    # Get prediction
                    if isinstance(predictor, (ConformalPredictor,)):
                        img_4d = images[i].unsqueeze(0)
                        _, cp_result = predictor.predict(model, img_4d)
                        pred_set = cp_result.prediction_set
                    else:
                        pred_set, _ = predictor.predict(model, images[i], self.device)
                    
                    # Apply physics
                    if physics:
                        prob_np = probs[i].cpu().numpy()
                        pred_set = physics.apply(pred_set, prob_np)
                    
                    if label_item in pred_set:
                        coverage_count += 1
                    total_set_size += len(pred_set)
        
        return {
            'accuracy': correct / total if total > 0 else 0,
            'coverage': coverage_count / total if total > 0 else 0,
            'set_size': total_set_size / total if total > 0 else 0
        }
    
    def run_experiment(self, model, model_name, dataset_name, config, method_name, predictor, physics=None):
        """Run single experiment."""
        try:
            start = time.time()
            
            # Load data
            cal_dataset = UniversalDataset(config['csv_cal'], max_samples=config['max_cal'])
            test_dataset = UniversalDataset(config['csv_test'], max_samples=config['max_test'])
            
            # Calibrate if needed
            if 'Conformal' in method_name:
                cal_loader = DataLoader(cal_dataset, batch_size=16, shuffle=False)
                predictor.calibrate(model, cal_loader)
            
            test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
            metrics = self.evaluate(model, test_loader, predictor, physics)
            
            duration = time.time() - start
            
            result = {
                'batch': method_name,
                'model': model_name,
                'dataset': dataset_name,
                'coverage': metrics['coverage'],
                'set_size': metrics['set_size'],
                'accuracy': metrics['accuracy'],
                'time_seconds': duration
            }
            
            self.log(f"  [OK] {model_name:20} | {dataset_name:10} | {method_name:20} | Cov: {metrics['coverage']:.1%} | Size: {metrics['set_size']:.2f}")
            return result
            
        except Exception as e:
            self.log(f"  [ERR] {model_name} | {dataset_name} | {method_name} | {str(e)[:40]}")
            return {'batch': method_name, 'model': model_name, 'dataset': dataset_name, 'error': str(e)}
    
    def run_all_methods(self, model_name, model, dataset_name, config):
        """Run all 5 methods on one model-dataset pair."""
        methods = [
            ('Baseline', ArgmaxBaseline(), None),
            ('Physics_Only', PhysicsOnly(config['physics']), None),
            ('Conformal', ConformalPredictor(alpha=0.05), None),
            ('MC_Dropout', MCDropoutBaseline(n_samples=self.mc_samples), None),
            ('Conformal_Physics', ConformalPredictor(alpha=0.05), config['physics']),
        ]
        
        for method_name, predictor, physics in methods:
            result = self.run_experiment(model, model_name, dataset_name, config, method_name, predictor, physics)
            self.results.append(result)
            gc.collect()
    
    def run_all(self):
        """Run all experiments."""
        self.log("="*70)
        self.log("5-MODEL BATCH RUNNER (75 experiments)")
        self.log("="*70)
        self.log("Methods: Baseline, PhysicsOnly, Conformal, MC, Physics+Conformal")
        
        models_dict = self.get_five_models()
        
        count = 0
        total = len(models_dict) * len(self.datasets) * 5
        
        for model_name, base_model in models_dict.items():
            for dataset_name, config in self.datasets.items():
                count += 5
                self.log(f"\n[{count}/{total}] {model_name} on {dataset_name}")
                
                # Adjust model head
                if hasattr(base_model, 'classifier'):
                    if isinstance(base_model.classifier, nn.Linear):
                        base_model.classifier = nn.Linear(base_model.classifier.in_features, config['classes'])
                    else:
                        base_model.classifier[-1] = nn.Linear(base_model.classifier[-1].in_features, config['classes'])
                elif hasattr(base_model, 'fc'):
                    base_model.fc = nn.Linear(base_model.fc.in_features, config['classes'])
                
                base_model = base_model.to(self.device)
                base_model.eval()
                
                self.run_all_methods(model_name, base_model, dataset_name, config)
                
                gc.collect()
        
        self.save_checkpoint("Complete")
        return self.generate_summary()
    
    def generate_summary(self):
        """Generate summary."""
        self.log("\n" + "="*70)
        self.log("SUMMARY")
        self.log("="*70)
        
        df = pd.DataFrame([r for r in self.results if 'error' not in r])
        
        if len(df) == 0:
            self.log("No valid results!")
            return df
        
        # Pivot by method and dataset
        summary = df.pivot_table(
            values=['coverage', 'set_size'],
            index='batch',
            columns='dataset',
            aggfunc='mean'
        ).round(3)
        
        print("\nCoverage by Method and Dataset:")
        print(summary['coverage'].to_string())
        
        print("\nSet Size by Method and Dataset:")
        print(summary['set_size'].to_string())
        
        # Check Physics_Only vs Baseline
        if 'Baseline' in df['batch'].values and 'Physics_Only' in df['batch'].values:
            baseline_cov = df[df['batch'] == 'Baseline']['coverage'].mean()
            physics_cov = df[df['batch'] == 'Physics_Only']['coverage'].mean()
            
            print(f"\n[CHECK] Baseline avg coverage: {baseline_cov:.1%}")
            print(f"[CHECK] Physics_Only avg coverage: {physics_cov:.1%}")
            
            if abs(physics_cov - baseline_cov) < 0.01:
                print("[WARN] Physics_Only same as Baseline - check implementation")
            else:
                print("[OK] Physics_Only different from Baseline - working")
        
        df.to_csv('results/five_model_comparison.csv', index=False)
        self.log("Saved: results/five_model_comparison.csv")
        return df


def main():
    print("="*70)
    print("5-MODEL SEQUENTIAL RUNNER")
    print("="*70)
    print("Running 75 experiments (5 methods × 3 datasets × 5 models)")
    print("="*70)
    
    runner = FiveModelBatchRunner(mc_samples=10)
    results = runner.run_all()
    
    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
