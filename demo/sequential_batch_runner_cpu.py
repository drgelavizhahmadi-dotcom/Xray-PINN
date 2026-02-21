"""
Sequential Batch Runner - CPU Optimized
Reduced sample sizes for CPU speed.
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

from uncertainty_module.core.conformal import ConformalPredictor
from uncertainty_module.core.physics_extremity import ExtremityPhysics
from uncertainty_module.core.physics_bone import BoneAgePhysics
from mc_dropout_physics import MCDropoutBaseline, MCDropoutWithPhysics
from three_domain_trinity import UniversalDataset
from torch.utils.data import DataLoader
import torch.nn as nn

DEVICE = torch.device("cpu")
torch.set_num_threads(4)


class BaselineEvaluator:
    """Standard argmax baseline."""
    def __init__(self):
        self.name = "Baseline"
    
    def predict(self, model, image, device):
        model.eval()
        with torch.no_grad():
            image = image.unsqueeze(0).to(device)
            logits = model(image)
            probs = torch.softmax(logits, dim=1).cpu().squeeze()
            pred = probs.argmax().item()
            return [pred], probs.numpy()


class SequentialBatchRunnerCPU:
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
        df.to_csv('results/sequential_batch_results_cpu.csv', index=False)
        self.log(f"[SAVE] Saved: {batch_name} ({len(df)} rows)")
        
    def evaluate(self, model, dataloader, predictor, physics=None):
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
                    
                    # Get prediction from predictor
                    if isinstance(predictor, BaselineEvaluator):
                        pred_set, _ = predictor.predict(model, images[i], self.device)
                    else:
                        # Conformal predictor - needs 4D input [1, C, H, W]
                        img_4d = images[i].unsqueeze(0)  # [1, C, H, W]
                        _, cp_result = predictor.predict(model, img_4d)
                        pred_set = cp_result.prediction_set
                    
                    # Apply physics if provided
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
        try:
            start = time.time()
            
            # Load data with reduced samples
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
                'time_seconds': duration,
                'timestamp': datetime.now().isoformat()
            }
            
            self.log(f"  [OK] {model_name:20} | {dataset_name:10} | Cov: {metrics['coverage']:.1%} | Size: {metrics['set_size']:.2f} | {duration:.1f}s")
            return result
            
        except Exception as e:
            self.log(f"  [ERR] ERROR: {model_name} | {dataset_name} | {str(e)[:50]}")
            return {'batch': method_name, 'model': model_name, 'dataset': dataset_name, 'error': str(e)}
    
    def run_all(self):
        self.log("="*70)
        self.log("SEQUENTIAL BATCH RUNNER - CPU")
        self.log("="*70)
        self.log(f"Device: CPU (threads: 4)")
        self.log(f"MC samples: {self.mc_samples}")
        
        from torchvision import models
        
        for dataset_name, config in self.datasets.items():
            self.log(f"\n--- {dataset_name} ---")
            
            model = models.densenet121(pretrained=True)
            model.classifier = nn.Linear(model.classifier.in_features, config['classes'])
            model = model.to(self.device)
            model.eval()
            
            # 1. Baseline
            baseline = BaselineEvaluator()
            result = self.run_experiment(model, 'densenet121', dataset_name, config, 'Baseline', baseline)
            self.results.append(result)
            gc.collect()
            
            # 2. Physics-Only
            result = self.run_experiment(model, 'densenet121', dataset_name, config, 'Physics_Only', baseline, config['physics'])
            self.results.append(result)
            gc.collect()
            
            # 3. Conformal
            conformal = ConformalPredictor(alpha=0.05)
            result = self.run_experiment(model, 'densenet121', dataset_name, config, 'Conformal', conformal)
            self.results.append(result)
            gc.collect()
            
            # 4. Conformal + Physics
            conformal2 = ConformalPredictor(alpha=0.05)
            result = self.run_experiment(model, 'densenet121', dataset_name, config, 'Conformal_Physics', conformal2, config['physics'])
            self.results.append(result)
            gc.collect()
            
            del model
            gc.collect()
        
        self.save_checkpoint("Complete")
        return pd.DataFrame(self.results)


def main():
    print("="*70)
    print("SEQUENTIAL BATCH RUNNER - CPU OPTIMIZED")
    print("="*70)
    print("Running:")
    print("  - Baseline")
    print("  - Physics-Only")
    print("  - Conformal")
    print("  - Conformal + Physics")
    print("Across 3 datasets with reduced sample sizes")
    print("="*70)
    
    runner = SequentialBatchRunnerCPU(mc_samples=10)
    results = runner.run_all()
    
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    # Pivot for display
    df = pd.DataFrame([r for r in runner.results if 'error' not in r])
    if len(df) > 0:
        summary = df.pivot_table(values='coverage', index='batch', columns='dataset', aggfunc='mean')
        print("\nCoverage by Method and Dataset:")
        print(summary.round(3).to_string())
        
        print("\nSet Size by Method and Dataset:")
        size_summary = df.pivot_table(values='set_size', index='batch', columns='dataset', aggfunc='mean')
        print(size_summary.round(2).to_string())
    
    print("\nSaved to: results/sequential_batch_results_cpu.csv")
    print("="*70)


if __name__ == "__main__":
    main()
