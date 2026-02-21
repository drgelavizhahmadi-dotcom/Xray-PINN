"""
Sequential Batch Runner for GPU
Runs experiments in batches to manage GPU memory.
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
from physics_only import ArgmaxBaseline
from three_domain_trinity import UniversalDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


class SequentialBatchRunner:
    def __init__(self, device='cuda'):
        self.device = device
        self.results = []
        self.checkpoint_file = 'results/sequential_batch_results.csv'
        
        self.datasets = {
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
        
    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")
        
    def save_checkpoint(self, batch_name):
        df = pd.DataFrame(self.results)
        df.to_csv(self.checkpoint_file, index=False)
        self.log(f"[SAVE] Saved: {batch_name} ({len(df)} rows)")
        
    def run_experiment(self, model, model_name, dataset_name, config, method_name, predictor, physics=None):
        try:
            start = time.time()
            
            # Load data
            cal_dataset = UniversalDataset(config['csv_cal'], max_samples=100)
            test_dataset = UniversalDataset(config['csv_test'], max_samples=50)
            
            # Calibrate if conformal
            if 'Conformal' in method_name and physics is None:
                from torch.utils.data import DataLoader
                cal_loader = DataLoader(cal_dataset, batch_size=16, shuffle=False)
                predictor.calibrate(model, cal_loader)
            
            # Evaluate
            from torch.utils.data import DataLoader
            test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
            
            metrics = self._evaluate(model, test_loader, predictor, physics)
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
            
            self.log(f"  [OK] {model_name[:20]:20} | {dataset_name:10} | Cov: {metrics['coverage']:.1%} | {duration:.1f}s")
            return result
            
        except Exception as e:
            self.log(f"  [ERR] ERROR: {model_name} | {dataset_name} | {str(e)[:50]}")
            return {'batch': method_name, 'model': model_name, 'dataset': dataset_name, 'error': str(e)}
    
    def _evaluate(self, model, dataloader, predictor, physics):
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
                    pred_set, _ = predictor.predict(model, images[i], self.device)
                    
                    if physics:
                        pred_set = physics.apply(pred_set, probs[i].cpu().numpy())
                    
                    if label_item in pred_set:
                        coverage_count += 1
                    total_set_size += len(pred_set)
        
        return {
            'accuracy': correct / total if total > 0 else 0,
            'coverage': coverage_count / total if total > 0 else 0,
            'set_size': total_set_size / total if total > 0 else 0
        }
    
    def run_all(self):
        self.log("STARTING BATCH PROCESSING")
        self.log(f"Device: {self.device}")
        
        # For simplicity, run with DenseNet121 only
        from torchvision import models
        import torch.nn as nn
        
        for dataset_name, config in self.datasets.items():
            model = models.densenet121(pretrained=True)
            model.classifier = nn.Linear(model.classifier.in_features, config['classes'])
            model = model.to(self.device)
            model.eval()
            
            # Baseline
            baseline = BaselineEvaluator()
            result = self.run_experiment(model, 'densenet121', dataset_name, config, 'Baseline', baseline)
            self.results.append(result)
            
            # Conformal
            conformal = ConformalPredictor(alpha=0.05)
            result = self.run_experiment(model, 'densenet121', dataset_name, config, 'Conformal_Vanilla', conformal)
            self.results.append(result)
            
            # Conformal + Physics
            conformal2 = ConformalPredictor(alpha=0.05)
            result = self.run_experiment(model, 'densenet121', dataset_name, config, 'Conformal_Physics', conformal2, config['physics'])
            self.results.append(result)
            
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        self.save_checkpoint("Complete")
        return pd.DataFrame(self.results)


def main():
    print("="*60)
    print("SEQUENTIAL BATCH RUNNER")
    print("="*60)
    
    runner = SequentialBatchRunner(device=DEVICE)
    results = runner.run_all()
    
    print("\n" + "="*60)
    print("Results:")
    print(results[['batch', 'dataset', 'coverage', 'set_size']].to_string())
    print("="*60)


if __name__ == "__main__":
    main()
