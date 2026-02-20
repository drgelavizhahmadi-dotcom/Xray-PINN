"""Batch evaluator for overnight uncertainty analysis."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchxrayvision as xrv
from tqdm import tqdm

from mdss_uncertainty.engine import UncertaintyEngine


class BatchEvaluator:
    """Evaluate uncertainty on validation set."""
    
    def __init__(self, data_dir: str = "./data", output_dir: str = "./data/results"):
        self.engine = UncertaintyEngine()
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_validation_set(self, max_samples: int = 500) -> list:
        """Load CheXpert validation or generate dummy data."""
        try:
            dataset = xrv.datasets.CheX_Dataset(
                imgpath=str(self.data_dir / "raw" / "chexpert"),
                csvpath=str(self.data_dir / "raw" / "chexpert" / "valid.csv"),
                transform=None,
                data_aug=None,
            )
            samples = [(dataset[i][0], dataset[i][1], {"idx": i}) for i in range(min(max_samples, len(dataset)))]
            return samples
        except Exception:
            print("CheXpert not found, generating dummy data...")
            return [(torch.randn(1, 224, 224) * 0.5 + 0.5, np.random.randint(0, 2, 14).astype(float), {"idx": i}) for i in range(max_samples)]
            
    def process_single(self, image_tensor, ground_truth=None) -> dict:
        """Process single image."""
        result = self.engine.monte_carlo_uncertainty(image_tensor)
        policy = self.engine.oversight_policy(result["epistemic_uncertainty"], result["mean_confidence"])
        
        output = {
            "uncertainty": result["epistemic_uncertainty"],
            "confidence": result["mean_confidence"],
            "ci_lower": result["ci_lower"],
            "ci_upper": result["ci_upper"],
            "risk_level": policy["risk_level"],
            "oversight_action": policy["oversight_action"],
            "article_14_status": policy["article_14_interpretation"],
        }
        if ground_truth is not None:
            output["ground_truth"] = ground_truth.tolist() if hasattr(ground_truth, "tolist") else ground_truth
        if "flags" in result:
            output["flags"] = result["flags"]
        return output
        
    def run_batch(self, max_samples: int = 500) -> list:
        """Process batch with progress bar."""
        samples = self.load_validation_set(max_samples)
        results = []
        
        for i, (img, label, meta) in enumerate(tqdm(samples, desc="Processing")):
            try:
                result = self.process_single(img, label)
                result["idx"] = meta.get("idx", i)
                results.append(result)
                if (i + 1) % 50 == 0:
                    self._save_checkpoint(results, i + 1)
            except Exception as e:
                results.append({"idx": i, "error": str(e)})
                
        self._save_final(results)
        self.generate_calibration_plot(results, str(self.output_dir / "calibration_plot.png"))
        return results
        
    def _save_checkpoint(self, results: list, n: int):
        with open(self.output_dir / f"checkpoint_{n}.json", "w") as f:
            json.dump(results, f)
            
    def _save_final(self, results: list):
        with open(self.output_dir / "aggregate_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved {len(results)} results")
        
    def generate_calibration_plot(self, results: list, output_path: str):
        """Generate reliability diagram with ECE."""
        valid = [r for r in results if "confidence" in r and "ground_truth" in r and "error" not in r]
        
        if len(valid) < 10:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot([0, 1], [0, 1], "k--")
            ax.set_title("Insufficient Data")
            plt.savefig(output_path, dpi=150)
            plt.close()
            return
            
        confs = [r["confidence"] for r in valid]
        accs = [r["ground_truth"][0] if isinstance(r["ground_truth"], list) else 0.5 for r in valid]
        
        # 10 bins
        bins = np.linspace(0, 1, 11)
        bin_accs, bin_confs, bin_counts = [], [], []
        
        for i in range(10):
            mask = [(c > bins[i]) and (c <= bins[i+1]) for c in confs]
            if sum(mask) > 0:
                bin_accs.append(sum([a for a, m in zip(accs, mask) if m]) / sum(mask))
                bin_confs.append(sum([c for c, m in zip(confs, mask) if m]) / sum(mask))
                bin_counts.append(sum(mask))
            else:
                bin_accs.append(0)
                bin_confs.append(0)
                bin_counts.append(0)
                
        ece = sum([abs(a - c) * (n / len(valid)) for a, c, n in zip(bin_accs, bin_confs, bin_counts)])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot([0, 1], [0, 1], "k--", label="Perfect")
        ax.bar([(bins[i] + bins[i+1])/2 for i in range(10)], bin_accs, width=0.08, alpha=0.7, edgecolor="black")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Reliability Diagram (ECE: {ece:.4f})")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Calibration plot: {output_path} (ECE: {ece:.4f})")


if __name__ == "__main__":
    evaluator = BatchEvaluator()
    results = evaluator.run_batch(500)
    print(f"\nDone: {sum(1 for r in results if 'error' not in r)}/{len(results)} successful")
