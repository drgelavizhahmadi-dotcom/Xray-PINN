"""Batch processor for overnight evaluation runs."""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from tqdm import tqdm

from uncertainty_module.core.engine import OversightResult, UncertaintyOversightEngine


@dataclass
class BatchResult:
    """Results from batch processing."""
    total_images: int
    processed_images: int
    failed_images: int
    human_review_required: int
    risk_distribution: dict[str, int] = field(default_factory=dict)
    processing_time_seconds: float = 0.0


class BatchProcessor:
    """Processes batches of X-ray images for uncertainty analysis."""
    
    def __init__(
        self,
        engine: UncertaintyOversightEngine,
        output_dir: str = "./data/results",
    ):
        self.engine = engine
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._results: list[OversightResult] = []
        
    def process(
        self,
        images: list[torch.Tensor],
        image_ids: list[str],
    ) -> BatchResult:
        """Process a dataset of images."""
        start_time = time.time()
        total = len(images)
        
        print(f"Processing {total} images...")
        
        self._results = []
        failed = []
        
        for img, img_id in tqdm(zip(images, image_ids), total=total):
            try:
                result = self.engine.analyze(img, img_id)
                self._results.append(result)
            except Exception as e:
                print(f"Failed to process {img_id}: {e}")
                failed.append({"image_id": img_id, "error": str(e)})
                
        # Calculate statistics
        batch_result = self._calculate_stats(total, len(failed), start_time)
        
        # Save results
        self._save_results(batch_result, failed)
        
        print(f"\nComplete! Results saved to {self.output_dir}")
        print(f"Human review required: {batch_result.human_review_required}/{batch_result.processed_images}")
        
        return batch_result
        
    def _calculate_stats(
        self,
        total: int,
        n_failed: int,
        start_time: float,
    ) -> BatchResult:
        """Calculate aggregate statistics."""
        risk_counts = {}
        for r in self._results:
            level = r.uncertainty.risk_level.value
            risk_counts[level] = risk_counts.get(level, 0) + 1
            
        human_review_count = sum(1 for r in self._results if r.requires_human_review)
        
        return BatchResult(
            total_images=total,
            processed_images=len(self._results),
            failed_images=n_failed,
            human_review_required=human_review_count,
            risk_distribution=risk_counts,
            processing_time_seconds=time.time() - start_time,
        )
        
    def _save_results(
        self,
        batch_result: BatchResult,
        failed: list[dict],
    ) -> None:
        """Save results to disk."""
        # Save aggregate stats
        stats_path = self.output_dir / "stats.json"
        with open(stats_path, 'w') as f:
            json.dump({
                "total": batch_result.total_images,
                "processed": batch_result.processed_images,
                "failed": batch_result.failed_images,
                "human_review": batch_result.human_review_required,
                "risk_dist": batch_result.risk_distribution,
                "time_seconds": batch_result.processing_time_seconds,
            }, f, indent=2)
            
        # Export to CSV
        records = []
        for r in self._results:
            records.append({
                "image_id": r.image_id,
                "prediction": r.prediction,
                "confidence": r.prediction_probability,
                "requires_review": r.requires_human_review,
                "risk_level": r.uncertainty.risk_level.value,
                "total_uncertainty": r.uncertainty.total_uncertainty,
            })
            
        df = pd.DataFrame(records)
        df.to_csv(self.output_dir / "results.csv", index=False)
        
        # Save failed log
        if failed:
            with open(self.output_dir / "failed.json", 'w') as f:
                json.dump(failed, f, indent=2)
                
    def get_dataframe(self) -> pd.DataFrame:
        """Get results as a pandas DataFrame."""
        if not self._results:
            return pd.DataFrame()
            
        records = []
        for r in self._results:
            records.append({
                "image_id": r.image_id,
                "prediction": r.prediction,
                "confidence": r.prediction_probability,
                "requires_review": r.requires_human_review,
                "risk_level": r.uncertainty.risk_level.value,
                "total_uncertainty": r.uncertainty.total_uncertainty,
                "epistemic": r.uncertainty.epistemic_uncertainty,
                "aleatoric": r.uncertainty.aleatoric_uncertainty,
            })
            
        return pd.DataFrame(records)
