"""Generate demo PDFs for MDSS presentation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
from PIL import Image

from uncertainty_module import SupportingDocumentation, UncertaintyOversightEngine
from uncertainty_module.core.engine import RiskLevel, UncertaintyMetrics, OversightResult


def create_synthetic_image(pattern: str = "normal") -> Image.Image:
    """Create synthetic X-ray-like image."""
    np.random.seed(42)
    size = 1024
    
    if pattern == "normal":
        img = np.random.normal(0.5, 0.1, (size, size))
    elif pattern == "noisy":
        img = np.random.normal(0.5, 0.2, (size, size))
    else:
        img = np.ones((size, size)) * 0.5
        
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img, mode='L')


def create_result(level: str) -> OversightResult:
    """Create sample result for risk level."""
    configs = {
        "low": (0.08, 0.92, RiskLevel.LOW, False, []),
        "medium": (0.20, 0.80, RiskLevel.MEDIUM, False, []),
        "high": (0.35, 0.60, RiskLevel.HIGH, True, [
            "Confidence below threshold",
            "High uncertainty detected"
        ]),
    }
    
    unc_val, conf, risk, review, reasons = configs[level]
    
    return OversightResult(
        image_id=f"demo_{level}_001",
        prediction="No Finding" if level == "low" else "Pneumonia",
        prediction_probability=conf,
        uncertainty=UncertaintyMetrics(
            total_uncertainty=unc_val,
            aleatoric_uncertainty=unc_val * 0.4,
            epistemic_uncertainty=unc_val * 0.6,
            confidence_score=conf,
            risk_level=risk,
        ),
        requires_human_review=review,
        override_reasons=reasons,
        confidence_interval=(conf - 0.1, conf + 0.1),
    )


def main():
    """Generate demo PDFs."""
    print("Generating demo PDFs...")
    
    output_dir = Path("demo/expected_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_gen = SupportingDocumentation()
    
    for level, pattern in [("low", "normal"), ("medium", "noisy"), ("high", "extreme")]:
        print(f"  Creating {level} uncertainty demo...")
        
        result = create_result(level)
        output_path = output_dir / f"demo_{level}_uncertainty.pdf"
        report_gen.generate(result, output_path)
        
        print(f"    ✓ {output_path}")
    
    print("\nDemo PDFs generated successfully!")


if __name__ == "__main__":
    main()
