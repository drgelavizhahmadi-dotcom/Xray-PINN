"""FastAPI interface for uncertainty quantification."""

import io
import json
import sys
from pathlib import Path

# Force UTF-8 for Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import numpy as np
import torch
import torchvision.transforms as T
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image

from mdss_uncertainty.engine import UncertaintyEngine
from mdss_uncertainty.report_generator import ReportGenerator

app = FastAPI(title="MDSS Uncertainty API", version="0.1.0")

# CORS for frontend demo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine instance
_engine: UncertaintyEngine | None = None


def get_engine() -> UncertaintyEngine:
    """Get or initialize uncertainty engine."""
    global _engine
    if _engine is None:
        _engine = UncertaintyEngine()
    return _engine


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocess PIL image to tensor [1, 1, 224, 224]."""
    if image.mode != "L":
        image = image.convert("L")
    image = image.resize((224, 224))
    
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])
    return transform(image)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    engine = get_engine()
    return {
        "status": "ok",
        "model_loaded": engine.model is not None,
    }


@app.post("/analyze")
async def analyze_xray(file: UploadFile = File(...)):
    try:
        # Read file bytes properly
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file")
            
        # Load image from bytes
        img = Image.open(io.BytesIO(contents))
        
        # Convert to RGB if necessary (handle PNG with transparency)
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        # Resize to 224x224 (model input size)
        img = img.resize((224, 224))
        
        # Convert to grayscale (single channel) then to tensor [1, 1, 224, 224]
        img_gray = img.convert('L')  # L = grayscale
        img_array = np.array(img_gray, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)
        
        # Run engine
        engine = get_engine()
        unc_result = engine.monte_carlo_uncertainty(img_tensor)
        policy = engine.oversight_policy(
            unc_result["epistemic_uncertainty"], 
            unc_result["mean_confidence"]
        )
        
        return {
            "filename": file.filename,
            "uncertainty": unc_result["epistemic_uncertainty"],
            "confidence": unc_result["mean_confidence"],
            "risk_level": policy["risk_level"],
            "oversight_action": policy["oversight_action"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e).encode('ascii', 'ignore').decode('ascii')
        raise HTTPException(status_code=400, detail=f"Processing failed: {error_msg}")


@app.post("/generate-dossier")
async def generate_dossier(file: UploadFile = File(...)):
    """Generate PDF dossier for uploaded image."""
    # Check if aggregate results exist
    agg_path = Path("./data/results/aggregate_results.json")
    if not agg_path.exists():
        raise HTTPException(
            status_code=503,
            detail="Run batch processor first: python -m mdss_uncertainty.batch_processor"
        )
    
    try:
        # Load aggregate stats
        with open(agg_path) as f:
            agg_data = json.load(f)
        
        # Calculate aggregate stats
        valid = [r for r in agg_data if "uncertainty" in r and "error" not in r]
        n_samples = len(valid)
        
        # Simple ECE approximation
        ece = sum([abs(r.get("confidence", 0.5) - (r.get("ground_truth", [0.5])[0] if isinstance(r.get("ground_truth"), list) else 0.5)) for r in valid]) / len(valid) if valid else 0.05
        
        aggregate_stats = {"n_samples": n_samples, "ece": ece}
        
        # Process single image
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file")
            
        # Load image from bytes
        img = Image.open(io.BytesIO(contents))
        
        # Convert to RGB if necessary (handle PNG with transparency)
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        # Resize to 224x224 (model input size)
        img = img.resize((224, 224))
        
        # Convert to grayscale (single channel) then to tensor [1, 1, 224, 224]
        img_gray = img.convert('L')  # L = grayscale
        img_array = np.array(img_gray, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)
        
        engine = get_engine()
        result = engine.monte_carlo_uncertainty(img_tensor)
        policy = engine.oversight_policy(result["epistemic_uncertainty"], result["mean_confidence"])
        
        single_case = {
            "uncertainty": result["epistemic_uncertainty"],
            "confidence": result["mean_confidence"],
            "ci_lower": result["ci_lower"],
            "ci_upper": result["ci_upper"],
            "risk_level": policy["risk_level"],
            "oversight_action": policy["oversight_action"],
            "article_14_status": policy["article_14_interpretation"],
        }
        
        # Generate PDF
        output_path = f"./data/results/dossier_{file.filename}.pdf"
        generator = ReportGenerator(single_case=single_case, aggregate_stats=aggregate_stats)
        generator.generate(output_path)
        
        return FileResponse(
            output_path,
            media_type="application/pdf",
            filename=f"uncertainty_dossier_{file.filename}.pdf"
        )
    except HTTPException:
        raise
    except Exception as e:
        # Strip Unicode from error messages to avoid Windows encoding issues
        error_msg = str(e).encode('ascii', 'ignore').decode('ascii')
        raise HTTPException(status_code=500, detail=f"Dossier generation failed: {error_msg}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
