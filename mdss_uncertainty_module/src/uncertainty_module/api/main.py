"""FastAPI application for MDSS Uncertainty Module."""

from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Optional

import torch
import torchvision.transforms as T
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image

from uncertainty_module import SupportingDocumentation, UncertaintyOversightEngine
from uncertainty_module.core.batch_processor import BatchProcessor

# Global instances
engine: Optional[UncertaintyOversightEngine] = None
processor: Optional[BatchProcessor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize components on startup."""
    global engine, processor
    
    engine = UncertaintyOversightEngine()
    processor = BatchProcessor(engine)
    
    # Load a simple demo model (replace with actual model in production)
    from torch import nn
    
    class DemoModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(1024*1024, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, 14),
            )
        def forward(self, x):
            return self.fc(x)
    
    demo_model = DemoModel()
    class_names = ["No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
                   "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
                   "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
                   "Pleural Other", "Fracture", "Support Devices"]
    engine.register_model(demo_model, class_names)
    
    print("✓ Engine initialized")
    yield
    
    # Cleanup
    print("Shutting down...")


app = FastAPI(
    title="MDSS Uncertainty Module",
    description="Physics-based uncertainty quantification for medical AI compliance",
    version="0.1.0",
    lifespan=lifespan,
)


def _preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocess PIL image for model input."""
    if image.mode != 'L':
        image = image.convert('L')
    image = image.resize((1024, 1024))
    
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])
    return transform(image)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy" if engine else "uninitialized",
        "model_loaded": engine._model is not None if engine else False,
    }


@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """Analyze a single X-ray image."""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
        
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        tensor = _preprocess_image(image)
        
        result = engine.analyze(tensor, image_id=file.filename or "upload")
        
        return {
            "image_id": result.image_id,
            "prediction": result.prediction,
            "confidence": result.prediction_probability,
            "risk_level": result.uncertainty.risk_level.value,
            "requires_human_review": result.requires_human_review,
            "uncertainty": {
                "total": result.uncertainty.total_uncertainty,
                "aleatoric": result.uncertainty.aleatoric_uncertainty,
                "epistemic": result.uncertainty.epistemic_uncertainty,
            },
            "reasons": result.override_reasons,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/report")
async def generate_report(file: UploadFile = File(...)):
    """Generate PDF report for an image."""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
        
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        tensor = _preprocess_image(image)
        
        result = engine.analyze(tensor, file.filename or "report")
        
        # Generate PDF
        report_gen = SupportingDocumentation()
        output_path = Path(f"./data/results/report_{result.image_id}.pdf")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report_gen.generate(result, output_path)
        
        return StreamingResponse(
            open(output_path, "rb"),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={output_path.name}"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/config")
async def get_config():
    """Get current configuration."""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
        
    return {
        "confidence_threshold": engine.confidence_threshold,
        "uncertainty_threshold_low": engine.uncertainty_threshold_low,
        "uncertainty_threshold_high": engine.uncertainty_threshold_high,
        "mc_dropout_samples": engine.mc_dropout_samples,
    }
