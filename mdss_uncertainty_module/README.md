# MDSS Uncertainty Module

Physics-based uncertainty quantification for EU AI Act medical device compliance.

## Features

- **Uncertainty Quantification**: Monte Carlo dropout with aleatoric/epistemic decomposition
- **Risk Classification**: 4-tier system (LOW, MEDIUM, HIGH, CRITICAL)
- **AI Act Compliance**: Automated supporting documentation
- **FastAPI**: Real-time inference API

## Quick Start

### One-time setup
```bash
chmod +x scripts/setup.sh
./scripts/setup.sh
```

### Development
```bash
source .venv/bin/activate
make run-api
```

### Generate demos
```bash
make generate-demo
```

## Usage

```python
from uncertainty_module import UncertaintyOversightEngine

engine = UncertaintyOversightEngine()
engine.register_model(your_model, class_names)

result = engine.analyze(image_tensor, "patient_001")

print(f"Prediction: {result.prediction}")
print(f"Risk Level: {result.uncertainty.risk_level.value}")
print(f"Human Review Required: {result.requires_human_review}")
```

## API

| Endpoint | Description |
|----------|-------------|
| `POST /analyze` | Analyze single image |
| `POST /report` | Generate PDF report |
| `GET /health` | Health check |
| `GET /config` | View configuration |

## Testing

```bash
make test
make lint
```

## Structure

```
mdss_uncertainty_module/
├── src/uncertainty_module/    # Source code
│   ├── core/                  # Uncertainty engine
│   ├── api/                   # FastAPI
│   ├── reporting/             # PDF generation
│   └── data/                  # Data loaders
├── tests/                     # Test suite
├── scripts/                   # Automation
└── docs/                      # Documentation
```

## License

Proprietary - Medical AI Systems Inc.
