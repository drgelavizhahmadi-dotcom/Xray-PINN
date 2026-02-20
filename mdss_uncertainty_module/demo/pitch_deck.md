# MDSS Uncertainty Module - Pitch Deck

## The Problem

EU AI Act (effective 2025) requires:
- **Article 15**: Accuracy metrics with uncertainty quantification
- **Article 14**: Human oversight with clear risk thresholds

Notified Bodies (TÜV, BSI) want statistical proof, not hand-waving.

**Current state**: Companies guess at "high confidence" or use softmax as probability (wrong).

## The Solution

Physics-based Bayesian uncertainty for medical X-ray AI.

**What it does:**
1. Takes X-ray image → runs 30 stochastic forward passes (MC Dropout)
2. Quantifies epistemic uncertainty (what the model doesn't know)
3. Maps to 3-tier risk: LOW/MEDIUM/HIGH
4. Generates PDF dossier for notified body submission

**Live Demo:**
```bash
make setup
./scripts/run_overnight.sh  # Generates 500-sample validation
make run-api
# Upload X-ray → get uncertainty + PDF dossier
```

## Technical Credibility

- **Architecture**: DenseNet121 (torchxrayvision) with Bayesian Dropout
- **Validation**: Reliability diagrams, ECE (Expected Calibration Error)
- **Compliance**: Direct mapping to AI Act Articles 14 & 15
- **Physics**: Epistemic uncertainty = variance across MC samples (Gal & Ghahramani, 2016)

## Business Model

- **Price**: €2,000 per technical dossier
- **Split**: 50/50 with MDSS
- **Target**: 10 dossiers/month = €20k revenue
- **Customers**: Medical AI startups seeking CE mark (Class IIa)

## Tech Stack

- PyTorch + torchxrayvision (pre-trained chest X-ray models)
- FastAPI (real-time inference)
- ReportLab (PDF generation)
- Monte Carlo Dropout (Bayesian neural networks)

## Status

- ✅ Core engine: MC Dropout + oversight policy
- ✅ Batch processor: 500-sample validation with ECE
- ✅ PDF generator: AI Act-compliant dossiers
- ✅ API: Upload X-ray → get uncertainty → download PDF
- ⏳ Saturday night: Run overnight batch on validation set
- ⏳ Sunday: Final integration testing

## Next Steps

1. **This weekend**: Complete overnight run, validate ECE < 0.1
2. **Monday**: Demo to MDSS leadership
3. **Week 2**: First pilot customer (identify via MDSS network)
4. **Month 1**: 3 paid dossiers, refine process

## Why Now?

EU AI Act enforcement starts August 2025. Companies scrambling for compliance documentation. First-mover advantage in "uncertainty-as-a-service" for medical AI.

---

**Contact**: team@mdss.com  
**Repo**: `github.com/mdss/mdss-uncertainty` (private)
