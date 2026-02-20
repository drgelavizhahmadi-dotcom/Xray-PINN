# LERA - Lower Extremity Radiographs Dataset

## Dataset Information
- **Source:** Stanford AIMI
- **Type:** Lower extremity radiographs (hip, knee, ankle)
- **Images:** Lower extremity X-rays
- **Format:** DICOM or PNG

## Download Instructions

### Method 1: Using Our Script (Python SDK)
```bash
# Create directory
mkdir -p data/lera

# Run download with the SAS URL
python scripts/download_lera.py

# Or pass URL directly
python scripts/download_bone_age_direct.py "YOUR_SAS_URL"
```

### Method 2: AzCopy (Recommended for large datasets)
```bash
# Install AzCopy
winget install Microsoft.AzCopy

# Download
azcopy copy "YOUR_SAS_URL" "data/lera" --recursive
```

### Method 3: Azure Storage Explorer (GUI)
1. Download: https://azure.microsoft.com/en-us/products/storage/storage-explorer
2. Connect with SAS URL
3. Download to `data/lera/`

## Expected Structure
```
data/lera/
├── train/
├── validation/
└── test/
```

## Using with Conformal Prediction

```python
from uncertainty_module.data.lera_loader import get_lera_loaders
from uncertainty_module.core.conformal import ConformalPredictor

# Load data
train_loader, val_loader = get_lera_loaders("data/lera")

# Apply conformal prediction for fracture detection
```

## Notes
- SAS token is time-limited (expires March 22, 2026)
- If download fails, request a new SAS URL from dataset provider
