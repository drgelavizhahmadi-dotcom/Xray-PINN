# Bone Age X-ray Dataset Download Guide

## Dataset Information
- **Source:** RSNA Pediatric Bone Age Machine Learning Challenge
- **Images:** ~14,000 hand/wrist X-rays
- **Labels:** Bone age (months)
- **Size:** ~8GB

## Download Methods

### Method 1: AzCopy (Recommended for large datasets)

**Step 1: Install AzCopy**
```powershell
# Windows
winget install Microsoft.AzCopy

# Or download manually:
# https://docs.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10
```

**Step 2: Get SAS URL**
- Request download access from dataset provider
- You'll receive an Azure SAS URL (valid 24-48 hours)
- URL format: `https://<account>.blob.core.windows.net/<container>?<sas-token>`

**Step 3: Download**
```bash
# Using our script
python scripts/download_bone_age_azcopy.py

# Or manually with AzCopy
azcopy copy "<YOUR_SAS_URL>" "data/bone_age" --recursive
```

### Method 2: Azure Storage Explorer (GUI)

**Step 1: Download Azure Storage Explorer**
- https://azure.microsoft.com/en-us/products/storage/storage-explorer

**Step 2: Connect to Blob Container**
1. Open Storage Explorer
2. Click "Connect to Azure resources"
3. Select "Blob container"
4. Choose "Shared access signature (SAS)"
5. Paste your SAS URL
6. Connect and browse

**Step 3: Download**
1. Right-click the container
2. Select "Download"
3. Choose destination: `data/bone_age/`

### Method 3: Python (Slower but no extra tools)

```bash
pip install azure-storage-blob
python scripts/download_bone_age_azcopy.py
# Select Python SDK option when prompted
```

## Expected Directory Structure

```
data/bone_age/
├── train/
│   ├── 0001.png
│   ├── 0002.png
│   └── ...
├── validation/
│   ├── 0001.png
│   └── ...
├── train.csv          # bone age labels
└── validation.csv     # bone age labels
```

## Using with Conformal Prediction

Once downloaded, you can test conformal prediction on bone age estimation:

```python
from uncertainty_module.core.conformal import ConformalPredictor
from uncertainty_module.data.bone_age_loader import get_bone_age_loader

# Load bone age data
train_loader, val_loader = get_bone_age_loader("data/bone_age")

# Apply conformal prediction
# ( bone age is regression, so we'd use conformalized quantile regression )
```

## Alternative: RSNA Kaggle

If Azure download is problematic, try Kaggle:
- https://www.kaggle.com/c/rsna-bone-age

```bash
pip install kaggle
kaggle competitions download -c rsna-bone-age
```

## Troubleshooting

### "SAS token expired"
- Request a new download link from the dataset provider
- SAS tokens are time-limited for security

### "AzCopy not found"
- Add AzCopy to your PATH, or
- Use Azure Storage Explorer GUI instead

### Slow download
- AzCopy is fastest for bulk transfers
- Python SDK is slower but works without installation
- Check your internet connection (8GB download)

## References

- **Original Paper:** Halabi et al., "The RSNA Pediatric Bone Age Machine Learning Challenge"
- **RSNA:** https://www.rsna.org/
- **Challenge:** https://www.rsna.org/education/ai-resources-and-training/ai-image-challenge/RSNA-Pediatric-Bone-Age-Machine-Learning-Challenge-2017
