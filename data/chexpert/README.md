# CheXpert Dataset Download Instructions

## Official Source
**Website:** https://stanfordmlgroup.github.io/competitions/chexpert/

## Manual Download Steps

1. **Go to:** https://stanfordmlgroup.github.io/competitions/chexpert/

2. **Click:** "Download CheXpert-v1.0-small" (requires registration)

3. **After download, extract:**
   ```bash
   # Windows PowerShell
   Expand-Archive CheXpert-v1.0-small.zip -DestinationPath .
   
   # Or use 7-Zip, WinRAR, etc.
   ```

4. **Expected structure:**
   ```
   data/chexpert/
   └── CheXpert-v1.0-small/
       ├── train/
       ├── valid/
       ├── train.csv
       └── valid.csv
   ```

## Alternative: Kaggle
**URL:** https://www.kaggle.com/datasets/jaykumar1607/chexpert

No registration required, just Kaggle account.

## Dataset Statistics
- **Train:** 223,414 images
- **Validation:** 234 images  
- **Size:** ~11GB (small version)
- **Labels:** 14 pathology classes

## Using with Our Code
Once downloaded, the data loader will automatically find it at:
```
data/chexpert/CheXpert-v1.0-small/
```

## Quick Test Without Full Dataset
The demos work with synthetic data for development:
```bash
python demo/physics_efficiency_test.py
```

For production training, you'll need the real CheXpert data.
