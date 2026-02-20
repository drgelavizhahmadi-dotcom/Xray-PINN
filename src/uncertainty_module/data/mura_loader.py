"""
MURA (MUsculoskeletal RAdiographs) Dataset Loader
Bone X-rays with abnormality labels for conformal prediction
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np
from typing import Optional, Tuple


class MURADataset(Dataset):
    """
    MURA dataset: Musculoskeletal radiographs.
    7 body parts: elbow, finger, forearm, hand, humerus, shoulder, wrist
    Labels: 0 = normal, 1 = abnormal
    """
    
    def __init__(
        self,
        csv_file: str,
        data_dir: str = "data/mura",
        transform: Optional[transforms.Compose] = None,
        target_size: Tuple[int, int] = (224, 224)
    ):
        self.data_dir = Path(data_dir)
        self.target_size = target_size
        
        # Load CSV
        self.df = pd.read_csv(csv_file)
        
        if len(self.df) == 0:
            raise ValueError(f"No data found in {csv_file}")
        
        # Default transform if none provided
        self.transform = transform or self._default_transform()
        
        print(f"Loaded {len(self.df)} MURA images from {csv_file}")
        print(f"  Body parts: {self.df['body_part'].value_counts().to_dict()}")
        print(f"  Abnormal rate: {self.df['label'].mean():.2%}")
    
    def _default_transform(self):
        """Default preprocessing for MURA X-rays."""
        return transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        img_path = self.data_dir / row['image_path']
        image = Image.open(img_path).convert('L')  # Grayscale
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get label
        label = int(row['label'])
        
        return image, torch.tensor(label, dtype=torch.long)


def get_mura_loaders(
    data_dir: str = "data/mura",
    batch_size: int = 16,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """
    Get calibration and test data loaders for MURA dataset.
    
    Args:
        data_dir: Directory containing MURA-v1.1 and processed CSVs
        batch_size: Batch size
        num_workers: Number of data loading workers
        
    Returns:
        Tuple of (cal_loader, test_loader)
    """
    processed_dir = Path(data_dir) / 'processed'
    
    cal_csv = processed_dir / 'mura_valid_calibration.csv'
    test_csv = processed_dir / 'mura_valid_test.csv'
    
    if not cal_csv.exists():
        raise FileNotFoundError(
            f"{cal_csv} not found. Run: python scripts/prepare_mura.py"
        )
    
    # Create datasets
    cal_dataset = MURADataset(str(cal_csv), data_dir)
    test_dataset = MURADataset(str(test_csv), data_dir)
    
    # Create loaders
    cal_loader = DataLoader(
        cal_dataset,
        batch_size=batch_size,
        shuffle=False,  # No shuffle for calibration
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return cal_loader, test_loader


if __name__ == "__main__":
    # Test the loader
    print("Testing MURA data loader...")
    
    try:
        cal_loader, test_loader = get_mura_loaders()
        
        # Get one batch
        images, labels = next(iter(cal_loader))
        
        print(f"\nBatch shape: {images.shape}")
        print(f"Labels: {labels[:10]}")
        print(f"Abnormal rate in batch: {labels.float().mean():.2%}")
        
        print("\nData loader working correctly!")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have:")
        print("1. Downloaded MURA dataset to data/mura/MURA-v1.1/")
        print("2. Run: python scripts/prepare_mura.py")
