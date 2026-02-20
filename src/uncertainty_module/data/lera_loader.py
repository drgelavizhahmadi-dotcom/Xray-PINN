"""
LERA - Lower Extremity Radiographs Dataset Loader
Loads hip, knee, ankle X-rays for analysis
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np
from typing import Optional, Tuple


class LERADataset(Dataset):
    """
    LERA (Lower Extremity Radiographs) dataset.
    Hip, knee, and ankle X-rays.
    """
    
    def __init__(
        self,
        data_dir: str = "data/lera",
        transform: Optional[transforms.Compose] = None,
        target_size: Tuple[int, int] = (224, 224)
    ):
        self.data_dir = Path(data_dir)
        self.target_size = target_size
        
        # Default transform if none provided
        self.transform = transform or self._default_transform()
        
        # Find all image files (DICOM or PNG/JPG)
        self.image_files = []
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.dcm"]:
            self.image_files.extend(self.data_dir.glob(ext))
            self.image_files.extend(self.data_dir.rglob(ext))  # Recursive
        
        self.image_files = sorted(self.image_files)
        
        if len(self.image_files) == 0:
            raise ValueError(f"No image files found in {data_dir}")
        
        print(f"Loaded {len(self.image_files)} LERA X-rays from {data_dir}")
    
    def _default_transform(self):
        """Default preprocessing for lower extremity X-rays."""
        return transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        
        # Load image (handle DICOM if needed)
        if img_path.suffix.lower() == '.dcm':
            try:
                import pydicom
                dcm = pydicom.dcmread(img_path)
                image = Image.fromarray(dcm.pixel_array).convert('L')
            except ImportError:
                raise ImportError("pydicom required for DICOM files. Install: pip install pydicom")
        else:
            image = Image.open(img_path).convert('L')  # Grayscale
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Placeholder labels - in real use, these would come from annotations
        # Random binary labels for demonstration (e.g., fracture present/absent)
        label = np.random.randint(0, 2)
        
        return image, torch.tensor(label, dtype=torch.long)


def get_lera_loaders(
    data_dir: str = "data/lera",
    batch_size: int = 16,
    train_split: float = 0.8,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """
    Get train and validation data loaders for LERA dataset.
    
    Args:
        data_dir: Directory containing LERA X-rays
        batch_size: Batch size for training
        train_split: Fraction of data for training
        num_workers: Number of data loading workers
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create dataset
    dataset = LERADataset(data_dir)
    
    # Split train/val
    n_train = int(len(dataset) * train_split)
    n_val = len(dataset) - n_train
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val]
    )
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    print(f"Train samples: {n_train}, Val samples: {n_val}")
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test the loader
    print("Testing LERA data loader...")
    
    try:
        train_loader, val_loader = get_lera_loaders()
        
        # Get one batch
        images, labels = next(iter(train_loader))
        
        print(f"Batch shape: {images.shape}")
        print(f"Labels: {labels[:5]}")
        
        print("\nData loader working correctly!")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have downloaded the LERA dataset:")
        print("  python scripts/download_lera.py")
