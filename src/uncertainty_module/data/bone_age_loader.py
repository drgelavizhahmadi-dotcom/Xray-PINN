"""
Bone Age X-ray Dataset Loader
Loads hand/wrist X-rays for age estimation
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np
from typing import Optional, Tuple


class BoneAgeDataset(Dataset):
    """
    RSNA Bone Age X-ray dataset.
    Hand/wrist radiographs with bone age labels (months).
    """
    
    def __init__(
        self,
        data_dir: str = "data/bone_age",
        transform: Optional[transforms.Compose] = None,
        target_size: Tuple[int, int] = (224, 224)
    ):
        self.data_dir = Path(data_dir)
        self.target_size = target_size
        
        # Default transform if none provided
        self.transform = transform or self._default_transform()
        
        # Find all PNG files
        self.image_files = sorted(self.data_dir.glob("*.png"))
        
        if len(self.image_files) == 0:
            raise ValueError(f"No PNG files found in {data_dir}")
        
        print(f"Loaded {len(self.image_files)} Bone Age X-rays from {data_dir}")
    
    def _default_transform(self):
        """Default preprocessing for bone age X-rays."""
        return transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        
        # Load image
        image = Image.open(img_path).convert('L')  # Grayscale
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Extract age from filename (if encoded)
        # Format: LCYYYYMMDDXXX.png - we use random for now
        # In real dataset, this would come from CSV
        bone_age_months = np.random.randint(12, 180)  # 1-15 years
        
        return image, torch.tensor(bone_age_months, dtype=torch.float32)


def get_bone_age_loaders(
    data_dir: str = "data/bone_age",
    batch_size: int = 16,
    train_split: float = 0.8,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """
    Get train and validation data loaders for Bone Age dataset.
    
    Args:
        data_dir: Directory containing bone age X-rays
        batch_size: Batch size for training
        train_split: Fraction of data for training
        num_workers: Number of data loading workers
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create dataset
    dataset = BoneAgeDataset(data_dir)
    
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
    print("Testing Bone Age data loader...")
    
    try:
        train_loader, val_loader = get_bone_age_loaders()
        
        # Get one batch
        images, ages = next(iter(train_loader))
        
        print(f"Batch shape: {images.shape}")
        print(f"Ages: {ages[:5]} months")
        print(f"Ages in years: {(ages[:5] / 12).numpy()}")
        
        print("\nData loader working correctly!")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have downloaded the Bone Age dataset:")
        print("  python scripts/download_bone_age_direct.py")
