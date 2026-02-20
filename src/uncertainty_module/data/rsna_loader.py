"""
RSNA Bone Age Dataset Loader
Hand/wrist X-rays for pediatric bone age estimation
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np
from typing import Optional, Tuple


class RSNABoneAgeDataset(Dataset):
    """
    RSNA Bone Age dataset.
    Pediatric hand/wrist X-rays with bone age labels (months).
    Supports both regression (age in months) and classification (age bins).
    """
    
    def __init__(
        self,
        csv_file: str,
        data_dir: str = "data/rsna_bone_age",
        transform: Optional[transforms.Compose] = None,
        target_size: Tuple[int, int] = (224, 224),
        task: str = 'classification'  # 'classification' or 'regression'
    ):
        self.data_dir = Path(data_dir)
        self.target_size = target_size
        self.task = task
        
        # Load CSV
        self.df = pd.read_csv(csv_file)
        
        if len(self.df) == 0:
            raise ValueError(f"No data found in {csv_file}")
        
        # Default transform if none provided
        self.transform = transform or self._default_transform()
        
        print(f"Loaded {len(self.df)} RSNA images from {csv_file}")
        print(f"  Task: {task}")
        if task == 'classification':
            print(f"  Age classes: {self.df['age_class'].value_counts().to_dict()}")
        else:
            print(f"  Age range: {self.df['bone_age'].min():.0f} - {self.df['bone_age'].max():.0f} months")
        print(f"  Gender: {(self.df['gender_binary'] == 0).sum()} male, {(self.df['gender_binary'] == 1).sum()} female")
    
    def _default_transform(self):
        """Default preprocessing for RSNA bone age X-rays."""
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
        
        # Get target based on task
        if self.task == 'classification':
            # 4-class age classification
            label = int(row['age_label'])
            target = torch.tensor(label, dtype=torch.long)
        else:
            # Regression (bone age in months)
            age = float(row['bone_age'])
            target = torch.tensor(age, dtype=torch.float32)
        
        # Also return gender as auxiliary info
        gender = int(row['gender_binary'])
        
        return image, target, gender


def get_rsna_loaders(
    data_dir: str = "data/rsna_bone_age",
    batch_size: int = 16,
    num_workers: int = 0,
    task: str = 'classification'
) -> Tuple[DataLoader, DataLoader]:
    """
    Get calibration and test data loaders for RSNA dataset.
    
    Args:
        data_dir: Directory containing RSNA data and processed CSVs
        batch_size: Batch size
        num_workers: Number of data loading workers
        task: 'classification' (age bins) or 'regression' (months)
        
    Returns:
        Tuple of (cal_loader, test_loader)
    """
    processed_dir = Path(data_dir) / 'processed'
    
    cal_csv = processed_dir / 'rsna_calibration.csv'
    test_csv = processed_dir / 'rsna_test.csv'
    
    if not cal_csv.exists():
        raise FileNotFoundError(
            f"{cal_csv} not found. Run: python scripts/prepare_rsna.py"
        )
    
    # Create datasets
    cal_dataset = RSNABoneAgeDataset(
        str(cal_csv), data_dir, task=task
    )
    test_dataset = RSNABoneAgeDataset(
        str(test_csv), data_dir, task=task
    )
    
    # Create loaders
    cal_loader = DataLoader(
        cal_dataset,
        batch_size=batch_size,
        shuffle=False,
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
    print("Testing RSNA data loader...")
    
    try:
        # Test classification
        print("\n--- Classification Task ---")
        cal_loader, test_loader = get_rsna_loaders(task='classification')
        
        images, labels, genders = next(iter(cal_loader))
        print(f"Batch shape: {images.shape}")
        print(f"Labels: {labels[:10]}")
        print(f"Genders: {genders[:10]} (0=M, 1=F)")
        
        # Test regression
        print("\n--- Regression Task ---")
        cal_loader, test_loader = get_rsna_loaders(task='regression')
        
        images, ages, genders = next(iter(cal_loader))
        print(f"Ages (months): {ages[:10].numpy()}")
        print(f"Ages (years): {(ages[:10] / 12).numpy()}")
        
        print("\nData loader working correctly!")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have:")
        print("1. Downloaded RSNA dataset to data/rsna_bone_age/")
        print("2. Run: python scripts/prepare_rsna.py")
