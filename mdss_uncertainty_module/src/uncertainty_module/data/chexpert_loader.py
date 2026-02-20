"""CheXpert dataset loader."""

from pathlib import Path
from typing import Callable, Optional, Union

import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# CheXpert labels
CHEXPERT_LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
    "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
    "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
    "Pleural Other", "Fracture", "Support Devices",
]


class CheXpertDataset(Dataset):
    """CheXpert X-ray dataset."""
    
    def __init__(
        self,
        root_dir: Union[str, Path],
        split: str = "train",
        transform: Optional[Callable] = None,
        use_small: bool = True,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform or self._default_transform()
        
        csv_name = f"{split}_small.csv" if use_small else f"{split}.csv"
        self.csv_path = self.root_dir / csv_name
        
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CheXpert CSV not found: {self.csv_path}")
            
        self.df = pd.read_csv(self.csv_path)
        self.df = self.df.fillna(0).replace(-1, 0)  # Handle uncertainty labels
        
    def __len__(self) -> int:
        return len(self.df)
        
    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        
        img_path = self.root_dir / row['Path']
        try:
            image = Image.open(img_path).convert('L')
        except Exception:
            image = Image.new('L', (1024, 1024))
            
        image = self.transform(image)
        
        labels = torch.tensor([row.get(col, 0) for col in CHEXPERT_LABELS], dtype=torch.float32)
        
        return image, labels, str(img_path)
        
    def _default_transform(self):
        return T.Compose([
            T.Resize((1024, 1024)),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
        ])


def get_chexpert_loader(
    root_dir: Union[str, Path],
    split: str = "train",
    batch_size: int = 16,
    num_workers: int = 4,
    use_small: bool = True,
) -> DataLoader:
    """Get a DataLoader for CheXpert dataset."""
    dataset = CheXpertDataset(root_dir, split, use_small=use_small)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
    )
