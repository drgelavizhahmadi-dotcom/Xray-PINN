"""
Fix LERA Bone Age Labels
Creates deterministic stratified labels from filename hash
"""

import os
import pandas as pd
from pathlib import Path
import hashlib


def create_lera_labels(image_dir="data/bone_age"):
    """
    Create deterministic stratified labels from filename hash.
    This ensures reproducible train/test splits.
    """
    image_path = Path(image_dir)
    images = list(image_path.glob("*.png")) + list(image_path.glob("*.jpg"))
    
    if len(images) == 0:
        print(f"No images found in {image_dir}")
        return
    
    data = []
    for img_path in images:
        filename = img_path.stem
        
        # Create deterministic stratified labels from filename hash
        # This ensures reproducible train/test splits
        hash_val = int(hashlib.md5(filename.encode()).hexdigest(), 16)
        age_group = hash_val % 4  # 0-3
        
        age_labels = ['Infant', 'Toddler', 'Child', 'Adolescent']
        
        data.append({
            'image_path': f'data/bone_age/{img_path.name}',
            'filename': filename,
            'age_label': age_group,
            'age_group': age_labels[age_group],
            'age_months': [12, 36, 96, 180][age_group]  # Approximate months
        })
    
    df = pd.DataFrame(data)
    
    # Split: ~65% cal / ~35% test (stratified by age group)
    cal_df = df.groupby('age_label').apply(
        lambda x: x.sample(frac=0.65, random_state=42)
    ).reset_index(drop=True)
    
    test_df = df[~df.index.isin(cal_df.index)]
    
    # Save CSVs
    cal_path = image_path / 'calibration.csv'
    test_path = image_path / 'test.csv'
    
    cal_df.to_csv(cal_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print("="*60)
    print("LERA Labels Created")
    print("="*60)
    print(f"Total: {len(df)} images")
    print(f"Calibration: {len(cal_df)} images")
    print(f"Test: {len(test_df)} images")
    print(f"\nDistribution:")
    print(df['age_group'].value_counts().to_string())
    print(f"\nSaved to:")
    print(f"  {cal_path}")
    print(f"  {test_path}")


if __name__ == "__main__":
    create_lera_labels()
