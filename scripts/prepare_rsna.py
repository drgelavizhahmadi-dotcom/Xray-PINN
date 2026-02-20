"""
Prepare RSNA Bone Age Dataset for Conformal Prediction
Converts bone age regression to classification bins
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path


def prepare_rsna_bone_age(data_dir='data/rsna_bone_age'):
    """
    Convert bone age regression to classification bins.
    Also creates binary gender classification for additional task.
    
    Args:
        data_dir: Root directory containing rsna-bone-age data
    """
    data_path = Path(data_dir)
    
    # Check for CSV file
    csv_candidates = [
        data_path / 'Bone Age Ground Truth.csv',
        data_path / 'boneage-training-dataset.csv',
        data_path / 'training.csv'
    ]
    
    csv_file = None
    for candidate in csv_candidates:
        if candidate.exists():
            csv_file = candidate
            break
    
    if csv_file is None:
        print(f"Ground truth CSV not found in {data_dir}")
        print("Looking for:")
        for c in csv_candidates:
            print(f"  - {c}")
        return None, None
    
    # Load ground truth
    print(f"Loading: {csv_file}")
    df = pd.read_csv(csv_file)
    
    # Standardize column names (different versions have different names)
    column_mapping = {
        'Image ID': 'image_id',
        'id': 'image_id',
        'Bone Age (months)': 'bone_age',
        'boneage': 'bone_age',
        'Sex': 'sex',
        'male': 'sex'
    }
    
    # Rename columns if they exist
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns:
            df = df.rename(columns={old_name: new_name})
    
    # Add image paths
    img_dir = data_path / 'boneage-training-dataset'
    df['image_path'] = df['image_id'].apply(
        lambda x: str(img_dir / f'{int(x)}.png')
    )
    
    # Check which files exist
    df['exists'] = df['image_path'].apply(os.path.exists)
    df = df[df['exists']].copy()
    
    if len(df) == 0:
        print(f"No images found in {img_dir}")
        print("Make sure PNG files are extracted.")
        return None, None
    
    print(f"Found {len(df)} valid images")
    
    # Create age bins (classification targets)
    # Converts regression to 4-class problem
    bins = [0, 18, 60, 144, 216]  # months
    labels = ['Infant', 'Toddler', 'Child', 'Adolescent']
    age_classes = [0, 1, 2, 3]
    
    df['age_class'] = pd.cut(df['bone_age'], bins=bins, labels=labels)
    df['age_label'] = pd.cut(df['bone_age'], bins=bins, labels=age_classes).astype(int)
    
    # Gender: Male=0, Female=1 (binary classification)
    if 'sex' in df.columns:
        # Handle different formats
        if df['sex'].dtype == object:
            df['gender_binary'] = (df['sex'].str.upper() == 'F').astype(int)
        else:
            df['gender_binary'] = df['sex'].astype(int)
    else:
        # Default to 0 if column not found
        df['gender_binary'] = 0
    
    # Sample calibration and test sets
    n_total = len(df)
    n_cal = min(800, int(n_total * 0.67))
    n_test = min(400, n_total - n_cal)
    
    cal_df = df.sample(n=n_cal, random_state=42)
    remaining = df[~df.index.isin(cal_df.index)]
    test_df = remaining.sample(n=min(n_test, len(remaining)), random_state=42)
    
    # Save CSVs
    output_dir = data_path / 'processed'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cal_path = output_dir / 'rsna_calibration.csv'
    test_path = output_dir / 'rsna_test.csv'
    
    cal_df.to_csv(cal_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\nRSNA Bone Age split:")
    print(f"  Calibration: {len(cal_df)} images")
    print(f"  Test: {len(test_df)} images")
    print(f"\nAge distribution (calibration):")
    print(cal_df['age_class'].value_counts())
    print(f"\nGender distribution (calibration):")
    print(f"  Male: {(cal_df['gender_binary'] == 0).sum()}")
    print(f"  Female: {(cal_df['gender_binary'] == 1).sum()}")
    
    print(f"\nSaved to:")
    print(f"  {cal_path}")
    print(f"  {test_path}")
    
    return cal_df, test_df


def main():
    """Process RSNA Bone Age dataset."""
    print("="*60)
    print("RSNA Bone Age Dataset Preparation")
    print("="*60)
    
    data_dir = 'data/rsna_bone_age'
    
    # Check if dataset exists
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"\nRSNA dataset not found at {data_path}")
        print("\nTo download RSNA Bone Age:")
        print("1. Visit: https://www.kaggle.com/c/rsna-bone-age")
        print("2. Download dataset (~8GB)")
        print("3. Extract to:")
        print(f"   {data_dir}/boneage-training-dataset/")
        print(f"   {data_dir}/Bone Age Ground Truth.csv")
        return
    
    prepare_rsna_bone_age(data_dir)
    
    print("\n" + "="*60)
    print("RSNA preparation complete!")
    print("="*60)


if __name__ == "__main__":
    main()
