"""
Prepare MURA Dataset for Conformal Prediction
Parses MURA structure and creates calibration/test splits
"""

import pandas as pd
import os
from pathlib import Path


def parse_mura_csv(data_dir='data/mura', split='valid'):
    """
    MURA structure: patient/study_label/image.png
    We need: image_path, study_id, label (0/1), body_part
    
    Args:
        data_dir: Root directory containing MURA-v1.1
        split: 'train' or 'valid'
    """
    data = []
    # Handle nested MURA-v1.1/MURA-v1.1 structure
    mura_path = os.path.join(data_dir, 'MURA-v1.1')
    if os.path.exists(os.path.join(mura_path, 'MURA-v1.1')):
        mura_path = os.path.join(mura_path, 'MURA-v1.1')
    base_path = os.path.join(mura_path, split)
    
    if not os.path.exists(base_path):
        print(f"Warning: {base_path} not found. Skipping.")
        return None, None
    
    body_parts = ['XR_ELBOW', 'XR_FINGER', 'XR_FOREARM', 'XR_HAND', 
                  'XR_HUMERUS', 'XR_SHOULDER', 'XR_WRIST']
    
    for body_part in body_parts:
        part_path = os.path.join(base_path, body_part)
        if not os.path.exists(part_path):
            continue
            
        for patient in os.listdir(part_path):
            patient_path = os.path.join(part_path, patient)
            if not os.path.isdir(patient_path):
                continue
                
            for study in os.listdir(patient_path):
                study_path = os.path.join(patient_path, study)
                if not os.path.isdir(study_path):
                    continue
                
                # Label is in folder name: study1_positive or study2_negative
                label = 1 if 'positive' in study.lower() else 0
                
                # Get all images in this study
                for img_file in os.listdir(study_path):
                    if img_file.endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(study_path, img_file)
                        data.append({
                            'image_path': img_path,
                            'study_id': f"{patient}_{study}",
                            'label': label,
                            'body_part': body_part.replace('XR_', ''),
                            'patient': patient
                        })
    
    if len(data) == 0:
        print(f"No data found in {base_path}")
        return None, None
    
    df = pd.DataFrame(data)
    print(f"\nTotal {split} images: {len(df)}")
    print(f"Body parts: {df['body_part'].value_counts().to_dict()}")
    
    # For conformal prediction, we sample calibration and test sets
    # Stratify by body part to ensure representation
    cal_samples = []
    test_samples = []
    
    for part in df['body_part'].unique():
        part_df = df[df['body_part'] == part]
        
        # Sample calibration (up to 150 per body part)
        n_cal = min(len(part_df), 150)
        part_cal = part_df.sample(n=n_cal, random_state=42)
        cal_samples.append(part_cal)
        
        # Sample test from remaining (up to 75 per body part)
        remaining = part_df[~part_df.index.isin(part_cal.index)]
        n_test = min(len(remaining), 75)
        if n_test > 0:
            part_test = remaining.sample(n=n_test, random_state=42)
            test_samples.append(part_test)
    
    cal_df = pd.concat(cal_samples, ignore_index=True)
    test_df = pd.concat(test_samples, ignore_index=True) if test_samples else pd.DataFrame()
    
    # Save CSVs
    output_dir = Path(data_dir) / 'processed'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cal_path = output_dir / f'mura_{split}_calibration.csv'
    test_path = output_dir / f'mura_{split}_test.csv'
    
    cal_df.to_csv(cal_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\nMURA {split} split:")
    print(f"  Calibration: {len(cal_df)} images")
    print(f"  Test: {len(test_df)} images")
    print(f"  Positive rate: {cal_df['label'].mean():.2%} (cal), {test_df['label'].mean():.2%} (test)")
    print(f"\nSaved to:")
    print(f"  {cal_path}")
    print(f"  {test_path}")
    
    return cal_df, test_df


def main():
    """Process both train and valid splits."""
    print("="*60)
    print("MURA Dataset Preparation")
    print("="*60)
    
    data_dir = 'data/mura'
    
    # Check if MURA exists
    mura_path = Path(data_dir) / 'MURA-v1.1'
    if not mura_path.exists():
        print(f"\nMURA dataset not found at {mura_path}")
        print("\nTo download MURA:")
        print("1. Visit: https://stanfordmlgroup.github.io/competitions/mura/")
        print("2. Request access (requires registration)")
        print("3. Download MURA-v1.1.zip")
        print(f"4. Extract to: {data_dir}/MURA-v1.1/")
        return
    
    # Process valid split (recommended for conformal calibration)
    print("\nProcessing validation split...")
    parse_mura_csv(data_dir, 'valid')
    
    # Optionally process train
    print("\nProcessing train split...")
    parse_mura_csv(data_dir, 'train')
    
    print("\n" + "="*60)
    print("MURA preparation complete!")
    print("="*60)


if __name__ == "__main__":
    main()
