"""
Download CheXpert dataset (small version).
Validation set only (~11GB).
"""

import urllib.request
import zipfile
import os
from pathlib import Path
import sys

def download_with_progress(url: str, output_path: str):
    """Download file with progress bar."""
    import ssl
    
    # Create request with headers to avoid 403
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    # Handle SSL
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    req = urllib.request.Request(url, headers=headers)
    
    print(f"Downloading from: {url}")
    print(f"Saving to: {output_path}")
    print("This will download ~11GB. Press Ctrl+C to cancel.")
    print("Starting in 3 seconds...")
    import time
    time.sleep(3)
    
    with urllib.request.urlopen(req, context=ssl_context) as response:
        total_size = int(response.headers.get('Content-Length', 0))
        block_size = 8192
        downloaded = 0
        
        with open(output_path, 'wb') as f:
            while True:
                chunk = response.read(block_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                
                if total_size > 0:
                    percent = downloaded * 100 / total_size
                    mb_downloaded = downloaded / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)
                    sys.stdout.write(f"\r  Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)")
                    sys.stdout.flush()
    
    print("\nDownload complete!")

def main():
    # Config
    data_dir = Path("data/chexpert")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    zip_path = data_dir / "CheXpert-v1.0-small.zip"
    extract_dir = data_dir
    
    url = "https://storage.googleapis.com/chexpert/CheXpert-v1.0-small.zip"
    
    # Check if already downloaded
    if zip_path.exists():
        print(f"ZIP file already exists: {zip_path}")
        response = input("Re-download? (y/N): ").lower()
        if response != 'y':
            print("Skipping download.")
        else:
            zip_path.unlink()
            download_with_progress(url, str(zip_path))
    else:
        download_with_progress(url, str(zip_path))
    
    # Extract
    if zip_path.exists():
        print(f"\nExtracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print("Extraction complete!")
        
        # Optional: Remove zip to save space
        print(f"\nZIP file location: {zip_path}")
        response = input("Remove ZIP file to save space? (y/N): ").lower()
        if response == 'y':
            zip_path.unlink()
            print("ZIP file removed.")
    
    print(f"\nCheXpert dataset ready at: {extract_dir}")

if __name__ == "__main__":
    main()
