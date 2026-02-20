"""
Download Bone Age X-ray Dataset from Azure
Uses AzCopy or Azure Storage Python SDK
"""

import subprocess
import sys
from pathlib import Path
import os


def check_azcopy():
    """Check if AzCopy is installed."""
    try:
        result = subprocess.run(['azcopy', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"AzCopy found: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    return False


def download_with_azcopy(sas_url: str, output_dir: str):
    """
    Download dataset using AzCopy.
    
    Args:
        sas_url: The Azure SAS URL (contains the time-limited key)
        output_dir: Where to save the dataset
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading Bone Age dataset to: {output_path}")
    print("This may take 30+ minutes depending on dataset size...")
    
    # AzCopy command: copy from URL to local directory
    cmd = [
        'azcopy',
        'copy',
        sas_url,
        str(output_path),
        '--recursive',
        '--check-length=false'  # Skip length check for performance
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode == 0:
        print("\nDownload complete!")
        return True
    else:
        print(f"\nDownload failed with code: {result.returncode}")
        return False


def download_with_python(sas_url: str, output_dir: str):
    """
    Alternative: Download using Azure Python SDK (slower but no AzCopy needed).
    """
    try:
        from azure.storage.blob import BlobServiceClient
    except ImportError:
        print("Azure Storage SDK not installed.")
        print("Install with: pip install azure-storage-blob")
        return False
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading using Azure Python SDK to: {output_path}")
    
    # Create blob service client from SAS URL
    blob_service_client = BlobServiceClient(account_url=sas_url)
    
    # List containers and download
    # Note: This is a simplified example - actual implementation depends on 
    # the container structure of the Bone Age dataset
    
    print("Azure SDK download not fully implemented in this script.")
    print("Please use AzCopy for best performance.")
    return False


def main():
    """Main download function."""
    print("="*80)
    print("Bone Age X-ray Dataset Download")
    print("="*80)
    
    # Configuration
    output_dir = "data/bone_age"
    
    print(f"\nOutput directory: {output_dir}")
    print("\nYou need the Azure SAS URL from the dataset provider.")
    print("This URL looks like:")
    print("  https://<account>.blob.core.windows.net/<container>?<sas-token>")
    print("\nThe SAS token expires after a limited time (usually 24-48 hours).")
    
    # Get SAS URL from user
    print("\n" + "-"*80)
    sas_url = input("\nPaste your SAS URL here: ").strip()
    
    if not sas_url:
        print("No URL provided. Exiting.")
        return
    
    if not sas_url.startswith('http'):
        print("Invalid URL. Must start with http or https.")
        return
    
    # Check for AzCopy
    if check_azcopy():
        print("\nUsing AzCopy for download...")
        success = download_with_azcopy(sas_url, output_dir)
    else:
        print("\nAzCopy not found!")
        print("\nOptions:")
        print("1. Install AzCopy (recommended for large datasets):")
        print("   Windows: winget install Microsoft.AzCopy")
        print("   Linux:   wget -O azcopy.tar.gz https://aka.ms/downloadazcopy-v10-linux")
        print("   Mac:     brew install azcopy")
        print("\n2. Use Azure Storage Explorer (GUI):")
        print("   Download: https://azure.microsoft.com/en-us/products/storage/storage-explorer")
        print("\n3. Use Azure Python SDK (slower):")
        print("   pip install azure-storage-blob")
        
        response = input("\nTry Python SDK anyway? (y/N): ").lower()
        if response == 'y':
            success = download_with_python(sas_url, output_dir)
        else:
            success = False
    
    if success:
        print(f"\nDataset downloaded to: {output_dir}")
        print("\nNext steps:")
        print("1. Verify the download")
        print("2. Run: python demo/bone_age_demo.py")
    else:
        print("\nDownload failed. Please try Azure Storage Explorer as an alternative.")


if __name__ == "__main__":
    main()
