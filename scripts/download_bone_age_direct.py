"""
Download Bone Age X-ray Dataset using Azure Python SDK
"""

from azure.storage.blob import ContainerClient
from pathlib import Path
import os
from tqdm import tqdm


def download_bone_age(sas_url: str, output_dir: str = "data/bone_age"):
    """
    Download Bone Age dataset from Azure Blob Storage.
    
    Args:
        sas_url: Azure SAS URL
        output_dir: Where to save files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Connecting to Azure Blob Storage...")
    container_client = ContainerClient.from_container_url(sas_url)
    
    # List all blobs
    print("Listing files...")
    blobs = list(container_client.list_blobs())
    print(f"Found {len(blobs)} files to download")
    
    # Download each blob
    downloaded = 0
    failed = []
    
    for blob in tqdm(blobs, desc="Downloading"):
        blob_name = blob.name
        local_path = output_path / blob_name
        
        # Create subdirectories if needed
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Download blob
            blob_client = container_client.get_blob_client(blob_name)
            with open(local_path, 'wb') as f:
                download_stream = blob_client.download_blob()
                f.write(download_stream.readall())
            downloaded += 1
        except Exception as e:
            failed.append((blob_name, str(e)))
    
    print(f"\nDownload complete!")
    print(f"  Successfully downloaded: {downloaded}/{len(blobs)}")
    
    if failed:
        print(f"  Failed: {len(failed)}")
        for name, error in failed[:5]:  # Show first 5 failures
            print(f"    - {name}: {error}")
    
    return downloaded, len(blobs)


if __name__ == "__main__":
    import sys
    
    # SAS URL from command line or hardcoded
    if len(sys.argv) > 1:
        sas_url = sys.argv[1]
    else:
        # Use the provided URL
        sas_url = "https://aimistanforddatasets01.blob.core.windows.net/boneagexray?sv=2019-02-02&sr=c&sig=GNg%2BZjbCbGtUUU2IMjx8jStFYiUFw2ver3voGj1xeFM%3D&st=2026-02-20T19%3A10%3A18Z&se=2026-03-22T19%3A15%3A18Z&sp=rl"
    
    print("="*80)
    print("Bone Age X-ray Dataset Download")
    print("="*80)
    print(f"SAS URL: {sas_url[:80]}...")
    print("This will download ~8GB of data. Press Ctrl+C to cancel.")
    print()
    
    try:
        downloaded, total = download_bone_age(sas_url)
        print(f"\nDataset ready at: data/bone_age/")
    except KeyboardInterrupt:
        print("\n\nDownload cancelled by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
