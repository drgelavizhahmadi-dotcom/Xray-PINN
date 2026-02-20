"""
Download LERA (Lower Extremity Radiographs) Dataset from Azure
"""

from azure.storage.blob import ContainerClient
from pathlib import Path
from tqdm import tqdm


def download_lera(sas_url: str, output_dir: str = "data/lera"):
    """
    Download LERA dataset from Azure Blob Storage.
    Handles large files (like zip archives) with chunked download.
    
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
    
    if len(blobs) == 0:
        print("No files found. Check SAS URL.")
        return 0, 0
    
    # Download each blob with progress
    downloaded = 0
    failed = []
    
    for blob in blobs:
        blob_name = blob.name
        local_path = output_path / blob_name
        
        # Create subdirectories if needed
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if already exists
        if local_path.exists() and local_path.stat().st_size == blob.size:
            print(f"Skipping {blob_name} (already downloaded)")
            downloaded += 1
            continue
        
        print(f"\nDownloading: {blob_name}")
        print(f"Size: {blob.size / 1024 / 1024:.2f} MB")
        
        try:
            blob_client = container_client.get_blob_client(blob_name)
            
            # Chunked download for large files
            chunk_size = 4 * 1024 * 1024  # 4MB chunks
            offset = 0
            
            with open(local_path, 'wb') as f:
                while offset < blob.size:
                    # Download chunk
                    end = min(offset + chunk_size - 1, blob.size - 1)
                    download_stream = blob_client.download_blob(
                        offset=offset,
                        length=chunk_size
                    )
                    data = download_stream.readall()
                    f.write(data)
                    
                    offset += len(data)
                    
                    # Progress
                    percent = (offset / blob.size) * 100
                    mb_done = offset / 1024 / 1024
                    mb_total = blob.size / 1024 / 1024
                    print(f"\r  Progress: {percent:.1f}% ({mb_done:.1f}/{mb_total:.1f} MB)", end='')
            
            print(f"\n  Complete: {blob_name}")
            downloaded += 1
            
            # If it's a zip, offer to extract
            if blob_name.endswith('.zip'):
                print(f"\n  ZIP file downloaded. Extract?")
                response = input("  Extract now? (y/N): ").lower()
                if response == 'y':
                    import zipfile
                    print(f"  Extracting {blob_name}...")
                    with zipfile.ZipFile(local_path, 'r') as zip_ref:
                        zip_ref.extractall(output_path)
                    print(f"  Extraction complete!")
                    
                    # Offer to delete zip
                    response = input("  Delete ZIP file to save space? (y/N): ").lower()
                    if response == 'y':
                        local_path.unlink()
                        print(f"  ZIP file deleted.")
            
        except Exception as e:
            failed.append((blob_name, str(e)))
            print(f"\n  Error: {e}")
    
    print(f"\n\nDownload complete!")
    print(f"  Successfully downloaded: {downloaded}/{len(blobs)}")
    
    if failed:
        print(f"  Failed: {len(failed)}")
        for name, error in failed:
            print(f"    - {name}: {error}")
    
    return downloaded, len(blobs)


if __name__ == "__main__":
    import sys
    
    print("="*80)
    print("LERA - Lower Extremity Radiographs Dataset Download")
    print("="*80)
    
    # SAS URL from command line or use the provided one
    if len(sys.argv) > 1:
        sas_url = sys.argv[1]
    else:
        # Use the provided URL
        sas_url = "https://aimistanforddatasets01.blob.core.windows.net/leralowerextremityradiographs-6?sv=2019-02-02&sr=c&sig=dIHvY3CfkXH6re6kEEbLZJEw%2FvCOEDX0atwV2dqGtys%3D&st=2026-02-20T19%3A16%3A33Z&se=2026-03-22T19%3A21%3A33Z&sp=rl"
    
    print(f"SAS URL: {sas_url[:80]}...")
    print("This may take several minutes depending on dataset size.")
    print("Press Ctrl+C to cancel.\n")
    
    try:
        downloaded, total = download_lera(sas_url)
        print(f"\nDataset ready at: data/lera/")
    except KeyboardInterrupt:
        print("\n\nDownload cancelled by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
