"""
Download MURA Dataset from Azure
"""

from azure.storage.blob import ContainerClient
from pathlib import Path
import sys


def download_mura(sas_url: str, output_dir: str = "data/mura"):
    """Download MURA dataset (single ZIP file)."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Connecting to Azure...")
    container_client = ContainerClient.from_container_url(sas_url)
    
    blobs = list(container_client.list_blobs())
    print(f"Found {len(blobs)} files")
    
    for blob in blobs:
        local_path = output_path / blob.name
        
        # Check if already exists
        if local_path.exists() and local_path.stat().st_size == blob.size:
            print(f"Already downloaded: {blob.name}")
            continue
        
        print(f"\nDownloading: {blob.name}")
        print(f"Size: {blob.size / 1024 / 1024:.1f} MB")
        print("This will take 10-20 minutes...")
        
        try:
            blob_client = container_client.get_blob_client(blob.name)
            
            # Stream download with progress
            with open(local_path, 'wb') as f:
                download_stream = blob_client.download_blob()
                downloaded = 0
                chunk_size = 4 * 1024 * 1024  # 4MB
                
                for chunk in download_stream.chunks():
                    f.write(chunk)
                    downloaded += len(chunk)
                    percent = (downloaded / blob.size) * 100
                    print(f"\r  Progress: {percent:.1f}%", end='')
            
            print(f"\n  Complete: {local_path}")
            
            # Ask to extract
            if blob.name.endswith('.zip'):
                print("\nExtract ZIP file?")
                response = input("Extract now? (y/N): ").lower()
                if response == 'y':
                    import zipfile
                    print("Extracting...")
                    with zipfile.ZipFile(local_path, 'r') as zf:
                        zf.extractall(output_path)
                    print("Extraction complete!")
                    
                    response = input("Delete ZIP to save space? (y/N): ").lower()
                    if response == 'y':
                        local_path.unlink()
                        print("ZIP deleted.")
        
        except Exception as e:
            print(f"Error: {e}")
            return False
    
    return True


if __name__ == "__main__":
    print("="*60)
    print("MURA Dataset Download")
    print("="*60)
    
    # Default URL or from command line
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        url = "https://aimistanforddatasets01.blob.core.windows.net/muramskxrays?sv=2019-02-02&sr=c&sig=64nQp3ksRZnIlAi2Fv6MwDBCmBiHmFqmUM2TCH7SlL8%3D&st=2026-02-20T19%3A56%3A12Z&se=2026-03-22T20%3A01%3A12Z&sp=rl"
    
    success = download_mura(url)
    if success:
        print("\nMURA dataset ready!")
        print("Next: python scripts/prepare_mura.py")
