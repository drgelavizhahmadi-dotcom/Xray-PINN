"""Download CheXpert dataset."""

from pathlib import Path


def main():
    """Download CheXpert data."""
    chexpert_dir = Path("./data/raw/chexpert")
    chexpert_dir.mkdir(parents=True, exist_ok=True)
    
    print("CheXpert dataset requires manual download.")
    print("\n1. Visit: https://stanfordmlgroup.github.io/competitions/chexpert/")
    print("2. Register and download CheXpert-v1.0-small.zip")
    print(f"3. Extract to: {chexpert_dir}")
    
    # Create placeholder
    (chexpert_dir / "README.txt").write_text(
        "Place CheXpert data in this directory.\n"
        "Download from: https://stanfordmlgroup.github.io/competitions/chexpert/\n"
    )


if __name__ == "__main__":
    main()
