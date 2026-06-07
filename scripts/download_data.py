"""Download and organize Kaggle deepfake dataset."""
import kagglehub
import shutil
import os
from pathlib import Path

def main():
    print("Downloading dataset from Kaggle...")
    print("This may take 10-15 minutes (2-3 GB)...")
    
    # Download dataset
    path = kagglehub.dataset_download("yuvrajpaikhot/extracted-deepfake-frames")
    print(f"Downloaded to: {path}")
    
    dataset_root = Path(path)
    
    # Find the actual data folder (sometimes nested one level)
    for subdir in dataset_root.iterdir():
        if subdir.is_dir() and any(d.name.startswith(("train", "val", "test")) for d in subdir.iterdir() if d.is_dir()):
            dataset_root = subdir
            break
    
    # Target structure
    target = Path("data/processed")
    target.mkdir(parents=True, exist_ok=True)
    
    # Copy splits
    for split in ["train", "val", "test"]:
        src = dataset_root / split
        if src.exists():
            dst = target / split
            if dst.exists():
                print(f"Removing existing {dst}")
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            print(f"Copied {split}: {src} -> {dst}")
            
            # Show what's inside
            for label_dir in sorted(dst.iterdir()):
                if label_dir.is_dir():
                    count = len(list(label_dir.glob("*.jpg")) + list(label_dir.glob("*.png")))
                    print(f"  - {label_dir.name}: {count} images")
        else:
            print(f"Warning: {split} not found at {src}")
    
    print("\nDone. Verify with: ls data/processed -Recurse")

if __name__ == "__main__":
    main()