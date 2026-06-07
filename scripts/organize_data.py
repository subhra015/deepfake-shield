import shutil
from pathlib import Path

src = Path.home() / ".cache/kagglehub/datasets/yuvrajpaikhot/extracted-deepfake-frames/versions/1"
dst = Path("data/processed")

print(f"Source: {src}")
print(f"Destination: {dst}")

for subdir in src.rglob("*"):
    if subdir.is_dir() and any(d.name.startswith(("train", "val", "test")) for d in subdir.iterdir() if d.is_dir()):
        src = subdir
        break

print(f"Actual data root: {src}")

for split in ["train", "val", "test"]:
    src_split = src / split
    if src_split.exists():
        dst_split = dst / split
        if dst_split.exists():
            shutil.rmtree(dst_split)
        shutil.copytree(src_split, dst_split)
        print(f"Copied {split}")
        for label_dir in sorted(dst_split.iterdir()):
            if label_dir.is_dir():
                count = len(list(label_dir.glob("*.jpg")) + list(label_dir.glob("*.png")) + list(label_dir.glob("*.jpeg")))
                print(f"  - {label_dir.name}: {count} images")
    else:
        print(f"Warning: {split} not found")

print("Done.")
