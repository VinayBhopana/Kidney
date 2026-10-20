from pathlib import Path
from PIL import Image
import hashlib
from collections import Counter

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = Path(r"D:\Kidney\kidney US images\my dataset final 512x512(implemented)")

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ============================================================
# SCAN DATASET
# ============================================================

images = []

for class_dir in DATASET_DIR.iterdir():

    if not class_dir.is_dir():
        continue

    class_name = class_dir.name

    for file in class_dir.rglob("*"):

        if file.suffix.lower() in VALID_EXTENSIONS:
            images.append((file, class_name))

print("\n" + "=" * 60)
print("KIDNEY ULTRASOUND DATASET AUDIT")
print("=" * 60)

print(f"\nDataset path:")
print(DATASET_DIR)

print(f"\nTotal images: {len(images)}")

# ============================================================
# CLASS DISTRIBUTION
# ============================================================

class_counts = Counter(class_name for _, class_name in images)

print("\nCLASS DISTRIBUTION")
print("-" * 30)

for class_name, count in class_counts.items():
    print(f"{class_name}: {count}")

# ============================================================
# IMAGE PROPERTIES
# ============================================================

dimensions = Counter()
modes = Counter()

corrupted = []

for file, class_name in images:

    try:
        with Image.open(file) as img:

            # Force image loading
            img.load()

            dimensions[img.size] += 1
            modes[img.mode] += 1

    except Exception as e:

        corrupted.append((str(file), str(e)))

print("\nIMAGE DIMENSIONS")
print("-" * 30)

for dimension, count in dimensions.most_common():
    print(f"{dimension}: {count}")

print("\nIMAGE COLOR MODES")
print("-" * 30)

for mode, count in modes.most_common():
    print(f"{mode}: {count}")

# ============================================================
# CORRUPTED FILES
# ============================================================

print("\nCORRUPTED IMAGES")
print("-" * 30)

print(f"Corrupted images: {len(corrupted)}")

for file, error in corrupted[:20]:
    print(file)
    print(error)

# ============================================================
# EXACT DUPLICATE DETECTION
# ============================================================

print("\nCHECKING FOR EXACT DUPLICATES...")
print("-" * 30)

hash_map = {}

for file, class_name in images:

    try:
        with open(file, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()

        if file_hash not in hash_map:
            hash_map[file_hash] = []

        hash_map[file_hash].append((file, class_name))

    except Exception:
        pass

duplicate_groups = [
    files for files in hash_map.values()
    if len(files) > 1
]

duplicate_images = sum(
    len(group) - 1
    for group in duplicate_groups
)

print(f"Duplicate groups: {len(duplicate_groups)}")
print(f"Duplicate images: {duplicate_images}")

if duplicate_groups:

    print("\nEXAMPLE DUPLICATES:")

    for group in duplicate_groups[:10]:

        print("\nGroup:")

        for file, class_name in group:
            print(f"  [{class_name}] {file}")

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print(f"Total images       : {len(images)}")

for class_name, count in class_counts.items():
    print(f"{class_name:<20}: {count}")

print(f"Corrupted images   : {len(corrupted)}")
print(f"Duplicate groups   : {len(duplicate_groups)}")
print(f"Duplicate images   : {duplicate_images}")

print("\nAudit complete.")