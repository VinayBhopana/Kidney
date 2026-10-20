from pathlib import Path
import hashlib
import shutil
from collections import Counter

# ============================================================
# CONFIGURATION
# ============================================================

SOURCE_DIR = Path(
    r"D:\Kidney\kidney US images\my dataset final 512x512(implemented)"
)

OUTPUT_DIR = Path(
    r"D:\Kidney\kidney US images\my dataset deduplicated"
)

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ============================================================
# SAFETY CHECK
# ============================================================

if OUTPUT_DIR.exists():
    print("\nERROR:")
    print(f"Output directory already exists:\n{OUTPUT_DIR}")
    print("\nDelete/rename it manually before running this script again.")
    raise SystemExit

OUTPUT_DIR.mkdir(parents=True)

# ============================================================
# COLLECT FILES
# ============================================================

images = []

for class_dir in SOURCE_DIR.iterdir():

    if not class_dir.is_dir():
        continue

    class_name = class_dir.name

    for file in class_dir.rglob("*"):

        if file.suffix.lower() in VALID_EXTENSIONS:
            images.append((file, class_name))

print("=" * 70)
print("CREATING DEDUPLICATED KIDNEY ULTRASOUND DATASET")
print("=" * 70)

print(f"\nOriginal images: {len(images)}")

# ============================================================
# CREATE CLASS DIRECTORIES
# ============================================================

classes = sorted(set(class_name for _, class_name in images))

for class_name in classes:
    (OUTPUT_DIR / class_name).mkdir(parents=True, exist_ok=True)

# ============================================================
# DEDUPLICATION
# ============================================================

seen_hashes = {}

copied_counts = Counter()
removed_counts = Counter()

duplicate_log = []

for file, class_name in images:

    # Calculate exact file hash
    with open(file, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()

    # First occurrence -> preserve it
    if file_hash not in seen_hashes:

        destination = OUTPUT_DIR / class_name / file.name

        # Protect against accidental filename collision
        if destination.exists():

            stem = file.stem
            suffix = file.suffix

            counter = 1

            while destination.exists():

                destination = (
                    OUTPUT_DIR
                    / class_name
                    / f"{stem}_{counter}{suffix}"
                )

                counter += 1

        shutil.copy2(file, destination)

        seen_hashes[file_hash] = (file, class_name)

        copied_counts[class_name] += 1

    # Subsequent occurrence -> duplicate
    else:

        original_file, original_class = seen_hashes[file_hash]

        removed_counts[class_name] += 1

        duplicate_log.append(
            {
                "kept": str(original_file),
                "removed": str(file),
                "class": class_name
            }
        )

# ============================================================
# VERIFY OUTPUT
# ============================================================

output_counts = Counter()

for class_dir in OUTPUT_DIR.iterdir():

    if class_dir.is_dir():

        output_counts[class_dir.name] = sum(
            1
            for file in class_dir.rglob("*")
            if file.suffix.lower() in VALID_EXTENSIONS
        )

# ============================================================
# SAVE DUPLICATE LOG
# ============================================================

log_file = OUTPUT_DIR / "duplicate_removal_log.txt"

with open(log_file, "w", encoding="utf-8") as f:

    f.write("KIDNEY ULTRASOUND EXACT DUPLICATE REMOVAL LOG\n")
    f.write("=" * 70 + "\n\n")

    for entry in duplicate_log:

        f.write(f"Class   : {entry['class']}\n")
        f.write(f"Kept    : {entry['kept']}\n")
        f.write(f"Removed : {entry['removed']}\n")
        f.write("-" * 70 + "\n")

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DEDUPLICATION SUMMARY")
print("=" * 70)

for class_name in classes:

    original = (
        copied_counts[class_name]
        + removed_counts[class_name]
    )

    print(f"\n{class_name}")
    print(f"  Original : {original}")
    print(f"  Unique   : {copied_counts[class_name]}")
    print(f"  Removed  : {removed_counts[class_name]}")

print("\n" + "-" * 70)

print(f"Original total : {len(images)}")
print(f"Unique total   : {sum(copied_counts.values())}")
print(f"Removed total  : {sum(removed_counts.values())}")

print("\nOutput verification:")

for class_name, count in output_counts.items():
    print(f"{class_name}: {count}")

print(f"\nDuplicate log saved to:")
print(log_file)

print("\nDeduplicated dataset created successfully.")