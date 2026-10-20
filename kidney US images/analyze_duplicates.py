from pathlib import Path
from collections import Counter
import hashlib

# ============================================================
# CONFIG
# ============================================================

DATASET_DIR = Path(
    r"D:\Kidney\kidney US images\my dataset final 512x512(implemented)"
)

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ============================================================
# COLLECT IMAGES
# ============================================================

images = []

for class_dir in DATASET_DIR.iterdir():

    if not class_dir.is_dir():
        continue

    class_name = class_dir.name

    for file in class_dir.rglob("*"):

        if file.suffix.lower() in VALID_EXTENSIONS:
            images.append((file, class_name))

print("=" * 70)
print("DUPLICATE ANALYSIS")
print("=" * 70)

print(f"\nTotal images: {len(images)}")

# ============================================================
# HASH FILES
# ============================================================

hash_map = {}

for file, class_name in images:

    with open(file, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()

    if file_hash not in hash_map:
        hash_map[file_hash] = []

    hash_map[file_hash].append((file, class_name))

# ============================================================
# DUPLICATE GROUPS
# ============================================================

duplicate_groups = [
    group
    for group in hash_map.values()
    if len(group) > 1
]

print(f"Duplicate groups: {len(duplicate_groups)}")

total_duplicate_images = sum(
    len(group) - 1
    for group in duplicate_groups
)

print(f"Duplicate images removable: {total_duplicate_images}")

# ============================================================
# SAME-CLASS VS CROSS-CLASS
# ============================================================

same_class_groups = []
cross_class_groups = []

for group in duplicate_groups:

    classes = set(class_name for _, class_name in group)

    if len(classes) == 1:
        same_class_groups.append(group)

    else:
        cross_class_groups.append(group)

print("\n" + "=" * 70)
print("DUPLICATE TYPE")
print("=" * 70)

print(f"Same-class duplicate groups : {len(same_class_groups)}")
print(f"Cross-class duplicate groups: {len(cross_class_groups)}")

# ============================================================
# DUPLICATE COUNTS PER CLASS
# ============================================================

duplicate_extra_by_class = Counter()

for group in same_class_groups:

    class_name = group[0][1]

    duplicate_extra_by_class[class_name] += len(group) - 1

print("\n" + "=" * 70)
print("REMOVABLE DUPLICATES PER CLASS")
print("=" * 70)

for class_name, count in duplicate_extra_by_class.items():
    print(f"{class_name}: {count}")

# ============================================================
# GROUP SIZE DISTRIBUTION
# ============================================================

group_sizes = Counter(len(group) for group in duplicate_groups)

print("\n" + "=" * 70)
print("DUPLICATE GROUP SIZE DISTRIBUTION")
print("=" * 70)

for size, count in sorted(group_sizes.items()):
    print(f"{size} identical copies -> {count} groups")

# ============================================================
# CROSS-CLASS DUPLICATES
# ============================================================

print("\n" + "=" * 70)
print("CROSS-CLASS DUPLICATES")
print("=" * 70)

if len(cross_class_groups) == 0:

    print("No exact duplicate images appear in different classes.")

else:

    print(
        f"WARNING: {len(cross_class_groups)} duplicate groups "
        f"contain conflicting class labels."
    )

    for i, group in enumerate(cross_class_groups[:20], start=1):

        print(f"\nConflict group {i}")

        for file, class_name in group:
            print(f"[{class_name}] {file}")

# ============================================================
# ESTIMATE DATASET AFTER SAFE SAME-CLASS DEDUPLICATION
# ============================================================

original_class_counts = Counter(
    class_name for _, class_name in images
)

estimated_counts = original_class_counts.copy()

for class_name, duplicate_count in duplicate_extra_by_class.items():
    estimated_counts[class_name] -= duplicate_count

print("\n" + "=" * 70)
print("ESTIMATED DATASET AFTER SAME-CLASS DEDUPLICATION")
print("=" * 70)

for class_name in original_class_counts:

    print(
        f"{class_name}: "
        f"{original_class_counts[class_name]} "
        f"-> "
        f"{estimated_counts[class_name]}"
    )

print(
    "\nEstimated total after same-class deduplication:",
    sum(estimated_counts.values())
)

# ============================================================
# EXAMPLE DUPLICATE GROUPS
# ============================================================

print("\n" + "=" * 70)
print("EXAMPLE SAME-CLASS DUPLICATE GROUPS")
print("=" * 70)

for i, group in enumerate(same_class_groups[:10], start=1):

    print(f"\nGroup {i}")

    for file, class_name in group:
        print(f"[{class_name}] {file}")

print("\nAnalysis complete.")