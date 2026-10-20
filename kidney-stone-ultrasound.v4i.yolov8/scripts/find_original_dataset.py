from pathlib import Path
from collections import defaultdict


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(
    r"D:\Kidney\kidney US images"
)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp"
}


# ============================================================
# STORAGE
# ============================================================

direct_counts = defaultdict(int)

recursive_counts = defaultdict(int)


# ============================================================
# COUNT IMAGES DIRECTLY INSIDE EACH DIRECTORY
# ============================================================

print()
print("=" * 75)
print("SEARCHING FOR ORIGINAL CLEANED DATASET")
print("=" * 75)

all_images = []


for path in ROOT.rglob("*"):

    if (
        path.is_file()
        and
        path.suffix.lower() in IMAGE_EXTENSIONS
    ):

        all_images.append(path)

        direct_counts[
            path.parent
        ] += 1


print(
    f"\nTotal images found under root: "
    f"{len(all_images)}"
)


# ============================================================
# CALCULATE RECURSIVE COUNTS
# ============================================================

for image_path in all_images:

    current = image_path.parent

    while True:

        recursive_counts[
            current
        ] += 1

        if current == ROOT:
            break

        if ROOT not in current.parents:
            break

        current = current.parent


# ============================================================
# SHOW DIRECT IMAGE FOLDERS
# ============================================================

print()
print("=" * 75)
print("FOLDERS CONTAINING IMAGES DIRECTLY")
print("=" * 75)


direct_sorted = sorted(
    direct_counts.items(),
    key=lambda x: x[1],
    reverse=True
)


for folder, count in direct_sorted:

    print(
        f"{count:>7} images  |  {folder}"
    )


# ============================================================
# FIND DIRECTORIES NEAR 8376
# ============================================================

print()
print("=" * 75)
print("DIRECTORIES WITH RECURSIVE COUNTS NEAR 8,376")
print("=" * 75)


TARGET = 8376

TOLERANCE = 1000


candidates = []


for folder, count in recursive_counts.items():

    difference = abs(
        count - TARGET
    )

    if difference <= TOLERANCE:

        candidates.append(
            (
                difference,
                count,
                folder
            )
        )


candidates.sort(
    key=lambda x: x[0]
)


if candidates:

    for difference, count, folder in candidates:

        print(
            f"{count:>7} images "
            f"| difference={difference:>4} "
            f"| {folder}"
        )

else:

    print(
        "\nNo directory contains approximately "
        "8,376 images recursively."
    )


# ============================================================
# LOOK FOR COMMON DATASET STRUCTURES
# ============================================================

print()
print("=" * 75)
print("POSSIBLE TRAIN / VAL / TEST DATASETS")
print("=" * 75)


for folder in sorted(
    recursive_counts.keys(),
    key=str
):

    child_names = {
        child.name.lower()
        for child in folder.iterdir()
        if child.is_dir()
    }

    has_train = (
        "train" in child_names
    )

    has_test = (
        "test" in child_names
    )

    has_val = (
        "val" in child_names
        or
        "valid" in child_names
        or
        "validation" in child_names
    )

    if has_train and has_test:

        print()
        print(
            f"Candidate dataset root:\n"
            f"{folder}"
        )

        print(
            f"Recursive image count: "
            f"{recursive_counts[folder]}"
        )

        print(
            f"Has validation folder: "
            f"{has_val}"
        )


# ============================================================
# SHOW TOP-LEVEL STRUCTURE
# ============================================================

print()
print("=" * 75)
print("TOP-LEVEL DIRECTORY SUMMARY")
print("=" * 75)


for child in sorted(
    ROOT.iterdir(),
    key=lambda p: p.name.lower()
):

    if child.is_dir():

        count = recursive_counts.get(
            child,
            0
        )

        print(
            f"{count:>7} images | "
            f"{child.name}"
        )


print()
print("=" * 75)
print("SEARCH COMPLETE")
print("=" * 75)