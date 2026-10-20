from pathlib import Path
from collections import defaultdict, Counter
import hashlib
import csv


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = Path(
    r"D:\Kidney\kidney-stone-ultrasound.v4i.yolov8"
)

OUTPUT_DIR = DATASET_DIR / "leakage_analysis"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
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
# FUNCTION: GET ORIGINAL ROBOfLOW SOURCE NAME
# ============================================================

def get_source_name(filename):
    """
    Example:

    0001-N52FM9G2_jpg.rf.0101ebf5f7a643f7e8da3500745aff68.jpg

    becomes

    0001-N52FM9G2_jpg
    """

    name = Path(filename).stem

    if ".rf." in name:
        return name.split(".rf.")[0]

    return name


# ============================================================
# FUNCTION: SHA256
# ============================================================

def calculate_sha256(file_path):

    sha = hashlib.sha256()

    with open(file_path, "rb") as f:

        while True:

            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            sha.update(chunk)

    return sha.hexdigest()


# ============================================================
# COLLECT IMAGES
# ============================================================

records = []

source_groups = defaultdict(list)

hash_groups = defaultdict(list)

split_counts = Counter()


print()
print("=" * 75)
print("ECCKD TRAIN / VALID / TEST LEAKAGE ANALYSIS")
print("=" * 75)


for split in ["train", "valid", "test"]:

    images_dir = (
        DATASET_DIR /
        split /
        "images"
    )

    print(
        f"\nProcessing {split}..."
    )

    for image_path in images_dir.iterdir():

        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        source_name = get_source_name(
            image_path.name
        )

        sha256 = calculate_sha256(
            image_path
        )

        record = {
            "split": split,
            "filename": image_path.name,
            "source_name": source_name,
            "sha256": sha256,
            "path": str(image_path)
        }

        records.append(record)

        source_groups[
            source_name
        ].append(record)

        hash_groups[
            sha256
        ].append(record)

        split_counts[
            split
        ] += 1


# ============================================================
# SOURCE GROUP ANALYSIS
# ============================================================

cross_split_source_groups = {}

same_split_repeated_sources = {}

for source_name, group in source_groups.items():

    splits = {
        item["split"]
        for item in group
    }

    if len(splits) > 1:

        cross_split_source_groups[
            source_name
        ] = group

    elif len(group) > 1:

        same_split_repeated_sources[
            source_name
        ] = group


# ============================================================
# EXACT HASH DUPLICATE ANALYSIS
# ============================================================

exact_duplicate_groups = {}

cross_split_exact_groups = {}

for hash_value, group in hash_groups.items():

    if len(group) <= 1:
        continue

    exact_duplicate_groups[
        hash_value
    ] = group

    splits = {
        item["split"]
        for item in group
    }

    if len(splits) > 1:

        cross_split_exact_groups[
            hash_value
        ] = group


# ============================================================
# COUNT AFFECTED IMAGES
# ============================================================

images_in_cross_split_source_groups = sum(
    len(group)
    for group in cross_split_source_groups.values()
)

images_in_cross_split_exact_groups = sum(
    len(group)
    for group in cross_split_exact_groups.values()
)


unique_source_names = len(
    source_groups
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n")
print("=" * 75)
print("DATASET SUMMARY")
print("=" * 75)

print(
    f"\nTotal images: {len(records)}"
)

for split in ["train", "valid", "test"]:

    print(
        f"{split.capitalize():<10}: "
        f"{split_counts[split]}"
    )

print(
    f"\nUnique apparent source filenames: "
    f"{unique_source_names}"
)


print("\n")
print("=" * 75)
print("SOURCE-FILENAME LEAKAGE CHECK")
print("=" * 75)

print(
    f"\nSource groups appearing in more "
    f"than one split: "
    f"{len(cross_split_source_groups)}"
)

print(
    f"Images belonging to those groups: "
    f"{images_in_cross_split_source_groups}"
)

print(
    f"Repeated source groups contained "
    f"within one split only: "
    f"{len(same_split_repeated_sources)}"
)


# ============================================================
# SPLIT COMBINATION DISTRIBUTION
# ============================================================

split_combination_counts = Counter()

for group in cross_split_source_groups.values():

    splits = tuple(
        sorted(
            {
                item["split"]
                for item in group
            }
        )
    )

    split_combination_counts[
        splits
    ] += 1


print(
    "\nCross-split source-group distribution:"
)

for combination, count in (
    split_combination_counts.most_common()
):

    print(
        f"  {' + '.join(combination)}: "
        f"{count}"
    )


# ============================================================
# EXACT DUPLICATE SUMMARY
# ============================================================

print("\n")
print("=" * 75)
print("EXACT IMAGE DUPLICATE CHECK")
print("=" * 75)

print(
    f"\nExact duplicate groups overall: "
    f"{len(exact_duplicate_groups)}"
)

print(
    f"Exact duplicate groups crossing splits: "
    f"{len(cross_split_exact_groups)}"
)

print(
    f"Images involved in cross-split "
    f"exact duplicates: "
    f"{images_in_cross_split_exact_groups}"
)


# ============================================================
# SHOW EXAMPLES
# ============================================================

if cross_split_source_groups:

    print("\n")
    print("=" * 75)
    print("EXAMPLE CROSS-SPLIT SOURCE GROUPS")
    print("=" * 75)

    for index, (
        source_name,
        group
    ) in enumerate(
        list(
            cross_split_source_groups.items()
        )[:15],
        start=1
    ):

        print(
            f"\n[{index}] Source: {source_name}"
        )

        for item in group:

            print(
                f"    {item['split']:<6} "
                f"{item['filename']}"
            )


# ============================================================
# EXPORT ALL CROSS-SPLIT SOURCE GROUPS
# ============================================================

source_csv = (
    OUTPUT_DIR /
    "cross_split_source_groups.csv"
)

with open(
    source_csv,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow(
        [
            "source_name",
            "split",
            "filename",
            "sha256",
            "path"
        ]
    )

    for source_name, group in (
        cross_split_source_groups.items()
    ):

        for item in group:

            writer.writerow(
                [
                    source_name,
                    item["split"],
                    item["filename"],
                    item["sha256"],
                    item["path"]
                ]
            )


# ============================================================
# EXPORT EXACT DUPLICATES
# ============================================================

exact_csv = (
    OUTPUT_DIR /
    "cross_split_exact_duplicates.csv"
)

with open(
    exact_csv,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow(
        [
            "sha256",
            "split",
            "filename",
            "source_name",
            "path"
        ]
    )

    for hash_value, group in (
        cross_split_exact_groups.items()
    ):

        for item in group:

            writer.writerow(
                [
                    hash_value,
                    item["split"],
                    item["filename"],
                    item["source_name"],
                    item["path"]
                ]
            )


# ============================================================
# FINAL INTERPRETATION
# ============================================================

print("\n")
print("=" * 75)
print("PRELIMINARY INTERPRETATION")
print("=" * 75)


if len(cross_split_source_groups) == 0:

    print(
        "\nPASS: No repeated apparent source filenames "
        "were found across train/valid/test."
    )

else:

    percentage = (
        images_in_cross_split_source_groups
        / len(records)
    ) * 100

    print(
        "\nWARNING:"
    )

    print(
        f"{len(cross_split_source_groups)} source groups "
        f"appear in multiple dataset splits."
    )

    print(
        f"{images_in_cross_split_source_groups} images "
        f"({percentage:.2f}% of the dataset) belong "
        f"to these cross-split groups."
    )

    print(
        "\nThe supplied Roboflow train/valid/test split "
        "should NOT yet be assumed independent."
    )


print(
    "\nReports saved to:"
)

print(
    OUTPUT_DIR
)

print()
print("=" * 75)
print("LEAKAGE CHECK COMPLETE")
print("=" * 75)