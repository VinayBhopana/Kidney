from pathlib import Path
from collections import defaultdict, Counter
from PIL import Image
import hashlib
import csv


# ============================================================
# CONFIGURATION
# ============================================================

# Dataset A:
# Your ORIGINAL/CLEANED kidney ultrasound dataset.
#
# Change this path ONLY if your cleaned dataset is actually
# stored somewhere else.
ORIGINAL_DATASET_DIR = Path(
    r"D:\Kidney\kidney US images"
)

# Dataset B:
# Newly cleaned ECCKD group-aware split
ECCKD_DATASET_DIR = Path(
    r"D:\Kidney\ECCKD_group_split"
)

OUTPUT_DIR = Path(
    r"D:\Kidney\cross_dataset_analysis"
)

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
# HELPERS
# ============================================================

def find_images(root):
    """
    Recursively locate all image files.
    """

    images = []

    for path in root.rglob("*"):

        if (
            path.is_file()
            and
            path.suffix.lower() in IMAGE_EXTENSIONS
        ):
            images.append(path)

    return sorted(images)


def file_sha256(path):
    """
    SHA256 of raw file bytes.
    """

    sha = hashlib.sha256()

    with open(path, "rb") as f:

        while True:

            chunk = f.read(
                1024 * 1024
            )

            if not chunk:
                break

            sha.update(chunk)

    return sha.hexdigest()


def pixel_sha256(path):
    """
    SHA256 of decoded image pixels.

    This ignores file metadata and JPEG container differences.

    Width and height are included in the hash so images with
    different dimensions cannot accidentally match.
    """

    with Image.open(path) as img:

        img = img.convert("RGB")

        width, height = img.size

        pixel_bytes = img.tobytes()

    sha = hashlib.sha256()

    sha.update(
        str(width).encode()
    )

    sha.update(b"x")

    sha.update(
        str(height).encode()
    )

    sha.update(b"|RGB|")

    sha.update(
        pixel_bytes
    )

    return sha.hexdigest()


def infer_class_from_path(path):
    """
    Attempt to infer Normal/Stone from folder names.

    This is only for reporting.
    """

    parts = [
        part.lower()
        for part in path.parts
    ]

    for part in reversed(parts):

        if "normal" in part:
            return "normal"

        if "stone" in part:
            return "stone"

    return "unknown"


def get_ecckd_split(path):
    """
    Infer ECCKD split from path.
    """

    parts = [
        part.lower()
        for part in path.parts
    ]

    for split in [
        "train",
        "val",
        "test"
    ]:

        if split in parts:
            return split

    return "unknown"


# ============================================================
# VALIDATE PATHS
# ============================================================

print()
print("=" * 78)
print("CROSS-DATASET EXACT OVERLAP ANALYSIS")
print("=" * 78)


if not ORIGINAL_DATASET_DIR.exists():

    raise FileNotFoundError(
        f"\nOriginal dataset not found:\n"
        f"{ORIGINAL_DATASET_DIR}"
    )


if not ECCKD_DATASET_DIR.exists():

    raise FileNotFoundError(
        f"\nECCKD dataset not found:\n"
        f"{ECCKD_DATASET_DIR}"
    )


# ============================================================
# DISCOVER IMAGES
# ============================================================

print("\nSearching Dataset A...")

original_images = find_images(
    ORIGINAL_DATASET_DIR
)

print("Searching Dataset B...")

ecckd_images = find_images(
    ECCKD_DATASET_DIR
)


print()
print("-" * 78)
print("DISCOVERED IMAGE COUNTS")
print("-" * 78)

print(
    f"\nDataset A images found: "
    f"{len(original_images)}"
)

print(
    f"Dataset B images found: "
    f"{len(ecckd_images)}"
)


# ============================================================
# IMPORTANT COUNT WARNING
# ============================================================

if len(original_images) != 8376:

    print()
    print("WARNING:")
    print(
        "Dataset A does not contain exactly "
        "8,376 discovered images."
    )

    print(
        "This may mean the selected directory contains "
        "extra generated images/results, or the cleaned "
        "dataset is stored elsewhere."
    )


if len(ecckd_images) != 8648:

    print()
    print("WARNING:")
    print(
        "Dataset B does not contain exactly "
        "8,648 discovered images."
    )


# ============================================================
# STORAGE
# ============================================================

original_file_hashes = defaultdict(
    list
)

ecckd_file_hashes = defaultdict(
    list
)

original_pixel_hashes = defaultdict(
    list
)

ecckd_pixel_hashes = defaultdict(
    list
)

errors = []


# ============================================================
# HASH DATASET A
# ============================================================

print()
print("=" * 78)
print("HASHING DATASET A")
print("=" * 78)


for index, path in enumerate(
    original_images,
    start=1
):

    try:

        file_hash = file_sha256(
            path
        )

        pixel_hash = pixel_sha256(
            path
        )

        record = {
            "path": str(path),
            "filename": path.name,
            "class": infer_class_from_path(
                path
            )
        }

        original_file_hashes[
            file_hash
        ].append(
            record
        )

        original_pixel_hashes[
            pixel_hash
        ].append(
            record
        )

    except Exception as e:

        errors.append(
            {
                "dataset": "A",
                "path": str(path),
                "error": str(e)
            }
        )

    if (
        index % 1000 == 0
        or
        index == len(original_images)
    ):

        print(
            f"Dataset A: "
            f"{index}/{len(original_images)}"
        )


# ============================================================
# HASH DATASET B
# ============================================================

print()
print("=" * 78)
print("HASHING DATASET B")
print("=" * 78)


for index, path in enumerate(
    ecckd_images,
    start=1
):

    try:

        file_hash = file_sha256(
            path
        )

        pixel_hash = pixel_sha256(
            path
        )

        record = {
            "path": str(path),
            "filename": path.name,
            "class": infer_class_from_path(
                path
            ),
            "split": get_ecckd_split(
                path
            )
        }

        ecckd_file_hashes[
            file_hash
        ].append(
            record
        )

        ecckd_pixel_hashes[
            pixel_hash
        ].append(
            record
        )

    except Exception as e:

        errors.append(
            {
                "dataset": "B",
                "path": str(path),
                "error": str(e)
            }
        )

    if (
        index % 1000 == 0
        or
        index == len(ecckd_images)
    ):

        print(
            f"Dataset B: "
            f"{index}/{len(ecckd_images)}"
        )


# ============================================================
# CROSS-DATASET RAW FILE MATCHES
# ============================================================

common_file_hashes = (
    set(
        original_file_hashes.keys()
    )
    &
    set(
        ecckd_file_hashes.keys()
    )
)


# ============================================================
# CROSS-DATASET PIXEL MATCHES
# ============================================================

common_pixel_hashes = (
    set(
        original_pixel_hashes.keys()
    )
    &
    set(
        ecckd_pixel_hashes.keys()
    )
)


# ============================================================
# BUILD RAW FILE MATCH RECORDS
# ============================================================

file_overlap_rows = []


for hash_value in common_file_hashes:

    for original_record in (
        original_file_hashes[
            hash_value
        ]
    ):

        for ecckd_record in (
            ecckd_file_hashes[
                hash_value
            ]
        ):

            file_overlap_rows.append(
                {
                    "hash":
                        hash_value,

                    "original_file":
                        original_record[
                            "filename"
                        ],

                    "original_class":
                        original_record[
                            "class"
                        ],

                    "original_path":
                        original_record[
                            "path"
                        ],

                    "ecckd_file":
                        ecckd_record[
                            "filename"
                        ],

                    "ecckd_class":
                        ecckd_record[
                            "class"
                        ],

                    "ecckd_split":
                        ecckd_record[
                            "split"
                        ],

                    "ecckd_path":
                        ecckd_record[
                            "path"
                        ]
                }
            )


# ============================================================
# BUILD PIXEL MATCH RECORDS
# ============================================================

pixel_overlap_rows = []


for hash_value in common_pixel_hashes:

    for original_record in (
        original_pixel_hashes[
            hash_value
        ]
    ):

        for ecckd_record in (
            ecckd_pixel_hashes[
                hash_value
            ]
        ):

            pixel_overlap_rows.append(
                {
                    "hash":
                        hash_value,

                    "original_file":
                        original_record[
                            "filename"
                        ],

                    "original_class":
                        original_record[
                            "class"
                        ],

                    "original_path":
                        original_record[
                            "path"
                        ],

                    "ecckd_file":
                        ecckd_record[
                            "filename"
                        ],

                    "ecckd_class":
                        ecckd_record[
                            "class"
                        ],

                    "ecckd_split":
                        ecckd_record[
                            "split"
                        ],

                    "ecckd_path":
                        ecckd_record[
                            "path"
                        ]
                }
            )


# ============================================================
# COUNT UNIQUE AFFECTED IMAGES
# ============================================================

affected_original_file = {
    row["original_path"]
    for row in file_overlap_rows
}

affected_ecckd_file = {
    row["ecckd_path"]
    for row in file_overlap_rows
}


affected_original_pixel = {
    row["original_path"]
    for row in pixel_overlap_rows
}

affected_ecckd_pixel = {
    row["ecckd_path"]
    for row in pixel_overlap_rows
}


# ============================================================
# CLASS AGREEMENT
# ============================================================

pixel_class_pairs = Counter()


for row in pixel_overlap_rows:

    pixel_class_pairs[
        (
            row["original_class"],
            row["ecckd_class"]
        )
    ] += 1


# ============================================================
# SAVE FILE OVERLAPS
# ============================================================

file_csv = (
    OUTPUT_DIR /
    "cross_dataset_exact_file_matches.csv"
)


with open(
    file_csv,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    fieldnames = [
        "hash",
        "original_file",
        "original_class",
        "original_path",
        "ecckd_file",
        "ecckd_class",
        "ecckd_split",
        "ecckd_path"
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(
        file_overlap_rows
    )


# ============================================================
# SAVE PIXEL OVERLAPS
# ============================================================

pixel_csv = (
    OUTPUT_DIR /
    "cross_dataset_exact_pixel_matches.csv"
)


with open(
    pixel_csv,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    fieldnames = [
        "hash",
        "original_file",
        "original_class",
        "original_path",
        "ecckd_file",
        "ecckd_class",
        "ecckd_split",
        "ecckd_path"
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(
        pixel_overlap_rows
    )


# ============================================================
# SAVE ERRORS
# ============================================================

error_csv = (
    OUTPUT_DIR /
    "cross_dataset_hash_errors.csv"
)


with open(
    error_csv,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    fieldnames = [
        "dataset",
        "path",
        "error"
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(
        errors
    )


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 78)
print("CROSS-DATASET RESULTS")
print("=" * 78)


print(
    f"\nDataset A images analysed: "
    f"{len(original_images)}"
)

print(
    f"Dataset B images analysed: "
    f"{len(ecckd_images)}"
)

print(
    f"Read/hash errors: "
    f"{len(errors)}"
)


print()
print("-" * 78)
print("RAW FILE SHA-256")
print("-" * 78)

print(
    f"\nShared file hashes: "
    f"{len(common_file_hashes)}"
)

print(
    f"Dataset A images involved: "
    f"{len(affected_original_file)}"
)

print(
    f"Dataset B images involved: "
    f"{len(affected_ecckd_file)}"
)


print()
print("-" * 78)
print("DECODED PIXEL SHA-256")
print("-" * 78)

print(
    f"\nShared pixel hashes: "
    f"{len(common_pixel_hashes)}"
)

print(
    f"Dataset A images involved: "
    f"{len(affected_original_pixel)}"
)

print(
    f"Dataset B images involved: "
    f"{len(affected_ecckd_pixel)}"
)


if pixel_class_pairs:

    print(
        "\nClass combinations among "
        "pixel-identical matches:"
    )

    for (
        original_class,
        ecckd_class
    ), count in (
        pixel_class_pairs.most_common()
    ):

        print(
            f"  "
            f"{original_class}"
            f" -> "
            f"{ecckd_class}"
            f": "
            f"{count}"
        )


# ============================================================
# SHOW EXAMPLES
# ============================================================

if pixel_overlap_rows:

    print()
    print("-" * 78)
    print("EXAMPLE PIXEL-IDENTICAL MATCHES")
    print("-" * 78)

    for index, row in enumerate(
        pixel_overlap_rows[:10],
        start=1
    ):

        print(
            f"\n[{index}]"
        )

        print(
            f"Original: "
            f"{row['original_file']}"
        )

        print(
            f"  Class: "
            f"{row['original_class']}"
        )

        print(
            f"ECCKD: "
            f"{row['ecckd_file']}"
        )

        print(
            f"  Class: "
            f"{row['ecckd_class']}"
        )

        print(
            f"  Split: "
            f"{row['ecckd_split']}"
        )


# ============================================================
# INTERPRETATION
# ============================================================

print()
print("=" * 78)
print("PRELIMINARY INTERPRETATION")
print("=" * 78)


if (
    len(common_file_hashes) == 0
    and
    len(common_pixel_hashes) == 0
):

    print(
        "\nNo byte-identical or decoded-pixel-identical "
        "images were detected between the datasets."
    )

    print(
        "\nThis does NOT yet prove that the datasets are "
        "independent."
    )

    print(
        "Resized, cropped, compressed, contrast-adjusted "
        "or otherwise transformed copies will require "
        "near-duplicate analysis."
    )

else:

    print(
        "\nWARNING: Exact cross-dataset overlap was detected."
    )

    print(
        "These samples must not be treated as independent "
        "external-validation data."
    )


print()
print(
    f"Reports saved to:\n"
    f"{OUTPUT_DIR}"
)

print()
print("=" * 78)
print("EXACT CROSS-DATASET CHECK COMPLETE")
print("=" * 78)