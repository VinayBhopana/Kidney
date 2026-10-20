from pathlib import Path
from collections import Counter, defaultdict
from PIL import Image
import hashlib
import csv


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = Path(
    r"D:\Kidney\ECCKD_group_split"
)

MANIFEST_PATH = (
    DATASET_DIR /
    "group_split_manifest.csv"
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
# STORAGE
# ============================================================

split_stats = {}

hash_groups = defaultdict(list)

corrupted_images = []

global_dimensions = Counter()

global_modes = Counter()

global_formats = Counter()


# ============================================================
# AUDIT NEW SPLIT
# ============================================================

for split in ["train", "val", "test"]:

    split_total = 0

    class_counts = Counter()

    split_dimensions = Counter()

    split_modes = Counter()

    split_formats = Counter()

    print(
        f"\nProcessing {split}..."
    )

    for class_name in [
        "normal",
        "stone"
    ]:

        class_dir = (
            DATASET_DIR /
            split /
            class_name
        )

        image_files = [
            p
            for p in class_dir.iterdir()
            if p.suffix.lower()
            in IMAGE_EXTENSIONS
        ]

        class_counts[
            class_name
        ] = len(
            image_files
        )

        split_total += len(
            image_files
        )

        for image_path in image_files:

            # ----------------------------------------------
            # CORRUPTION CHECK
            # ----------------------------------------------

            try:

                with Image.open(
                    image_path
                ) as img:

                    img.verify()

                with Image.open(
                    image_path
                ) as img:

                    width, height = (
                        img.size
                    )

                    mode = img.mode

                    fmt = img.format

                    split_dimensions[
                        (width, height)
                    ] += 1

                    global_dimensions[
                        (width, height)
                    ] += 1

                    split_modes[
                        mode
                    ] += 1

                    global_modes[
                        mode
                    ] += 1

                    split_formats[
                        fmt
                    ] += 1

                    global_formats[
                        fmt
                    ] += 1

            except Exception as e:

                corrupted_images.append(
                    (
                        str(image_path),
                        str(e)
                    )
                )

                continue

            # ----------------------------------------------
            # EXACT HASH
            # ----------------------------------------------

            hash_value = (
                calculate_sha256(
                    image_path
                )
            )

            hash_groups[
                hash_value
            ].append(
                {
                    "split": split,
                    "class": class_name,
                    "path": str(
                        image_path
                    )
                }
            )

    split_stats[
        split
    ] = {

        "total":
            split_total,

        "class_counts":
            class_counts,

        "dimensions":
            split_dimensions,

        "modes":
            split_modes,

        "formats":
            split_formats
    }


# ============================================================
# EXACT DUPLICATE ANALYSIS
# ============================================================

duplicate_groups = {}

cross_split_duplicates = {}

cross_class_duplicates = {}


for hash_value, group in (
    hash_groups.items()
):

    if len(group) <= 1:
        continue

    duplicate_groups[
        hash_value
    ] = group

    splits = {
        item["split"]
        for item in group
    }

    classes = {
        item["class"]
        for item in group
    }

    if len(splits) > 1:

        cross_split_duplicates[
            hash_value
        ] = group

    if len(classes) > 1:

        cross_class_duplicates[
            hash_value
        ] = group


# ============================================================
# MANIFEST GROUP CHECK
# ============================================================

source_membership = defaultdict(
    set
)

manifest_rows = []


if MANIFEST_PATH.exists():

    with open(
        MANIFEST_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        reader = csv.DictReader(
            f
        )

        for row in reader:

            manifest_rows.append(
                row
            )

            source_membership[
                row[
                    "source_group"
                ]
            ].add(
                row[
                    "new_split"
                ]
            )


leaking_source_groups = {

    source:
        splits

    for source, splits
    in source_membership.items()

    if len(splits) > 1
}


# ============================================================
# PRINT SUMMARY
# ============================================================

print()
print("=" * 75)
print("ECCKD CLEAN SPLIT VERIFICATION")
print("=" * 75)


total_images = 0


for split in [
    "train",
    "val",
    "test"
]:

    stats = split_stats[
        split
    ]

    total_images += stats[
        "total"
    ]

    print()
    print("-" * 75)
    print(
        f"{split.upper()}"
    )
    print("-" * 75)

    print(
        f"Total images: "
        f"{stats['total']}"
    )

    print(
        f"Stone: "
        f"{stats['class_counts']['stone']}"
    )

    print(
        f"Normal: "
        f"{stats['class_counts']['normal']}"
    )

    stone_pct = (
        stats[
            "class_counts"
        ][
            "stone"
        ]
        /
        stats[
            "total"
        ]
        *
        100
    )

    normal_pct = (
        stats[
            "class_counts"
        ][
            "normal"
        ]
        /
        stats[
            "total"
        ]
        *
        100
    )

    print(
        f"Stone %: "
        f"{stone_pct:.2f}%"
    )

    print(
        f"Normal %: "
        f"{normal_pct:.2f}%"
    )

    print(
        "\nImage modes:"
    )

    for mode, count in (
        stats[
            "modes"
        ].items()
    ):

        print(
            f"  {mode}: "
            f"{count}"
        )

    print(
        "\nImage formats:"
    )

    for fmt, count in (
        stats[
            "formats"
        ].items()
    ):

        print(
            f"  {fmt}: "
            f"{count}"
        )

    print(
        "\nMost common dimensions:"
    )

    for dimensions, count in (
        stats[
            "dimensions"
        ].most_common(
            5
        )
    ):

        print(
            f"  "
            f"{dimensions[0]}"
            f" x "
            f"{dimensions[1]}"
            f": "
            f"{count}"
        )


# ============================================================
# GLOBAL VERIFICATION
# ============================================================

print()
print("=" * 75)
print("GLOBAL VERIFICATION")
print("=" * 75)

print(
    f"\nTotal images: "
    f"{total_images}"
)

print(
    f"Corrupted images: "
    f"{len(corrupted_images)}"
)

print(
    f"Exact duplicate groups: "
    f"{len(duplicate_groups)}"
)

print(
    f"Exact duplicate groups crossing splits: "
    f"{len(cross_split_duplicates)}"
)

print(
    f"Exact duplicate groups crossing classes: "
    f"{len(cross_class_duplicates)}"
)

print(
    f"Source groups crossing splits: "
    f"{len(leaking_source_groups)}"
)


# ============================================================
# FINAL STATUS
# ============================================================

print()
print("=" * 75)
print("FINAL STATUS")
print("=" * 75)


passed = True


if corrupted_images:
    passed = False

    print(
        "FAIL: Corrupted images found."
    )


if cross_split_duplicates:
    passed = False

    print(
        "FAIL: Exact duplicates cross splits."
    )


if cross_class_duplicates:
    passed = False

    print(
        "FAIL: Exact duplicates cross classes."
    )


if leaking_source_groups:
    passed = False

    print(
        "FAIL: Source groups cross splits."
    )


if passed:

    print(
        "\nPASS"
    )

    print(
        "The ECCKD classification split "
        "passes the current integrity checks."
    )

    print(
        "\nThe dataset is ready for the "
        "next stage of analysis."
    )


print()
print("=" * 75)
print("VERIFICATION COMPLETE")
print("=" * 75)