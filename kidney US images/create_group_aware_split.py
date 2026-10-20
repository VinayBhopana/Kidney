from pathlib import Path
from collections import defaultdict, Counter
import csv
import random
import shutil

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = Path(
    r"D:\Kidney\kidney US images\my dataset deduplicated"
)

GROUP_CSV = (
    DATASET_DIR / "near_duplicate_groups.csv"
)

OUTPUT_DIR = Path(
    r"D:\Kidney\kidney US images\dataset_group_split"
)

MANIFEST_FILE = (
    OUTPUT_DIR / "split_manifest.csv"
)

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

RANDOM_SEED = 42

SPLIT_RATIOS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15
}

# ============================================================
# SAFETY CHECK
# ============================================================

if OUTPUT_DIR.exists():

    print("\nERROR:")
    print(
        f"Output folder already exists:\n"
        f"{OUTPUT_DIR}"
    )

    print(
        "\nDelete or rename it before "
        "running this script again."
    )

    raise SystemExit

# ============================================================
# LOAD NEAR-DUPLICATE GROUPS
# ============================================================

group_lookup = {}

with open(
    GROUP_CSV,
    "r",
    encoding="utf-8"
) as f:

    reader = csv.DictReader(f)

    for row in reader:

        path = str(
            Path(
                row["image_path"]
            ).resolve()
        )

        group_id = (
            f"ND_GROUP_{row['group_id']}"
        )

        group_lookup[path] = group_id

print("=" * 75)
print("GROUP-AWARE DATASET SPLITTING")
print("=" * 75)

print(
    f"\nImages mapped to near-duplicate groups: "
    f"{len(group_lookup)}"
)

# ============================================================
# COLLECT ALL IMAGES
# ============================================================

images = []

for class_dir in DATASET_DIR.iterdir():

    if not class_dir.is_dir():
        continue

    class_name = class_dir.name

    for file in class_dir.rglob("*"):

        if file.suffix.lower() in VALID_EXTENSIONS:

            images.append(
                {
                    "path": file.resolve(),
                    "class": class_name
                }
            )

print(
    f"Total images found: "
    f"{len(images)}"
)

# ============================================================
# BUILD ATOMIC UNITS
#
# Near-duplicate images -> one group/unit
# Ordinary image         -> singleton unit
#
# Nothing inside one unit can be split.
# ============================================================

units_by_class = defaultdict(
    lambda: defaultdict(list)
)

for record in images:

    path_string = str(
        record["path"]
    )

    class_name = record["class"]

    if path_string in group_lookup:

        unit_id = group_lookup[
            path_string
        ]

    else:

        unit_id = (
            "SINGLE::"
            + path_string
        )

    units_by_class[
        class_name
    ][unit_id].append(
        record["path"]
    )

# ============================================================
# DISPLAY UNIT STATISTICS
# ============================================================

print("\n" + "=" * 75)
print("GROUPING STATISTICS")
print("=" * 75)

for class_name, units in units_by_class.items():

    grouped_units = sum(
        1
        for unit_id in units
        if unit_id.startswith(
            "ND_GROUP_"
        )
    )

    singleton_units = sum(
        1
        for unit_id in units
        if unit_id.startswith(
            "SINGLE::"
        )
    )

    print(f"\n{class_name}")

    print(
        f"  Near-duplicate groups : "
        f"{grouped_units}"
    )

    print(
        f"  Singleton images      : "
        f"{singleton_units}"
    )

    print(
        f"  Total independent units: "
        f"{len(units)}"
    )

# ============================================================
# SPLIT FUNCTION
# ============================================================

random.seed(
    RANDOM_SEED
)

assignments = []

for class_name, unit_dictionary in units_by_class.items():

    units = list(
        unit_dictionary.items()
    )

    # Randomize group order reproducibly
    random.shuffle(units)

    total_images = sum(
        len(files)
        for _, files in units
    )

    targets = {
        split: total_images * ratio
        for split, ratio
        in SPLIT_RATIOS.items()
    }

    current_counts = {
        "train": 0,
        "val": 0,
        "test": 0
    }

    # ========================================================
    # ASSIGN COMPLETE GROUPS
    # ========================================================

    for unit_id, files in units:

        # Pick split that is currently
        # furthest below its target proportion.
        def fill_ratio(split):

            target = targets[split]

            if target == 0:
                return float("inf")

            return (
                current_counts[split]
                / target
            )

        chosen_split = min(
            SPLIT_RATIOS.keys(),
            key=fill_ratio
        )

        for file in files:

            assignments.append(
                {
                    "image_path": file,
                    "class": class_name,
                    "unit_id": unit_id,
                    "split": chosen_split,
                    "near_duplicate_group":
                        unit_id.startswith(
                            "ND_GROUP_"
                        )
                }
            )

        current_counts[
            chosen_split
        ] += len(files)

    print("\n" + "-" * 75)

    print(
        f"{class_name} split:"
    )

    for split in [
        "train",
        "val",
        "test"
    ]:

        count = current_counts[
            split
        ]

        percentage = (
            count / total_images
        ) * 100

        print(
            f"  {split:<5}: "
            f"{count:>5} "
            f"({percentage:.2f}%)"
        )

# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

classes = sorted(
    units_by_class.keys()
)

for split in SPLIT_RATIOS:

    for class_name in classes:

        (
            OUTPUT_DIR
            / split
            / class_name
        ).mkdir(
            parents=True,
            exist_ok=True
        )

# ============================================================
# COPY IMAGES
# ============================================================

print("\n" + "=" * 75)
print("COPYING IMAGES")
print("=" * 75)

for i, record in enumerate(
    assignments,
    start=1
):

    source = record[
        "image_path"
    ]

    destination = (
        OUTPUT_DIR
        / record["split"]
        / record["class"]
        / source.name
    )

    # Filename collision protection
    if destination.exists():

        stem = source.stem
        suffix = source.suffix

        counter = 1

        while destination.exists():

            destination = (
                OUTPUT_DIR
                / record["split"]
                / record["class"]
                / f"{stem}_{counter}{suffix}"
            )

            counter += 1

    shutil.copy2(
        source,
        destination
    )

    record[
        "output_path"
    ] = destination

    if i % 1000 == 0:

        print(
            f"Copied "
            f"{i}/{len(assignments)} "
            f"images..."
        )

# ============================================================
# SAVE MANIFEST
# ============================================================

with open(
    MANIFEST_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    fieldnames = [
        "original_path",
        "output_path",
        "class",
        "unit_id",
        "split",
        "near_duplicate_group"
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()

    for record in assignments:

        writer.writerow(
            {
                "original_path":
                    str(
                        record[
                            "image_path"
                        ]
                    ),

                "output_path":
                    str(
                        record[
                            "output_path"
                        ]
                    ),

                "class":
                    record["class"],

                "unit_id":
                    record["unit_id"],

                "split":
                    record["split"],

                "near_duplicate_group":
                    record[
                        "near_duplicate_group"
                    ]
            }
        )

# ============================================================
# VERIFY SPLIT COUNTS
# ============================================================

print("\n" + "=" * 75)
print("FINAL SPLIT DISTRIBUTION")
print("=" * 75)

split_class_counts = Counter()

for record in assignments:

    split_class_counts[
        (
            record["split"],
            record["class"]
        )
    ] += 1

for split in [
    "train",
    "val",
    "test"
]:

    print(
        f"\n{split.upper()}"
    )

    split_total = 0

    for class_name in classes:

        count = split_class_counts[
            (
                split,
                class_name
            )
        ]

        split_total += count

        print(
            f"  {class_name:<10}: "
            f"{count}"
        )

    print(
        f"  TOTAL     : "
        f"{split_total}"
    )

# ============================================================
# VERIFY GROUP LEAKAGE
# ============================================================

group_split_lookup = defaultdict(
    set
)

for record in assignments:

    if record[
        "near_duplicate_group"
    ]:

        group_split_lookup[
            record["unit_id"]
        ].add(
            record["split"]
        )

leaking_groups = {
    group: splits
    for group, splits
    in group_split_lookup.items()
    if len(splits) > 1
}

print("\n" + "=" * 75)
print("LEAKAGE CHECK")
print("=" * 75)

print(
    f"\nNear-duplicate groups "
    f"appearing in multiple splits: "
    f"{len(leaking_groups)}"
)

if leaking_groups:

    print(
        "\nWARNING: GROUP LEAKAGE DETECTED"
    )

    for group, splits in list(
        leaking_groups.items()
    )[:20]:

        print(
            group,
            splits
        )

else:

    print(
        "PASS: No near-duplicate "
        "group crosses train/val/test."
    )

# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 75)

print(
    f"Total images assigned: "
    f"{len(assignments)}"
)

print(
    "\nSplit manifest saved to:"
)

print(
    MANIFEST_FILE
)

print(
    "\nGroup-aware split completed."
)