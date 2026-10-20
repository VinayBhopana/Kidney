from pathlib import Path
from collections import defaultdict, Counter
import random
import shutil
import csv
import yaml


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = Path(
    r"D:\Kidney\kidney-stone-ultrasound.v4i.yolov8"
)

OUTPUT_DIR = Path(
    r"D:\Kidney\ECCKD_group_split"
)

SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png"
}

random.seed(SEED)


# ============================================================
# LOAD CLASS MAP
# ============================================================

with open(
    DATASET_DIR / "data.yaml",
    "r",
    encoding="utf-8"
) as f:

    config = yaml.safe_load(f)

names = config["names"]

if isinstance(names, list):

    CLASS_MAP = {
        i: name
        for i, name in enumerate(names)
    }

else:

    CLASS_MAP = {
        int(k): v
        for k, v in names.items()
    }


# ============================================================
# HELPERS
# ============================================================

def get_source_name(filename):
    """
    Strip Roboflow suffix.

    Example:
    abc_jpg.rf.xxxxx.jpg
        ->
    abc_jpg
    """

    stem = Path(filename).stem

    if ".rf." in stem:
        return stem.split(".rf.")[0]

    return stem


def parse_label(label_path):

    text = label_path.read_text(
        encoding="utf-8"
    ).strip()

    if not text:
        return None

    classes = set()

    for line in text.splitlines():

        parts = line.split()

        if not parts:
            continue

        class_id = int(
            float(parts[0])
        )

        classes.add(class_id)

    if len(classes) == 0:
        return None

    if len(classes) > 1:

        raise ValueError(
            f"Multiple classes found in:\n"
            f"{label_path}\n"
            f"{classes}"
        )

    return next(iter(classes))


# ============================================================
# COLLECT ALL IMAGES
# ============================================================

groups = defaultdict(list)

excluded_empty = []

records = []


for original_split in [
    "train",
    "valid",
    "test"
]:

    images_dir = (
        DATASET_DIR /
        original_split /
        "images"
    )

    labels_dir = (
        DATASET_DIR /
        original_split /
        "labels"
    )

    for image_path in images_dir.iterdir():

        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        label_path = (
            labels_dir /
            f"{image_path.stem}.txt"
        )

        if not label_path.exists():

            print(
                f"WARNING: missing label for "
                f"{image_path.name}"
            )

            continue

        class_id = parse_label(
            label_path
        )

        if class_id is None:

            excluded_empty.append(
                str(image_path)
            )

            continue

        source_name = get_source_name(
            image_path.name
        )

        record = {
            "source_name": source_name,
            "image_path": image_path,
            "label_path": label_path,
            "class_id": class_id,
            "class_name": CLASS_MAP[class_id],
            "original_split": original_split
        }

        groups[
            source_name
        ].append(record)

        records.append(record)


# ============================================================
# VERIFY GROUP LABEL CONSISTENCY
# ============================================================

inconsistent_groups = []

group_info = []


for source_name, items in groups.items():

    classes = {
        item["class_id"]
        for item in items
    }

    if len(classes) != 1:

        inconsistent_groups.append(
            (
                source_name,
                classes
            )
        )

        continue

    class_id = next(iter(classes))

    group_info.append(
        {
            "source_name": source_name,
            "class_id": class_id,
            "class_name": CLASS_MAP[class_id],
            "items": items,
            "size": len(items)
        }
    )


if inconsistent_groups:

    print()
    print("=" * 70)
    print("ERROR: INCONSISTENT SOURCE GROUP LABELS")
    print("=" * 70)

    for source_name, classes in inconsistent_groups[:20]:

        print(
            source_name,
            classes
        )

    raise RuntimeError(
        "Some source groups contain different class labels."
    )


# ============================================================
# STRATIFY GROUPS BY CLASS
# ============================================================

groups_by_class = defaultdict(list)

for group in group_info:

    groups_by_class[
        group["class_id"]
    ].append(group)


# ============================================================
# SPLIT FUNCTION
# ============================================================

def split_groups(group_list):

    random.shuffle(
        group_list
    )

    n = len(group_list)

    train_end = int(
        n * TRAIN_RATIO
    )

    val_end = train_end + int(
        n * VAL_RATIO
    )

    train = group_list[
        :train_end
    ]

    val = group_list[
        train_end:val_end
    ]

    test = group_list[
        val_end:
    ]

    return train, val, test


# ============================================================
# BUILD SPLITS
# ============================================================

final_splits = {
    "train": [],
    "val": [],
    "test": []
}


for class_id, group_list in groups_by_class.items():

    train_groups, val_groups, test_groups = (
        split_groups(
            group_list.copy()
        )
    )

    final_splits[
        "train"
    ].extend(
        train_groups
    )

    final_splits[
        "val"
    ].extend(
        val_groups
    )

    final_splits[
        "test"
    ].extend(
        test_groups
    )


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

if OUTPUT_DIR.exists():

    print(
        f"\nRemoving old output directory:\n"
        f"{OUTPUT_DIR}"
    )

    shutil.rmtree(
        OUTPUT_DIR
    )


for split in [
    "train",
    "val",
    "test"
]:

    for class_folder in [
        "normal",
        "stone"
    ]:

        (
            OUTPUT_DIR /
            split /
            class_folder
        ).mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# COPY IMAGES
# ============================================================

manifest_rows = []

split_stats = {}


for split_name, split_groups_list in (
    final_splits.items()
):

    image_count = 0

    class_counts = Counter()

    source_count = len(
        split_groups_list
    )

    for group in split_groups_list:

        source_name = group[
            "source_name"
        ]

        class_id = group[
            "class_id"
        ]

        # Your classification convention:
        # 0 = stone
        # 1 = normal

        if class_id == 0:
            class_folder = "stone"

        elif class_id == 1:
            class_folder = "normal"

        else:
            raise ValueError(
                f"Unexpected class: {class_id}"
            )

        for item in group[
            "items"
        ]:

            src = item[
                "image_path"
            ]

            dst = (
                OUTPUT_DIR /
                split_name /
                class_folder /
                src.name
            )

            shutil.copy2(
                src,
                dst
            )

            image_count += 1

            class_counts[
                class_folder
            ] += 1

            manifest_rows.append(
                [
                    source_name,
                    split_name,
                    class_folder,
                    class_id,
                    src.name,
                    item[
                        "original_split"
                    ],
                    str(src),
                    str(dst)
                ]
            )

    split_stats[
        split_name
    ] = {
        "groups": source_count,
        "images": image_count,
        "normal": class_counts[
            "normal"
        ],
        "stone": class_counts[
            "stone"
        ]
    }


# ============================================================
# SAVE MANIFEST
# ============================================================

manifest_path = (
    OUTPUT_DIR /
    "group_split_manifest.csv"
)

with open(
    manifest_path,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow(
        [
            "source_group",
            "new_split",
            "class_name",
            "class_id",
            "filename",
            "original_roboflow_split",
            "source_path",
            "new_path"
        ]
    )

    writer.writerows(
        manifest_rows
    )


# ============================================================
# VERIFY NO GROUP LEAKAGE
# ============================================================

group_split_membership = defaultdict(
    set
)

for row in manifest_rows:

    source_group = row[0]
    new_split = row[1]

    group_split_membership[
        source_group
    ].add(
        new_split
    )


leaking_groups = {

    group:
        splits

    for group, splits in
    group_split_membership.items()

    if len(splits) > 1
}


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 75)
print("ECCKD GROUP-AWARE CLASSIFICATION SPLIT")
print("=" * 75)

print(
    f"\nOriginal usable images: "
    f"{len(records)}"
)

print(
    f"Excluded empty-label images: "
    f"{len(excluded_empty)}"
)

print(
    f"Unique source groups: "
    f"{len(group_info)}"
)

print()
print("-" * 75)
print("NEW SPLIT SUMMARY")
print("-" * 75)

for split in [
    "train",
    "val",
    "test"
]:

    stats = split_stats[
        split
    ]

    print(
        f"\n{split.upper()}"
    )

    print(
        f"  Source groups : "
        f"{stats['groups']}"
    )

    print(
        f"  Total images  : "
        f"{stats['images']}"
    )

    print(
        f"  Stone images  : "
        f"{stats['stone']}"
    )

    print(
        f"  Normal images : "
        f"{stats['normal']}"
    )


print()
print("-" * 75)
print("LEAKAGE VERIFICATION")
print("-" * 75)

print(
    f"\nGroups appearing in multiple "
    f"new splits: "
    f"{len(leaking_groups)}"
)


if len(leaking_groups) == 0:

    print(
        "PASS: No source group crosses "
        "train / val / test."
    )

else:

    print(
        "FAIL: Group leakage still exists."
    )


print(
    f"\nDataset created at:\n"
    f"{OUTPUT_DIR}"
)

print(
    f"\nManifest saved at:\n"
    f"{manifest_path}"
)

print()
print("=" * 75)
print("GROUP-AWARE SPLIT COMPLETE")
print("=" * 75)