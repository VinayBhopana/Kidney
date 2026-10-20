from pathlib import Path
from PIL import Image
import csv
import numpy as np
from collections import Counter, defaultdict

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = Path(
    r"D:\Kidney\kidney US images\my dataset deduplicated"
)

CSV_FILE = DATASET_DIR / "near_duplicate_pairs.csv"

OUTPUT_CSV = DATASET_DIR / "strict_near_duplicate_pairs.csv"
GROUP_CSV = DATASET_DIR / "near_duplicate_groups.csv"

# Small representation makes brightness/annotation differences less dominant
IMAGE_SIZE = 128

# Strict thresholds
CORRELATION_THRESHOLD = 0.995
MAE_THRESHOLD = 6.0

# ============================================================
# LOAD dHASH CANDIDATES
# ============================================================

pairs = []

with open(CSV_FILE, "r", encoding="utf-8") as f:

    reader = csv.DictReader(f)

    for row in reader:

        # IMPORTANT:
        # We deliberately analyse SAME-CLASS pairs only.
        # Cross-class visual inspection showed dHash false positives.
        if row["class_1"] == row["class_2"]:

            pairs.append(row)

print("=" * 75)
print("STRICT NEAR-DUPLICATE ANALYSIS")
print("=" * 75)

print(f"\nSame-class candidate pairs: {len(pairs)}")

print(
    f"Thresholds:"
    f"\n  Correlation >= {CORRELATION_THRESHOLD}"
    f"\n  MAE <= {MAE_THRESHOLD}"
)

# ============================================================
# IMAGE CACHE
# ============================================================

image_cache = {}


def load_image(path):

    if path in image_cache:
        return image_cache[path]

    with Image.open(path) as img:

        img = img.convert("L")

        img = img.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.LANCZOS
        )

        array = np.asarray(
            img,
            dtype=np.float32
        )

    image_cache[path] = array

    return array


# ============================================================
# SIMILARITY FUNCTIONS
# ============================================================

def calculate_mae(a, b):

    return float(
        np.mean(
            np.abs(a - b)
        )
    )


def calculate_correlation(a, b):

    a_flat = a.flatten()
    b_flat = b.flatten()

    a_centered = a_flat - np.mean(a_flat)
    b_centered = b_flat - np.mean(b_flat)

    denominator = (
        np.linalg.norm(a_centered)
        * np.linalg.norm(b_centered)
    )

    if denominator == 0:
        return 0.0

    correlation = np.dot(
        a_centered,
        b_centered
    ) / denominator

    return float(correlation)


# ============================================================
# STRICT COMPARISON
# ============================================================

strict_pairs = []

for i, pair in enumerate(pairs, start=1):

    file1 = pair["file_1"]
    file2 = pair["file_2"]

    try:

        img1 = load_image(file1)
        img2 = load_image(file2)

        mae = calculate_mae(
            img1,
            img2
        )

        correlation = calculate_correlation(
            img1,
            img2
        )

        if (
            correlation >= CORRELATION_THRESHOLD
            and mae <= MAE_THRESHOLD
        ):

            strict_pairs.append(
                {
                    "file_1": file1,
                    "file_2": file2,
                    "class": pair["class_1"],
                    "dhash_distance": pair["distance"],
                    "correlation": correlation,
                    "mae": mae
                }
            )

    except Exception as e:

        print("\nCould not process:")
        print(file1)
        print(file2)
        print(e)

    if i % 2000 == 0:

        print(
            f"Processed {i}/{len(pairs)} pairs..."
        )

# ============================================================
# SAVE STRICT PAIRS
# ============================================================

with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    fieldnames = [
        "file_1",
        "file_2",
        "class",
        "dhash_distance",
        "correlation",
        "mae"
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(strict_pairs)

# ============================================================
# UNION-FIND
#
# Convert pairs into groups:
#
# A ~ B
# B ~ C
#
# becomes:
#
# A, B, C = one near-duplicate group
# ============================================================

parent = {}


def find(x):

    if x not in parent:
        parent[x] = x

    if parent[x] != x:
        parent[x] = find(parent[x])

    return parent[x]


def union(a, b):

    root_a = find(a)
    root_b = find(b)

    if root_a != root_b:
        parent[root_b] = root_a


for pair in strict_pairs:

    union(
        pair["file_1"],
        pair["file_2"]
    )

# ============================================================
# BUILD GROUPS
# ============================================================

groups = defaultdict(list)

for file in parent:

    root = find(file)

    groups[root].append(file)

near_duplicate_groups = [
    files
    for files in groups.values()
    if len(files) > 1
]

near_duplicate_groups.sort(
    key=len,
    reverse=True
)

# ============================================================
# CLASS LOOKUP
# ============================================================

class_lookup = {}

for pair in strict_pairs:

    class_lookup[pair["file_1"]] = pair["class"]
    class_lookup[pair["file_2"]] = pair["class"]

# ============================================================
# SAVE GROUP FILE
# ============================================================

with open(
    GROUP_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    fieldnames = [
        "group_id",
        "class",
        "image_path"
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()

    for group_id, files in enumerate(
        near_duplicate_groups,
        start=1
    ):

        for file in files:

            writer.writerow(
                {
                    "group_id": group_id,
                    "class": class_lookup[file],
                    "image_path": file
                }
            )

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("STRICT SIMILARITY RESULTS")
print("=" * 75)

print(
    f"\nStrict near-duplicate pairs: "
    f"{len(strict_pairs)}"
)

# ============================================================
# PAIRS PER CLASS
# ============================================================

pair_class_counts = Counter(
    pair["class"]
    for pair in strict_pairs
)

print("\nStrict pairs by class:")

for class_name, count in pair_class_counts.items():

    print(
        f"  {class_name}: {count}"
    )

# ============================================================
# GROUP STATISTICS
# ============================================================

print(
    f"\nNear-duplicate groups: "
    f"{len(near_duplicate_groups)}"
)

images_in_groups = sum(
    len(group)
    for group in near_duplicate_groups
)

print(
    f"Images belonging to near-duplicate groups: "
    f"{images_in_groups}"
)

group_size_distribution = Counter(
    len(group)
    for group in near_duplicate_groups
)

print("\nGroup-size distribution:")

for size, count in sorted(
    group_size_distribution.items()
):

    print(
        f"  {size} images -> {count} groups"
    )

# ============================================================
# CLASS GROUP STATISTICS
# ============================================================

images_by_class = Counter()

for group in near_duplicate_groups:

    for file in group:

        images_by_class[
            class_lookup[file]
        ] += 1

print("\nImages in near-duplicate groups by class:")

for class_name, count in images_by_class.items():

    print(
        f"  {class_name}: {count}"
    )

# ============================================================
# SAMPLE HIGH-SIMILARITY PAIRS
# ============================================================

print("\n" + "=" * 75)
print("TOP 20 MOST SIMILAR PAIRS")
print("=" * 75)

sorted_pairs = sorted(
    strict_pairs,
    key=lambda x: (
        -x["correlation"],
        x["mae"]
    )
)

for i, pair in enumerate(
    sorted_pairs[:20],
    start=1
):

    print(f"\nPair {i}")

    print(
        f"[{pair['class']}] "
        f"{Path(pair['file_1']).name}"
    )

    print(
        f"[{pair['class']}] "
        f"{Path(pair['file_2']).name}"
    )

    print(
        f"Correlation: "
        f"{pair['correlation']:.6f}"
    )

    print(
        f"MAE: "
        f"{pair['mae']:.3f}"
    )

    print(
        f"dHash distance: "
        f"{pair['dhash_distance']}"
    )

# ============================================================
# OUTPUT PATHS
# ============================================================

print("\n" + "=" * 75)

print("\nStrict pair list saved to:")
print(OUTPUT_CSV)

print("\nNear-duplicate groups saved to:")
print(GROUP_CSV)

print("\nAnalysis complete.")