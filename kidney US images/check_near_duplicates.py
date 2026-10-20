from pathlib import Path
from PIL import Image
from collections import defaultdict, Counter
import csv

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = Path(
    r"D:\Kidney\kidney US images\my dataset deduplicated"
)

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Conservative threshold:
# 0 = identical perceptually
# 1-3 = extremely similar
HAMMING_THRESHOLD = 3

# ============================================================
# DIFFERENCE HASH (dHash)
# ============================================================

def dhash(image_path, hash_size=8):

    with Image.open(image_path) as img:

        img = img.convert("L")

        # dHash compares adjacent pixels
        img = img.resize(
            (hash_size + 1, hash_size),
            Image.Resampling.LANCZOS
        )

        pixels = list(img.getdata())

    hash_value = 0

    for row in range(hash_size):

        for col in range(hash_size):

            left = pixels[row * (hash_size + 1) + col]
            right = pixels[row * (hash_size + 1) + col + 1]

            hash_value <<= 1

            if left > right:
                hash_value |= 1

    return hash_value


def hamming_distance(hash1, hash2):
    return (hash1 ^ hash2).bit_count()


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
print("NEAR-DUPLICATE ANALYSIS")
print("=" * 70)

print(f"\nImages being checked: {len(images)}")
print(f"Hamming-distance threshold: <= {HAMMING_THRESHOLD}")

# ============================================================
# CALCULATE HASHES
# ============================================================

records = []

for i, (file, class_name) in enumerate(images, start=1):

    try:

        image_hash = dhash(file)

        records.append(
            {
                "path": file,
                "class": class_name,
                "hash": image_hash
            }
        )

    except Exception as e:

        print(f"Could not hash: {file}")
        print(e)

    if i % 1000 == 0:
        print(f"Hashed {i}/{len(images)} images...")

print(f"\nSuccessfully hashed: {len(records)}")

# ============================================================
# MULTI-INDEX HASHING
#
# A pair with <=3 differing bits must share at least one
# of four 16-bit blocks exactly.
#
# This avoids comparing all ~35 million possible pairs.
# ============================================================

buckets = defaultdict(list)

MASK = (1 << 16) - 1

for idx, record in enumerate(records):

    h = record["hash"]

    blocks = [
        (h >> 0) & MASK,
        (h >> 16) & MASK,
        (h >> 32) & MASK,
        (h >> 48) & MASK
    ]

    for block_position, block_value in enumerate(blocks):

        key = (block_position, block_value)

        buckets[key].append(idx)

# ============================================================
# GENERATE CANDIDATE PAIRS
# ============================================================

candidate_pairs = set()

for indices in buckets.values():

    if len(indices) < 2:
        continue

    for i in range(len(indices)):

        for j in range(i + 1, len(indices)):

            a = indices[i]
            b = indices[j]

            if a > b:
                a, b = b, a

            candidate_pairs.add((a, b))

print(f"Candidate pairs to inspect: {len(candidate_pairs)}")

# ============================================================
# FIND NEAR DUPLICATES
# ============================================================

near_duplicates = []

distance_counter = Counter()

same_class_pairs = 0
cross_class_pairs = 0

for a, b in candidate_pairs:

    record_a = records[a]
    record_b = records[b]

    distance = hamming_distance(
        record_a["hash"],
        record_b["hash"]
    )

    if distance <= HAMMING_THRESHOLD:

        same_class = (
            record_a["class"] == record_b["class"]
        )

        if same_class:
            same_class_pairs += 1
        else:
            cross_class_pairs += 1

        distance_counter[distance] += 1

        near_duplicates.append(
            {
                "file_1": str(record_a["path"]),
                "class_1": record_a["class"],
                "file_2": str(record_b["path"]),
                "class_2": record_b["class"],
                "distance": distance
            }
        )

# ============================================================
# SAVE RESULTS
# ============================================================

output_csv = DATASET_DIR / "near_duplicate_pairs.csv"

with open(
    output_csv,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "file_1",
            "class_1",
            "file_2",
            "class_2",
            "distance"
        ]
    )

    writer.writeheader()
    writer.writerows(near_duplicates)

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("NEAR-DUPLICATE RESULTS")
print("=" * 70)

print(f"\nNear-duplicate pairs found : {len(near_duplicates)}")
print(f"Same-class pairs          : {same_class_pairs}")
print(f"Cross-class pairs         : {cross_class_pairs}")

print("\nDistance distribution:")

for distance in sorted(distance_counter):
    print(
        f"Distance {distance}: "
        f"{distance_counter[distance]} pairs"
    )

# ============================================================
# SHOW SAMPLE PAIRS
# ============================================================

print("\n" + "=" * 70)
print("EXAMPLE NEAR-DUPLICATE PAIRS")
print("=" * 70)

for i, pair in enumerate(
    sorted(
        near_duplicates,
        key=lambda x: x["distance"]
    )[:20],
    start=1
):

    print(f"\nPair {i}")

    print(
        f"[{pair['class_1']}] "
        f"{pair['file_1']}"
    )

    print(
        f"[{pair['class_2']}] "
        f"{pair['file_2']}"
    )

    print(
        f"dHash distance = "
        f"{pair['distance']}"
    )

print("\nResults saved to:")
print(output_csv)

print("\nNear-duplicate analysis complete.")