from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import csv
import random

# ============================================================
# CONFIG
# ============================================================

DATASET_DIR = Path(
    r"D:\Kidney\kidney US images\my dataset deduplicated"
)

CSV_FILE = DATASET_DIR / "near_duplicate_pairs.csv"

OUTPUT_DIR = Path(
    r"D:\Kidney\kidney US images\near_duplicate_review"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)

# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 350
HEADER_HEIGHT = 70
GAP = 20

# ============================================================
# LOAD CSV
# ============================================================

pairs = []

with open(CSV_FILE, "r", encoding="utf-8") as f:

    reader = csv.DictReader(f)

    for row in reader:

        row["distance"] = int(row["distance"])

        pairs.append(row)

print("=" * 70)
print("NEAR-DUPLICATE VISUAL REVIEW")
print("=" * 70)

print(f"\nTotal candidate pairs loaded: {len(pairs)}")

# ============================================================
# SEPARATE PAIRS
# ============================================================

cross_class = [
    pair
    for pair in pairs
    if pair["class_1"] != pair["class_2"]
]

same_class = [
    pair
    for pair in pairs
    if pair["class_1"] == pair["class_2"]
]

print(f"Cross-class pairs: {len(cross_class)}")
print(f"Same-class pairs : {len(same_class)}")

# ============================================================
# SELECT PAIRS FOR MANUAL REVIEW
# ============================================================

selected = []

# All cross-class pairs
for pair in cross_class:

    pair["review_type"] = "CROSS_CLASS"
    selected.append(pair)

# Sample same-class pairs by distance
sample_sizes = {
    0: 20,
    1: 10,
    2: 10,
    3: 10
}

for distance, sample_size in sample_sizes.items():

    candidates = [
        pair
        for pair in same_class
        if pair["distance"] == distance
    ]

    if len(candidates) > sample_size:
        candidates = random.sample(
            candidates,
            sample_size
        )

    for pair in candidates:

        pair_copy = pair.copy()
        pair_copy["review_type"] = f"SAME_CLASS_D{distance}"

        selected.append(pair_copy)

print(f"\nPairs selected for visual review: {len(selected)}")

# ============================================================
# CREATE SIDE-BY-SIDE IMAGE
# ============================================================

def create_comparison(pair, output_path):

    image1 = Image.open(pair["file_1"]).convert("RGB")
    image2 = Image.open(pair["file_2"]).convert("RGB")

    # Resize while preserving full image
    image1 = image1.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.LANCZOS
    )

    image2 = image2.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.LANCZOS
    )

    canvas_width = (
        IMAGE_SIZE * 2
        + GAP
    )

    canvas_height = (
        IMAGE_SIZE
        + HEADER_HEIGHT
    )

    canvas = Image.new(
        "RGB",
        (canvas_width, canvas_height),
        "white"
    )

    canvas.paste(
        image1,
        (0, HEADER_HEIGHT)
    )

    canvas.paste(
        image2,
        (IMAGE_SIZE + GAP, HEADER_HEIGHT)
    )

    draw = ImageDraw.Draw(canvas)

    # Header text
    title = (
        f"{pair['review_type']} | "
        f"dHash distance = {pair['distance']}"
    )

    draw.text(
        (10, 5),
        title,
        fill="black"
    )

    label1 = (
        f"{pair['class_1']} | "
        f"{Path(pair['file_1']).name}"
    )

    label2 = (
        f"{pair['class_2']} | "
        f"{Path(pair['file_2']).name}"
    )

    draw.text(
        (10, 30),
        label1,
        fill="black"
    )

    draw.text(
        (IMAGE_SIZE + GAP, 30),
        label2,
        fill="black"
    )

    canvas.save(
        output_path,
        quality=95
    )

# ============================================================
# GENERATE REVIEW IMAGES
# ============================================================

for i, pair in enumerate(selected, start=1):

    review_type = pair["review_type"]

    folder = OUTPUT_DIR / review_type

    folder.mkdir(
        parents=True,
        exist_ok=True
    )

    filename = (
        f"pair_{i:03d}_"
        f"distance_{pair['distance']}.jpg"
    )

    output_path = folder / filename

    try:

        create_comparison(
            pair,
            output_path
        )

    except Exception as e:

        print("\nERROR creating comparison:")
        print(pair["file_1"])
        print(pair["file_2"])
        print(e)

# ============================================================
# CREATE REVIEW TEXT FILE
# ============================================================

review_file = OUTPUT_DIR / "review_pairs.txt"

with open(
    review_file,
    "w",
    encoding="utf-8"
) as f:

    f.write("NEAR-DUPLICATE MANUAL REVIEW\n")
    f.write("=" * 70 + "\n\n")

    for i, pair in enumerate(selected, start=1):

        f.write(f"PAIR {i}\n")
        f.write(
            f"Type     : {pair['review_type']}\n"
        )

        f.write(
            f"Distance : {pair['distance']}\n"
        )

        f.write(
            f"Image 1  : "
            f"[{pair['class_1']}] "
            f"{pair['file_1']}\n"
        )

        f.write(
            f"Image 2  : "
            f"[{pair['class_2']}] "
            f"{pair['file_2']}\n"
        )

        f.write("-" * 70 + "\n")

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("REVIEW DATA CREATED")
print("=" * 70)

print(f"\nCross-class comparisons : {len(cross_class)}")

for distance in range(4):

    count = sum(
        1
        for pair in selected
        if pair["review_type"] == f"SAME_CLASS_D{distance}"
    )

    print(
        f"Same-class distance {distance}: {count}"
    )

print("\nReview folder:")
print(OUTPUT_DIR)

print("\nIMPORTANT:")
print("Open the CROSS_CLASS folder first.")
print("Then inspect SAME_CLASS_D0.")

print("\nVisual review generation complete.")