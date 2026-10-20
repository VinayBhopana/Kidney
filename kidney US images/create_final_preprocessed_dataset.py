from pathlib import Path
from PIL import Image
import csv
import hashlib
from collections import Counter

# ============================================================
# CONFIGURATION
# ============================================================

SOURCE_DIR = Path(
    r"D:\Kidney\kidney US images\dataset_group_split"
)

OUTPUT_DIR = Path(
    r"D:\Kidney\kidney US images\dataset_preprocessed_grayscale_224"
)

IMAGE_SIZE = (224, 224)

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

SPLITS = [
    "train",
    "val",
    "test"
]

CLASSES = [
    "Normal",
    "stone"
]

# ============================================================
# SAFETY CHECK
# ============================================================

if OUTPUT_DIR.exists():

    print("=" * 70)
    print("ERROR")
    print("=" * 70)

    print(
        "\nOutput directory already exists:"
    )

    print(
        OUTPUT_DIR
    )

    print(
        "\nDelete or rename it before running "
        "the script again."
    )

    raise SystemExit

# ============================================================
# CREATE OUTPUT STRUCTURE
# ============================================================

for split in SPLITS:

    for class_name in CLASSES:

        (
            OUTPUT_DIR
            / split
            / class_name
        ).mkdir(
            parents=True,
            exist_ok=True
        )

# ============================================================
# UTILITY FUNCTION
# ============================================================

def calculate_md5(file_path):

    hasher = hashlib.md5()

    with open(file_path, "rb") as f:

        while True:

            chunk = f.read(8192)

            if not chunk:
                break

            hasher.update(chunk)

    return hasher.hexdigest()


# ============================================================
# PREPROCESSING
# ============================================================

print("=" * 70)
print("FINAL KIDNEY ULTRASOUND PREPROCESSING")
print("=" * 70)

print(
    "\nPipeline:"
)

print(
    "1. Leakage-safe group split already completed"
)

print(
    "2. Convert image to grayscale"
)

print(
    "3. Resize image to 224 x 224"
)

print(
    "4. Save as lossless PNG"
)

print(
    "\nNo augmentation applied."
)

print(
    "No ImageNet normalization applied to stored images."
)

print()

manifest_records = []

processed_counts = Counter()

corrupted_files = []

total_processed = 0

# ============================================================
# PROCESS EACH SPLIT
# ============================================================

for split in SPLITS:

    print(
        "\n" + "=" * 70
    )

    print(
        f"PROCESSING {split.upper()}"
    )

    print(
        "=" * 70
    )

    for class_name in CLASSES:

        source_class_dir = (
            SOURCE_DIR
            / split
            / class_name
        )

        output_class_dir = (
            OUTPUT_DIR
            / split
            / class_name
        )

        if not source_class_dir.exists():

            print(
                f"\nERROR: Missing folder:"
            )

            print(
                source_class_dir
            )

            raise SystemExit

        files = [
            file
            for file in source_class_dir.iterdir()
            if file.suffix.lower()
            in VALID_EXTENSIONS
        ]

        print(
            f"\n{class_name}: "
            f"{len(files)} images"
        )

        for index, source_file in enumerate(
            files,
            start=1
        ):

            try:

                # --------------------------------------------
                # OPEN IMAGE
                # --------------------------------------------

                with Image.open(
                    source_file
                ) as img:

                    original_width = (
                        img.width
                    )

                    original_height = (
                        img.height
                    )

                    original_mode = (
                        img.mode
                    )

                    # Force image loading
                    img.load()

                    # ----------------------------------------
                    # GRAYSCALE
                    # ----------------------------------------

                    img = img.convert(
                        "L"
                    )

                    # ----------------------------------------
                    # RESIZE
                    # ----------------------------------------

                    img = img.resize(
                        IMAGE_SIZE,
                        Image.Resampling.LANCZOS
                    )

                    # ----------------------------------------
                    # OUTPUT NAME
                    # ----------------------------------------

                    output_filename = (
                        source_file.stem
                        + ".png"
                    )

                    output_file = (
                        output_class_dir
                        / output_filename
                    )

                    # Protect against filename collision
                    counter = 1

                    while output_file.exists():

                        output_filename = (
                            f"{source_file.stem}_"
                            f"{counter}.png"
                        )

                        output_file = (
                            output_class_dir
                            / output_filename
                        )

                        counter += 1

                    # ----------------------------------------
                    # SAVE LOSSLESS PNG
                    # ----------------------------------------

                    img.save(
                        output_file,
                        format="PNG",
                        optimize=False
                    )

                # --------------------------------------------
                # HASH OUTPUT FOR REPRODUCIBILITY
                # --------------------------------------------

                output_md5 = calculate_md5(
                    output_file
                )

                manifest_records.append(
                    {
                        "split": split,
                        "class": class_name,
                        "original_file":
                            str(source_file),
                        "processed_file":
                            str(output_file),
                        "original_width":
                            original_width,
                        "original_height":
                            original_height,
                        "original_mode":
                            original_mode,
                        "processed_width":
                            IMAGE_SIZE[0],
                        "processed_height":
                            IMAGE_SIZE[1],
                        "processed_mode":
                            "L",
                        "md5":
                            output_md5
                    }
                )

                processed_counts[
                    (
                        split,
                        class_name
                    )
                ] += 1

                total_processed += 1

                if total_processed % 1000 == 0:

                    print(
                        f"Processed "
                        f"{total_processed} "
                        f"images..."
                    )

            except Exception as e:

                corrupted_files.append(
                    (
                        str(source_file),
                        str(e)
                    )
                )

                print(
                    "\nERROR processing:"
                )

                print(
                    source_file
                )

                print(
                    e
                )

# ============================================================
# SAVE MANIFEST
# ============================================================

MANIFEST_FILE = (
    OUTPUT_DIR
    / "preprocessing_manifest.csv"
)

with open(
    MANIFEST_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    fieldnames = [
        "split",
        "class",
        "original_file",
        "processed_file",
        "original_width",
        "original_height",
        "original_mode",
        "processed_width",
        "processed_height",
        "processed_mode",
        "md5"
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(
        manifest_records
    )

# ============================================================
# VERIFY OUTPUT
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "VERIFYING FINAL DATASET"
)

print(
    "=" * 70
)

verification_counts = Counter()

dimensions = Counter()

modes = Counter()

verification_errors = []

for split in SPLITS:

    for class_name in CLASSES:

        folder = (
            OUTPUT_DIR
            / split
            / class_name
        )

        for file in folder.glob(
            "*.png"
        ):

            verification_counts[
                (
                    split,
                    class_name
                )
            ] += 1

            try:

                with Image.open(
                    file
                ) as img:

                    img.load()

                    dimensions[
                        img.size
                    ] += 1

                    modes[
                        img.mode
                    ] += 1

            except Exception as e:

                verification_errors.append(
                    (
                        str(file),
                        str(e)
                    )
                )

# ============================================================
# DISPLAY COUNTS
# ============================================================

grand_total = 0

for split in SPLITS:

    print(
        f"\n{split.upper()}"
    )

    split_total = 0

    for class_name in CLASSES:

        count = (
            verification_counts[
                (
                    split,
                    class_name
                )
            ]
        )

        print(
            f"  {class_name:<10}: "
            f"{count}"
        )

        split_total += count

    print(
        f"  TOTAL     : "
        f"{split_total}"
    )

    grand_total += split_total

# ============================================================
# IMAGE PROPERTY CHECK
# ============================================================

print(
    "\nIMAGE DIMENSIONS"
)

for dimension, count in (
    dimensions.most_common()
):

    print(
        f"  {dimension}: "
        f"{count}"
    )

print(
    "\nIMAGE MODES"
)

for mode, count in (
    modes.most_common()
):

    print(
        f"  {mode}: "
        f"{count}"
    )

print(
    "\nVerification errors:"
)

print(
    len(verification_errors)
)

# ============================================================
# SAVE SUMMARY
# ============================================================

SUMMARY_FILE = (
    OUTPUT_DIR
    / "preprocessing_summary.txt"
)

with open(
    SUMMARY_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "KIDNEY ULTRASOUND FINAL PREPROCESSED DATASET\n"
    )

    f.write(
        "=" * 60 + "\n\n"
    )

    f.write(
        "Original dataset: 9,416 images\n"
    )

    f.write(
        "Exact duplicates removed: 1,040\n"
    )

    f.write(
        "Unique images retained: 8,376\n\n"
    )

    f.write(
        "Near-duplicate-aware splitting was performed "
        "before preprocessing.\n"
    )

    f.write(
        "Near-duplicate groups crossing splits: 0\n\n"
    )

    f.write(
        "Preprocessing pipeline:\n"
    )

    f.write(
        "1. Convert to grayscale\n"
    )

    f.write(
        "2. Resize to 224 x 224 pixels using "
        "Lanczos interpolation\n"
    )

    f.write(
        "3. Save as lossless PNG\n"
    )

    f.write(
        "4. No data augmentation permanently applied\n"
    )

    f.write(
        "5. No ImageNet normalization permanently applied\n\n"
    )

    for split in SPLITS:

        f.write(
            f"{split.upper()}\n"
        )

        split_total = 0

        for class_name in CLASSES:

            count = (
                verification_counts[
                    (
                        split,
                        class_name
                    )
                ]
            )

            f.write(
                f"  {class_name}: "
                f"{count}\n"
            )

            split_total += count

        f.write(
            f"  Total: "
            f"{split_total}\n\n"
        )

    f.write(
        f"Final total: "
        f"{grand_total}\n"
    )

    f.write(
        f"Verification errors: "
        f"{len(verification_errors)}\n"
    )

# ============================================================
# FINAL SUMMARY
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "FINAL PREPROCESSING SUMMARY"
)

print(
    "=" * 70
)

print(
    f"\nTotal processed images: "
    f"{grand_total}"
)

print(
    f"Corrupted / failed: "
    f"{len(corrupted_files)}"
)

print(
    f"Verification errors: "
    f"{len(verification_errors)}"
)

print(
    "\nExpected:"
)

print(
    "  Train : 5861"
)

print(
    "  Val   : 1258"
)

print(
    "  Test  : 1257"
)

print(
    "  Total : 8376"
)

print(
    "\nFinal dataset:"
)

print(
    OUTPUT_DIR
)

print(
    "\nManifest:"
)

print(
    MANIFEST_FILE
)

print(
    "\nSummary:"
)

print(
    SUMMARY_FILE
)

if (
    grand_total == 8376
    and len(verification_errors) == 0
    and dimensions == Counter(
        {
            (224, 224): 8376
        }
    )
    and modes == Counter(
        {
            "L": 8376
        }
    )
):

    print(
        "\nPASS: Final preprocessed "
        "dataset verified successfully."
    )

else:

    print(
        "\nWARNING: Verification did "
        "not exactly match expected values."
    )

print(
    "\nPreprocessing complete."
)