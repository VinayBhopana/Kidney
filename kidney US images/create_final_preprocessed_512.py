from pathlib import Path
from PIL import Image
from collections import Counter
import csv
import hashlib

# ============================================================
# CONFIGURATION
# ============================================================

SOURCE_DIR = Path(
    r"D:\Kidney\kidney US images\dataset_group_split"
)

OUTPUT_DIR = Path(
    r"D:\Kidney\kidney US images\dataset_preprocessed_grayscale_512"
)

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

EXPECTED_SIZE = (512, 512)

# ============================================================
# SAFETY CHECK
# ============================================================

if OUTPUT_DIR.exists():

    print("=" * 70)
    print("ERROR")
    print("=" * 70)

    print("\nOutput folder already exists:")
    print(OUTPUT_DIR)

    print(
        "\nDelete or rename it before running "
        "this script again."
    )

    raise SystemExit


# ============================================================
# CREATE OUTPUT FOLDERS
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
# HASH FUNCTION
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
# START
# ============================================================

print("=" * 70)
print("FINAL KIDNEY ULTRASOUND PREPROCESSING - 512 x 512")
print("=" * 70)

print("\nPipeline:")
print("1. Exact duplicates already removed")
print("2. Near-duplicate-safe split already completed")
print("3. Convert all images to grayscale")
print("4. KEEP original 512 x 512 resolution")
print("5. Save as lossless PNG")
print("6. No augmentation")
print("7. No ImageNet normalization")

manifest = []

processed_counts = Counter()

failed_files = []

total_processed = 0


# ============================================================
# PROCESS DATASET
# ============================================================

for split in SPLITS:

    print("\n" + "=" * 70)
    print(f"PROCESSING {split.upper()}")
    print("=" * 70)

    for class_name in CLASSES:

        source_folder = (
            SOURCE_DIR
            / split
            / class_name
        )

        destination_folder = (
            OUTPUT_DIR
            / split
            / class_name
        )

        if not source_folder.exists():

            print("\nERROR: Missing source folder:")
            print(source_folder)

            raise SystemExit

        files = [
            file
            for file in source_folder.iterdir()
            if file.suffix.lower()
            in VALID_EXTENSIONS
        ]

        print(
            f"\n{class_name}: "
            f"{len(files)} images"
        )

        for file in files:

            try:

                with Image.open(file) as img:

                    img.load()

                    original_size = img.size
                    original_mode = img.mode

                    # ----------------------------------------
                    # VERIFY ORIGINAL RESOLUTION
                    # ----------------------------------------

                    if original_size != EXPECTED_SIZE:

                        print(
                            f"\nWARNING: "
                            f"{file.name} has size "
                            f"{original_size}"
                        )

                    # ----------------------------------------
                    # CONVERT TO GRAYSCALE
                    # ----------------------------------------

                    img = img.convert("L")

                    # ----------------------------------------
                    # IMPORTANT:
                    # NO RESIZING IS DONE HERE
                    # ----------------------------------------

                    if img.size != EXPECTED_SIZE:

                        raise ValueError(
                            f"Unexpected image size "
                            f"{img.size}; expected "
                            f"{EXPECTED_SIZE}"
                        )

                    # ----------------------------------------
                    # SAVE AS LOSSLESS PNG
                    # ----------------------------------------

                    output_file = (
                        destination_folder
                        / f"{file.stem}.png"
                    )

                    # Protect against filename collisions
                    counter = 1

                    while output_file.exists():

                        output_file = (
                            destination_folder
                            / (
                                f"{file.stem}_"
                                f"{counter}.png"
                            )
                        )

                        counter += 1

                    img.save(
                        output_file,
                        format="PNG"
                    )

                output_hash = calculate_md5(
                    output_file
                )

                manifest.append(
                    {
                        "split": split,
                        "class": class_name,
                        "original_file": str(file),
                        "processed_file": str(output_file),
                        "original_width": original_size[0],
                        "original_height": original_size[1],
                        "original_mode": original_mode,
                        "processed_width": 512,
                        "processed_height": 512,
                        "processed_mode": "L",
                        "md5": output_hash
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
                        f"{total_processed} images..."
                    )

            except Exception as e:

                failed_files.append(
                    (
                        str(file),
                        str(e)
                    )
                )

                print("\nERROR processing:")
                print(file)
                print(e)


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
    writer.writerows(manifest)


# ============================================================
# VERIFY FINAL DATASET
# ============================================================

print("\n" + "=" * 70)
print("VERIFYING FINAL DATASET")
print("=" * 70)

verification_counts = Counter()

dimensions = Counter()

modes = Counter()

corrupted = []

for split in SPLITS:

    for class_name in CLASSES:

        folder = (
            OUTPUT_DIR
            / split
            / class_name
        )

        for file in folder.glob("*.png"):

            try:

                with Image.open(file) as img:

                    img.load()

                    verification_counts[
                        (
                            split,
                            class_name
                        )
                    ] += 1

                    dimensions[
                        img.size
                    ] += 1

                    modes[
                        img.mode
                    ] += 1

            except Exception as e:

                corrupted.append(
                    (
                        str(file),
                        str(e)
                    )
                )


# ============================================================
# PRINT SPLIT COUNTS
# ============================================================

grand_total = 0

for split in SPLITS:

    print(f"\n{split.upper()}")

    split_total = 0

    for class_name in CLASSES:

        count = verification_counts[
            (
                split,
                class_name
            )
        ]

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
# IMAGE PROPERTIES
# ============================================================

print("\nIMAGE DIMENSIONS")

for size, count in dimensions.items():

    print(
        f"  {size}: {count}"
    )


print("\nIMAGE MODES")

for mode, count in modes.items():

    print(
        f"  {mode}: {count}"
    )


print(
    f"\nCorrupted / invalid files: "
    f"{len(corrupted)}"
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
        "FINAL KIDNEY ULTRASOUND DATASET\n"
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
        "Strict near-duplicate grouping performed.\n"
    )

    f.write(
        "Near-duplicate groups crossing "
        "train/val/test: 0\n\n"
    )

    f.write(
        "Final preprocessing:\n"
    )

    f.write(
        "- Grayscale conversion\n"
    )

    f.write(
        "- Resolution preserved at 512 x 512\n"
    )

    f.write(
        "- Saved as lossless PNG\n"
    )

    f.write(
        "- No permanent augmentation\n"
    )

    f.write(
        "- No CLAHE permanently applied\n"
    )

    f.write(
        "- No ImageNet normalization "
        "permanently applied\n\n"
    )

    for split in SPLITS:

        f.write(
            f"{split.upper()}\n"
        )

        split_total = 0

        for class_name in CLASSES:

            count = verification_counts[
                (
                    split,
                    class_name
                )
            ]

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


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 70)
print("FINAL PREPROCESSING SUMMARY")
print("=" * 70)

print(
    f"\nTotal images: "
    f"{grand_total}"
)

print(
    f"Processing failures: "
    f"{len(failed_files)}"
)

print(
    f"Verification failures: "
    f"{len(corrupted)}"
)

print("\nExpected split:")

print("  Train : 5861")
print("  Val   : 1258")
print("  Test  : 1257")
print("  Total : 8376")

print("\nExpected image properties:")

print("  Resolution : 512 x 512")
print("  Mode       : L (grayscale)")


# ============================================================
# STRICT PASS CHECK
# ============================================================

correct_counts = (
    verification_counts[
        ("train", "Normal")
    ] == 3030
    and verification_counts[
        ("train", "stone")
    ] == 2831
    and verification_counts[
        ("val", "Normal")
    ] == 650
    and verification_counts[
        ("val", "stone")
    ] == 608
    and verification_counts[
        ("test", "Normal")
    ] == 650
    and verification_counts[
        ("test", "stone")
    ] == 607
)

correct_dimensions = (
    dimensions == Counter(
        {
            (512, 512): 8376
        }
    )
)

correct_modes = (
    modes == Counter(
        {
            "L": 8376
        }
    )
)

if (
    grand_total == 8376
    and correct_counts
    and correct_dimensions
    and correct_modes
    and len(failed_files) == 0
    and len(corrupted) == 0
):

    print(
        "\nPASS: FINAL 512 x 512 "
        "PREPROCESSED DATASET VERIFIED."
    )

else:

    print(
        "\nWARNING: Verification "
        "did not match expected values."
    )


print("\nFinal dataset folder:")

print(OUTPUT_DIR)

print("\nManifest:")

print(MANIFEST_FILE)

print("\nSummary:")

print(SUMMARY_FILE)

print("\nDone.")