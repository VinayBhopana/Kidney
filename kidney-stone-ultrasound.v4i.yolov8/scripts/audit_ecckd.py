from pathlib import Path
from collections import Counter, defaultdict
from PIL import Image
import yaml

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = Path(
    r"D:\Kidney\kidney-stone-ultrasound.v4i.yolov8"
)

YAML_PATH = DATASET_DIR / "data.yaml"

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
# LOAD CLASS CONFIGURATION
# ============================================================

with open(YAML_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

class_names = config["names"]

if isinstance(class_names, list):
    CLASS_MAP = {
        i: name for i, name in enumerate(class_names)
    }

elif isinstance(class_names, dict):
    CLASS_MAP = {
        int(k): v for k, v in class_names.items()
    }

else:
    raise ValueError("Unexpected format for class names")


# ============================================================
# STORAGE
# ============================================================

global_image_count = 0
global_label_count = 0

global_class_box_counts = Counter()
global_image_class_counts = Counter()

global_modes = Counter()
global_formats = Counter()
global_dimensions = Counter()

corrupted_images = []

missing_label_files = []

empty_label_files = []

invalid_label_lines = []

images_with_multiple_boxes = []

images_with_multiple_classes = []

split_summary = {}


# ============================================================
# FUNCTION: PARSE YOLO LABEL
# ============================================================

def parse_label_file(label_path):

    boxes = []

    if not label_path.exists():
        return None

    text = label_path.read_text(
        encoding="utf-8"
    ).strip()

    if not text:
        return []

    for line_number, line in enumerate(
        text.splitlines(),
        start=1
    ):

        parts = line.strip().split()

        if len(parts) != 5:

            invalid_label_lines.append(
                {
                    "file": str(label_path),
                    "line": line_number,
                    "content": line
                }
            )

            continue

        try:

            class_id = int(float(parts[0]))

            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

        except ValueError:

            invalid_label_lines.append(
                {
                    "file": str(label_path),
                    "line": line_number,
                    "content": line
                }
            )

            continue

        # Check class ID
        if class_id not in CLASS_MAP:

            invalid_label_lines.append(
                {
                    "file": str(label_path),
                    "line": line_number,
                    "content": line,
                    "reason": "Unknown class ID"
                }
            )

            continue

        # YOLO coordinates should normally be in [0, 1]
        coords = [
            x_center,
            y_center,
            width,
            height
        ]

        if any(v < 0 or v > 1 for v in coords):

            invalid_label_lines.append(
                {
                    "file": str(label_path),
                    "line": line_number,
                    "content": line,
                    "reason": "Coordinate outside [0,1]"
                }
            )

        boxes.append(
            {
                "class_id": class_id,
                "x_center": x_center,
                "y_center": y_center,
                "width": width,
                "height": height
            }
        )

    return boxes


# ============================================================
# AUDIT EACH SPLIT
# ============================================================

for split in ["train", "valid", "test"]:

    images_dir = DATASET_DIR / split / "images"
    labels_dir = DATASET_DIR / split / "labels"

    image_files = sorted(
        [
            p for p in images_dir.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )

    label_files = sorted(
        labels_dir.glob("*.txt")
    )

    split_class_boxes = Counter()
    split_image_classes = Counter()

    split_modes = Counter()
    split_formats = Counter()

    split_corrupted = []

    split_missing_labels = []

    split_empty_labels = []

    split_multi_box = []

    split_multi_class = []

    # --------------------------------------------------------
    # PROCESS IMAGES
    # --------------------------------------------------------

    for image_path in image_files:

        global_image_count += 1

        # ----------------------------------------------------
        # IMAGE VALIDATION
        # ----------------------------------------------------

        try:

            with Image.open(image_path) as img:

                img.verify()

            # Re-open after verify()
            with Image.open(image_path) as img:

                width, height = img.size
                mode = img.mode
                fmt = img.format

                global_dimensions[
                    (width, height)
                ] += 1

                global_modes[
                    mode
                ] += 1

                global_formats[
                    fmt
                ] += 1

                split_modes[
                    mode
                ] += 1

                split_formats[
                    fmt
                ] += 1

        except Exception as e:

            corrupted_images.append(
                (
                    str(image_path),
                    str(e)
                )
            )

            split_corrupted.append(
                str(image_path)
            )

            continue

        # ----------------------------------------------------
        # FIND MATCHING LABEL
        # ----------------------------------------------------

        label_path = (
            labels_dir /
            f"{image_path.stem}.txt"
        )

        if not label_path.exists():

            missing_label_files.append(
                str(image_path)
            )

            split_missing_labels.append(
                str(image_path)
            )

            continue

        boxes = parse_label_file(label_path)

        # ----------------------------------------------------
        # EMPTY LABEL
        # ----------------------------------------------------

        if boxes == []:

            empty_label_files.append(
                str(label_path)
            )

            split_empty_labels.append(
                str(label_path)
            )

            continue

        if boxes is None:
            continue

        # ----------------------------------------------------
        # BOX STATISTICS
        # ----------------------------------------------------

        if len(boxes) > 1:

            images_with_multiple_boxes.append(
                (
                    str(image_path),
                    len(boxes)
                )
            )

            split_multi_box.append(
                (
                    str(image_path),
                    len(boxes)
                )
            )

        class_ids = [
            box["class_id"]
            for box in boxes
        ]

        unique_classes = set(class_ids)

        if len(unique_classes) > 1:

            images_with_multiple_classes.append(
                (
                    str(image_path),
                    sorted(unique_classes)
                )
            )

            split_multi_class.append(
                (
                    str(image_path),
                    sorted(unique_classes)
                )
            )

        # Count bounding boxes
        for class_id in class_ids:

            split_class_boxes[
                class_id
            ] += 1

            global_class_box_counts[
                class_id
            ] += 1

        # Count images containing each class
        for class_id in unique_classes:

            split_image_classes[
                class_id
            ] += 1

            global_image_class_counts[
                class_id
            ] += 1

    # --------------------------------------------------------
    # SPLIT SUMMARY
    # --------------------------------------------------------

    split_summary[split] = {

        "images":
            len(image_files),

        "labels":
            len(label_files),

        "class_boxes":
            split_class_boxes,

        "image_classes":
            split_image_classes,

        "modes":
            split_modes,

        "formats":
            split_formats,

        "corrupted":
            len(split_corrupted),

        "missing_labels":
            len(split_missing_labels),

        "empty_labels":
            len(split_empty_labels),

        "multiple_boxes":
            len(split_multi_box),

        "multiple_classes":
            len(split_multi_class)
    }


# ============================================================
# PRINT REPORT
# ============================================================

print("\n")
print("=" * 70)
print("ECCKD FULL DATASET AUDIT")
print("=" * 70)

print("\nClass mapping:")

for class_id, class_name in CLASS_MAP.items():

    print(
        f"  {class_id} -> {class_name}"
    )


# ============================================================
# SPLIT REPORT
# ============================================================

for split in ["train", "valid", "test"]:

    info = split_summary[split]

    print("\n")
    print("-" * 70)
    print(f"{split.upper()} SPLIT")
    print("-" * 70)

    print(
        f"Images: {info['images']}"
    )

    print(
        f"Label files: {info['labels']}"
    )

    print(
        f"Corrupted images: {info['corrupted']}"
    )

    print(
        f"Missing labels: {info['missing_labels']}"
    )

    print(
        f"Empty labels: {info['empty_labels']}"
    )

    print(
        f"Images with multiple boxes: "
        f"{info['multiple_boxes']}"
    )

    print(
        f"Images with multiple classes: "
        f"{info['multiple_classes']}"
    )

    print("\nImage modes:")

    for mode, count in info["modes"].items():

        print(
            f"  {mode}: {count}"
        )

    print("\nImage formats:")

    for fmt, count in info["formats"].items():

        print(
            f"  {fmt}: {count}"
        )

    print("\nBounding-box counts:")

    for class_id in CLASS_MAP:

        print(
            f"  {CLASS_MAP[class_id]}: "
            f"{info['class_boxes'][class_id]}"
        )

    print("\nImages containing each class:")

    for class_id in CLASS_MAP:

        print(
            f"  {CLASS_MAP[class_id]}: "
            f"{info['image_classes'][class_id]}"
        )


# ============================================================
# GLOBAL REPORT
# ============================================================

print("\n")
print("=" * 70)
print("GLOBAL DATASET SUMMARY")
print("=" * 70)

print(
    f"\nTotal images: {global_image_count}"
)

print(
    f"Total corrupted images: "
    f"{len(corrupted_images)}"
)

print(
    f"Total images missing labels: "
    f"{len(missing_label_files)}"
)

print(
    f"Total empty label files: "
    f"{len(empty_label_files)}"
)

print(
    f"Invalid annotation lines: "
    f"{len(invalid_label_lines)}"
)

print(
    f"Images with multiple boxes: "
    f"{len(images_with_multiple_boxes)}"
)

print(
    f"Images containing multiple classes: "
    f"{len(images_with_multiple_classes)}"
)


print("\nOverall image-level class counts:")

for class_id in CLASS_MAP:

    print(
        f"  {CLASS_MAP[class_id]}: "
        f"{global_image_class_counts[class_id]}"
    )


print("\nOverall bounding-box counts:")

for class_id in CLASS_MAP:

    print(
        f"  {CLASS_MAP[class_id]}: "
        f"{global_class_box_counts[class_id]}"
    )


print("\nImage color modes:")

for mode, count in global_modes.most_common():

    print(
        f"  {mode}: {count}"
    )


print("\nImage formats:")

for fmt, count in global_formats.most_common():

    print(
        f"  {fmt}: {count}"
    )


print("\nMost common image dimensions:")

for dimensions, count in global_dimensions.most_common(15):

    print(
        f"  {dimensions[0]} x "
        f"{dimensions[1]}: "
        f"{count}"
    )


# ============================================================
# PROBLEM DETAILS
# ============================================================

if corrupted_images:

    print("\n")
    print("=" * 70)
    print("CORRUPTED IMAGES")
    print("=" * 70)

    for path, error in corrupted_images[:20]:

        print(path)
        print(f"  Error: {error}")


if missing_label_files:

    print("\n")
    print("=" * 70)
    print("IMAGES WITHOUT LABEL FILES")
    print("=" * 70)

    for path in missing_label_files[:20]:

        print(path)


if empty_label_files:

    print("\n")
    print("=" * 70)
    print("EMPTY LABEL FILES")
    print("=" * 70)

    for path in empty_label_files[:20]:

        print(path)


if invalid_label_lines:

    print("\n")
    print("=" * 70)
    print("INVALID LABEL ENTRIES")
    print("=" * 70)

    for item in invalid_label_lines[:20]:

        print(item)


if images_with_multiple_classes:

    print("\n")
    print("=" * 70)
    print("IMAGES CONTAINING MULTIPLE CLASSES")
    print("=" * 70)

    for path, classes in images_with_multiple_classes[:20]:

        names = [
            CLASS_MAP[c]
            for c in classes
        ]

        print(
            f"{path} -> {names}"
        )


print("\n")
print("=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)