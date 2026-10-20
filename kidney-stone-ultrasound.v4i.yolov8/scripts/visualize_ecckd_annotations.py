from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import random
import yaml


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = Path(
    r"D:\Kidney\kidney-stone-ultrasound.v4i.yolov8"
)

OUTPUT_DIR = DATASET_DIR / "audit_visualizations"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SAMPLES_PER_CATEGORY = 10

random.seed(42)


# ============================================================
# LOAD CLASS NAMES
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
# PARSE YOLO ANNOTATIONS
# ============================================================

def parse_yolo_label(label_path):

    annotations = []

    text = label_path.read_text(
        encoding="utf-8"
    ).strip()

    if not text:
        return annotations

    for line in text.splitlines():

        parts = line.strip().split()

        if len(parts) < 5:
            continue

        class_id = int(float(parts[0]))

        values = list(
            map(float, parts[1:])
        )

        # ----------------------------------------------------
        # STANDARD YOLO BOUNDING BOX
        # ----------------------------------------------------

        if len(values) == 4:

            annotations.append(
                {
                    "type": "box",
                    "class_id": class_id,
                    "values": values
                }
            )

        # ----------------------------------------------------
        # YOLO POLYGON / SEGMENTATION
        # ----------------------------------------------------

        elif (
            len(values) >= 6
            and
            len(values) % 2 == 0
        ):

            points = []

            for i in range(
                0,
                len(values),
                2
            ):

                points.append(
                    (
                        values[i],
                        values[i + 1]
                    )
                )

            annotations.append(
                {
                    "type": "polygon",
                    "class_id": class_id,
                    "values": points
                }
            )

        else:

            annotations.append(
                {
                    "type": "unknown",
                    "class_id": class_id,
                    "values": values
                }
            )

    return annotations


# ============================================================
# DRAW ANNOTATIONS
# ============================================================

def draw_annotations(
    image_path,
    label_path
):

    image = Image.open(
        image_path
    ).convert("RGB")

    draw = ImageDraw.Draw(image)

    width, height = image.size

    annotations = parse_yolo_label(
        label_path
    )

    for annotation in annotations:

        class_id = annotation[
            "class_id"
        ]

        class_name = CLASS_MAP.get(
            class_id,
            str(class_id)
        )

        # ----------------------------------------------------
        # BOX
        # ----------------------------------------------------

        if annotation["type"] == "box":

            xc, yc, bw, bh = annotation[
                "values"
            ]

            x1 = (
                xc - bw / 2
            ) * width

            y1 = (
                yc - bh / 2
            ) * height

            x2 = (
                xc + bw / 2
            ) * width

            y2 = (
                yc + bh / 2
            ) * height

            draw.rectangle(
                [x1, y1, x2, y2],
                outline="red",
                width=3
            )

            draw.text(
                (x1, max(0, y1 - 15)),
                class_name,
                fill="red"
            )

        # ----------------------------------------------------
        # POLYGON
        # ----------------------------------------------------

        elif annotation[
            "type"
        ] == "polygon":

            points = []

            for x, y in annotation[
                "values"
            ]:

                points.append(
                    (
                        x * width,
                        y * height
                    )
                )

            if len(points) >= 3:

                draw.line(
                    points + [points[0]],
                    fill="yellow",
                    width=3
                )

                draw.text(
                    points[0],
                    class_name,
                    fill="yellow"
                )

    return image


# ============================================================
# COLLECT DATASET SAMPLES
# ============================================================

stone_samples = []
normal_samples = []
empty_samples = []
polygon_samples = []
multi_box_samples = []

annotation_type_counts = {
    "box": 0,
    "polygon": 0,
    "unknown": 0
}


for split in [
    "train",
    "valid",
    "test"
]:

    images_dir = (
        DATASET_DIR /
        split /
        "images"
    )

    labels_dir = (
        DATASET_DIR /
        split /
        "labels"
    )

    for image_path in images_dir.glob(
        "*"
    ):

        if image_path.suffix.lower() not in {
            ".jpg",
            ".jpeg",
            ".png"
        }:

            continue

        label_path = (
            labels_dir /
            f"{image_path.stem}.txt"
        )

        if not label_path.exists():
            continue

        annotations = parse_yolo_label(
            label_path
        )

        record = (
            split,
            image_path,
            label_path
        )

        # ----------------------------------------------------
        # EMPTY
        # ----------------------------------------------------

        if len(annotations) == 0:

            empty_samples.append(
                record
            )

            continue

        # ----------------------------------------------------
        # COUNT TYPES
        # ----------------------------------------------------

        for ann in annotations:

            annotation_type_counts[
                ann["type"]
            ] += 1

            if ann[
                "type"
            ] == "polygon":

                polygon_samples.append(
                    record
                )

        # ----------------------------------------------------
        # IMAGE CLASSES
        # ----------------------------------------------------

        classes = {
            ann["class_id"]
            for ann in annotations
        }

        if 0 in classes:

            stone_samples.append(
                record
            )

        if 1 in classes:

            normal_samples.append(
                record
            )

        if len(annotations) > 1:

            multi_box_samples.append(
                record
            )


# ============================================================
# SAVE SAMPLE IMAGES
# ============================================================

categories = {

    "stone":
        stone_samples,

    "normal":
        normal_samples,

    "empty":
        empty_samples,

    "polygon":
        polygon_samples,

    "multiple_annotations":
        multi_box_samples
}


for category, samples in categories.items():

    category_dir = (
        OUTPUT_DIR /
        category
    )

    category_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    if not samples:
        continue

    selected = random.sample(
        samples,
        min(
            SAMPLES_PER_CATEGORY,
            len(samples)
        )
    )

    for index, (
        split,
        image_path,
        label_path
    ) in enumerate(
        selected,
        start=1
    ):

        image = draw_annotations(
            image_path,
            label_path
        )

        output_name = (
            f"{index:02d}_"
            f"{split}_"
            f"{image_path.name}"
        )

        image.save(
            category_dir /
            output_name
        )


# ============================================================
# PRINT SUMMARY
# ============================================================

print()
print("=" * 70)
print("ECCKD ANNOTATION VISUALIZATION")
print("=" * 70)

print(
    f"\nStone-labelled images: "
    f"{len(stone_samples)}"
)

print(
    f"Normal-labelled images: "
    f"{len(normal_samples)}"
)

print(
    f"Empty annotations: "
    f"{len(empty_samples)}"
)

print(
    f"Images containing polygon annotations: "
    f"{len(set(p[1] for p in polygon_samples))}"
)

print(
    f"Images with multiple annotations: "
    f"{len(multi_box_samples)}"
)

print("\nAnnotation type counts:")

for annotation_type, count in (
    annotation_type_counts.items()
):

    print(
        f"  {annotation_type}: {count}"
    )

print(
    "\nVisualizations saved to:"
)

print(
    OUTPUT_DIR
)

print()
print("=" * 70)
print("VISUALIZATION COMPLETE")
print("=" * 70)