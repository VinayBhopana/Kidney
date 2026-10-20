from pathlib import Path
import yaml

# -------------------------------------------------
# 1. Define dataset path
# -------------------------------------------------

DATASET_DIR = Path(
    r"D:\Kidney\kidney-stone-ultrasound.v4i.yolov8"
)

YAML_PATH = DATASET_DIR / "data.yaml"

# -------------------------------------------------
# 2. Check whether data.yaml exists
# -------------------------------------------------

if not YAML_PATH.exists():
    raise FileNotFoundError(
        f"Could not find data.yaml at:\n{YAML_PATH}"
    )

print("=" * 60)
print("ECCKD DATASET CONFIGURATION CHECK")
print("=" * 60)

print(f"\nDataset folder:\n{DATASET_DIR}")
print(f"\ndata.yaml:\n{YAML_PATH}")

# -------------------------------------------------
# 3. Load YAML
# -------------------------------------------------

with open(YAML_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

print("\n" + "=" * 60)
print("RAW YAML CONTENT")
print("=" * 60)

for key, value in config.items():
    print(f"{key}: {value}")

# -------------------------------------------------
# 4. Extract class information
# -------------------------------------------------

names = config.get("names", None)
nc = config.get("nc", None)

print("\n" + "=" * 60)
print("CLASS INFORMATION")
print("=" * 60)

if nc is not None:
    print(f"Number of classes declared: {nc}")
else:
    print("nc field not explicitly present.")

if names is None:
    print("No 'names' field found.")
else:

    # YOLO YAML can store names either as a list:
    # names: ['stone', 'normal']
    #
    # or dictionary:
    # names:
    #   0: stone
    #   1: normal

    if isinstance(names, list):

        print(f"Number of classes found: {len(names)}")

        for class_id, class_name in enumerate(names):
            print(f"Class {class_id}: {class_name}")

    elif isinstance(names, dict):

        print(f"Number of classes found: {len(names)}")

        for class_id, class_name in names.items():
            print(f"Class {class_id}: {class_name}")

    else:
        print(f"Unexpected class format: {type(names)}")

# -------------------------------------------------
# 5. Check split folders
# -------------------------------------------------

print("\n" + "=" * 60)
print("DATASET SPLIT CHECK")
print("=" * 60)

for split in ["train", "valid", "test"]:

    split_dir = DATASET_DIR / split
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"

    print(f"\n{split.upper()}")

    print(
        f"Images directory exists: "
        f"{images_dir.exists()}"
    )

    print(
        f"Labels directory exists: "
        f"{labels_dir.exists()}"
    )

print("\n" + "=" * 60)
print("CHECK COMPLETE")
print("=" * 60)