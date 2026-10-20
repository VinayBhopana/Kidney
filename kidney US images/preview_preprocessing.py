from pathlib import Path
import cv2
import random
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = Path(
    r"D:\Kidney\kidney US images\dataset_group_split"
)

OUTPUT_DIR = Path(
    r"D:\Kidney\kidney US images\preprocessing_preview"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLES_PER_CLASS = 10
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

# ============================================================
# PREPROCESSING FUNCTIONS
# ============================================================

def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_clahe(gray):

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    return clahe.apply(gray)


def bilateral_denoise(gray):

    return cv2.bilateralFilter(
        gray,
        d=5,
        sigmaColor=30,
        sigmaSpace=30
    )


def bilateral_clahe(gray):

    denoised = bilateral_denoise(gray)

    return apply_clahe(denoised)


# ============================================================
# CREATE COMPARISON IMAGE
# ============================================================

def add_title(image, title):

    if len(image.shape) == 2:
        image = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2BGR
        )

    canvas = np.zeros(
        (
            image.shape[0] + 40,
            image.shape[1],
            3
        ),
        dtype=np.uint8
    )

    canvas[40:, :] = image

    cv2.putText(
        canvas,
        title,
        (10, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    return canvas


def create_comparison(image_path, output_path):

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        print(f"Could not open {image_path}")
        return

    # -----------------------------------------------
    # Original converted to grayscale
    # -----------------------------------------------

    gray = grayscale(image)

    # -----------------------------------------------
    # CLAHE
    # -----------------------------------------------

    clahe = apply_clahe(gray)

    # -----------------------------------------------
    # Bilateral denoising
    # -----------------------------------------------

    denoised = bilateral_denoise(gray)

    # -----------------------------------------------
    # Bilateral + CLAHE
    # -----------------------------------------------

    denoised_clahe = bilateral_clahe(gray)

    # -----------------------------------------------
    # Make everything same display size
    # -----------------------------------------------

    DISPLAY_SIZE = (350, 350)

    gray = cv2.resize(
        gray,
        DISPLAY_SIZE
    )

    clahe = cv2.resize(
        clahe,
        DISPLAY_SIZE
    )

    denoised = cv2.resize(
        denoised,
        DISPLAY_SIZE
    )

    denoised_clahe = cv2.resize(
        denoised_clahe,
        DISPLAY_SIZE
    )

    # -----------------------------------------------
    # Add labels
    # -----------------------------------------------

    panel1 = add_title(
        gray,
        "1. Grayscale"
    )

    panel2 = add_title(
        clahe,
        "2. CLAHE"
    )

    panel3 = add_title(
        denoised,
        "3. Bilateral"
    )

    panel4 = add_title(
        denoised_clahe,
        "4. Bilateral + CLAHE"
    )

    # -----------------------------------------------
    # Combine 2 × 2
    # -----------------------------------------------

    top = np.hstack(
        [panel1, panel2]
    )

    bottom = np.hstack(
        [panel3, panel4]
    )

    comparison = np.vstack(
        [top, bottom]
    )

    cv2.imwrite(
        str(output_path),
        comparison
    )


# ============================================================
# SAMPLE TRAINING IMAGES
# ============================================================

TRAIN_DIR = (
    DATASET_DIR / "train"
)

classes = [
    "Normal",
    "stone"
]

print("=" * 70)
print("PREPROCESSING VISUAL COMPARISON")
print("=" * 70)

for class_name in classes:

    class_dir = (
        TRAIN_DIR / class_name
    )

    files = [
        file
        for file in class_dir.iterdir()
        if file.suffix.lower()
        in VALID_EXTENSIONS
    ]

    print(
        f"\n{class_name}: "
        f"{len(files)} training images"
    )

    selected = random.sample(
        files,
        min(
            SAMPLES_PER_CLASS,
            len(files)
        )
    )

    output_class_dir = (
        OUTPUT_DIR / class_name
    )

    output_class_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for index, file in enumerate(
        selected,
        start=1
    ):

        output_path = (
            output_class_dir
            / f"{index:02d}_{file.stem}_comparison.jpg"
        )

        create_comparison(
            file,
            output_path
        )

        print(
            f"Created: "
            f"{output_path.name}"
        )

# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

print(
    "\nPreprocessing comparisons saved to:"
)

print(
    OUTPUT_DIR
)

print(
    "\n20 comparison images generated:"
)

print(
    "10 Normal + 10 Stone"
)