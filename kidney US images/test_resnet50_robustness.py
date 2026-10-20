from pathlib import Path
import json
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import (
    resnet50,
    ResNet50_Weights
)

from PIL import (
    Image,
    ImageFilter,
    ImageEnhance
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from tqdm.auto import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = Path(
    r"D:\Kidney\kidney US images\dataset_preprocessed_grayscale_512"
)

TEST_DIR = (
    DATASET_DIR
    / "test"
)

CHECKPOINT = Path(
    r"D:\Kidney\kidney US images\results_resnet50_fast_cpu\best_resnet50_linear_head.pth"
)

OUTPUT_DIR = Path(
    r"D:\Kidney\kidney US images\results_resnet50_fast_cpu\robustness"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


IMAGE_SIZE = 224

BATCH_SIZE = 32

# Windows-safe
NUM_WORKERS = 0

NUM_CLASSES = 2

CLASS_NAMES = [
    "Normal",
    "stone"
]


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 80)
print("RESNET50 ROBUSTNESS TEST")
print("=" * 80)

print(
    f"\nDevice: "
    f"{DEVICE}"
)

if torch.cuda.is_available():

    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        f"CUDA: "
        f"{torch.version.cuda}"
    )

else:

    print(
        "Running on CPU"
    )


# ============================================================
# CHECK PATHS
# ============================================================

if not TEST_DIR.exists():

    raise FileNotFoundError(
        f"\nTest dataset not found:\n"
        f"{TEST_DIR}"
    )


if not CHECKPOINT.exists():

    raise FileNotFoundError(
        f"\nClassifier checkpoint not found:\n"
        f"{CHECKPOINT}"
    )


print(
    f"\nTest dataset:\n"
    f"{TEST_DIR}"
)

print(
    f"\nClassifier checkpoint:\n"
    f"{CHECKPOINT}"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(42)

np.random.seed(42)

torch.manual_seed(42)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(42)


# ============================================================
# ULTRASOUND CORRUPTIONS
# ============================================================

class Brightness:

    def __init__(
        self,
        factor
    ):

        self.factor = factor


    def __call__(
        self,
        image
    ):

        return ImageEnhance.Brightness(
            image
        ).enhance(
            self.factor
        )


# ------------------------------------------------------------

class Contrast:

    def __init__(
        self,
        factor
    ):

        self.factor = factor


    def __call__(
        self,
        image
    ):

        return ImageEnhance.Contrast(
            image
        ).enhance(
            self.factor
        )


# ------------------------------------------------------------

class GaussianBlur:

    def __init__(
        self,
        radius
    ):

        self.radius = radius


    def __call__(
        self,
        image
    ):

        return image.filter(
            ImageFilter.GaussianBlur(
                radius=self.radius
            )
        )


# ------------------------------------------------------------

class SpeckleNoise:

    def __init__(
        self,
        intensity=0.15
    ):

        self.intensity = intensity


    def __call__(
        self,
        image
    ):

        array = np.asarray(
            image
        ).astype(
            np.float32
        ) / 255.0

        # Multiplicative noise
        # appropriate for ultrasound-like corruption

        noise = np.random.normal(
            loc=1.0,
            scale=self.intensity,
            size=array.shape
        )

        noisy = (
            array
            * noise
        )

        noisy = np.clip(
            noisy,
            0,
            1
        )

        noisy = (
            noisy
            * 255
        ).astype(
            np.uint8
        )

        return Image.fromarray(
            noisy
        )


# ------------------------------------------------------------

class GaussianNoise:

    def __init__(
        self,
        sigma=0.08
    ):

        self.sigma = sigma


    def __call__(
        self,
        image
    ):

        array = np.asarray(
            image
        ).astype(
            np.float32
        ) / 255.0

        noise = np.random.normal(
            loc=0.0,
            scale=self.sigma,
            size=array.shape
        )

        noisy = (
            array
            + noise
        )

        noisy = np.clip(
            noisy,
            0,
            1
        )

        noisy = (
            noisy
            * 255
        ).astype(
            np.uint8
        )

        return Image.fromarray(
            noisy
        )


# ------------------------------------------------------------

class LowResolution:

    def __init__(
        self,
        small_size=96
    ):

        self.small_size = (
            small_size
        )


    def __call__(
        self,
        image
    ):

        # Downsample strongly

        image = image.resize(
            (
                self.small_size,
                self.small_size
            ),
            Image.Resampling.BILINEAR
        )

        # Restore model resolution

        image = image.resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE
            ),
            Image.Resampling.BILINEAR
        )

        return image


# ============================================================
# TRANSFORM BUILDER
# ============================================================

def build_transform(
    corruption=None
):

    transform_list = [

        # Stored dataset is grayscale.
        # Force consistency.

        transforms.Grayscale(
            num_output_channels=1
        ),

        transforms.Resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE
            )
        )
    ]


    # Apply corruption BEFORE tensor conversion

    if corruption is not None:

        transform_list.append(
            corruption
        )


    # Convert grayscale to three identical channels

    transform_list.extend(
        [

            transforms.Grayscale(
                num_output_channels=3
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406
                ],
                std=[
                    0.229,
                    0.224,
                    0.225
                ]
            )
        ]
    )


    return transforms.Compose(
        transform_list
    )


# ============================================================
# CORRUPTION TESTS
# ============================================================

CORRUPTIONS = {

    "Original":
        None,

    "Brightness -25%":
        Brightness(
            0.75
        ),

    "Brightness +25%":
        Brightness(
            1.25
        ),

    "Contrast -25%":
        Contrast(
            0.75
        ),

    "Contrast +25%":
        Contrast(
            1.25
        ),

    "Gaussian Blur":
        GaussianBlur(
            1.5
        ),

    "Gaussian Noise":
        GaussianNoise(
            0.08
        ),

    "Speckle Noise":
        SpeckleNoise(
            0.15
        ),

    "Low Resolution":
        LowResolution(
            96
        )
}


# ============================================================
# CREATE RESNET50 FEATURE EXTRACTOR
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "LOADING RESNET50"
)

print(
    "=" * 80
)


weights = (
    ResNet50_Weights.DEFAULT
)

backbone = resnet50(
    weights=weights
)


FEATURE_DIMENSION = (
    backbone.fc.in_features
)


# Remove ImageNet classifier

backbone.fc = nn.Identity()


# Freeze backbone

for parameter in backbone.parameters():

    parameter.requires_grad = False


backbone = backbone.to(
    DEVICE
)

backbone.eval()


print(
    f"\nFeature dimension: "
    f"{FEATURE_DIMENSION}"
)

print(
    "ResNet50 backbone frozen."
)


# ============================================================
# LOAD YOUR TRAINED CLASSIFIER
# ============================================================

print(
    "\nLoading trained Normal/Stone classifier..."
)


classifier = nn.Linear(
    FEATURE_DIMENSION,
    NUM_CLASSES
)


classifier_state = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=True
)


classifier.load_state_dict(
    classifier_state
)


classifier = classifier.to(
    DEVICE
)

classifier.eval()


print(
    "Classifier loaded successfully."
)

print(
    "\n0 = Normal"
)

print(
    "1 = Stone"
)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

@torch.inference_mode()
def evaluate_dataset(
    dataset,
    description
):

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(
            DEVICE.type == "cuda"
        )
    )


    targets_all = []

    predictions_all = []

    probabilities_all = []


    progress_bar = tqdm(
        loader,
        desc=description,
        unit="batch",
        dynamic_ncols=True
    )


    for images, targets in progress_bar:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        targets = targets.to(
            DEVICE,
            non_blocking=True
        )


        # ====================================================
        # RESNET50 FEATURE EXTRACTION
        # ====================================================

        if DEVICE.type == "cuda":

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):

                features = backbone(
                    images
                )

                outputs = classifier(
                    features
                )

        else:

            features = backbone(
                images
            )

            outputs = classifier(
                features
            )


        # ====================================================
        # PROBABILITIES
        # ====================================================

        probabilities = torch.softmax(
            outputs,
            dim=1
        )


        # Stone probability = class 1

        stone_probability = (
            probabilities[:, 1]
        )


        predictions = torch.argmax(
            outputs,
            dim=1
        )


        targets_all.extend(
            targets.cpu().numpy()
        )

        predictions_all.extend(
            predictions.cpu().numpy()
        )

        probabilities_all.extend(
            stone_probability
            .cpu()
            .numpy()
        )


        current_correct = (
            predictions
            == targets
        ).sum().item()


        current_accuracy = (
            current_correct
            / targets.size(0)
        )


        progress_bar.set_postfix(
            {
                "batch_acc":
                    f"{current_accuracy * 100:.1f}%"
            }
        )


    # ========================================================
    # CONVERT TO NUMPY
    # ========================================================

    targets_all = np.array(
        targets_all
    )

    predictions_all = np.array(
        predictions_all
    )

    probabilities_all = np.array(
        probabilities_all
    )


    # ========================================================
    # METRICS
    # ========================================================

    accuracy = accuracy_score(
        targets_all,
        predictions_all
    )


    precision = precision_score(
        targets_all,
        predictions_all,
        zero_division=0
    )


    stone_recall = recall_score(
        targets_all,
        predictions_all,
        zero_division=0
    )


    f1 = f1_score(
        targets_all,
        predictions_all,
        zero_division=0
    )


    try:

        auc = roc_auc_score(
            targets_all,
            probabilities_all
        )

    except ValueError:

        auc = float(
            "nan"
        )


    cm = confusion_matrix(
        targets_all,
        predictions_all,
        labels=[
            0,
            1
        ]
    )


    tn, fp, fn, tp = (
        cm.ravel()
    )


    specificity = (

        tn
        / (
            tn
            + fp
        )

        if (
            tn
            + fp
        ) > 0

        else 0.0
    )


    metrics = {

        "accuracy":
            accuracy,

        "precision":
            precision,

        "stone_recall":
            stone_recall,

        "specificity":
            specificity,

        "f1":
            f1,

        "auc":
            auc,

        "TN":
            int(tn),

        "FP":
            int(fp),

        "FN":
            int(fn),

        "TP":
            int(tp)
    }


    return metrics


# ============================================================
# RUN ROBUSTNESS TESTS
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "RUNNING ROBUSTNESS TESTS"
)

print(
    "=" * 80
)


results = []

original_metrics = None


for index, (
    condition_name,
    corruption
) in enumerate(
    CORRUPTIONS.items(),
    start=1
):

    print(
        "\n"
        + "-" * 80
    )

    print(
        f"TEST {index}/{len(CORRUPTIONS)}"
    )

    print(
        condition_name
    )

    print(
        "-" * 80
    )


    # ========================================================
    # BUILD TRANSFORM
    # ========================================================

    transform = build_transform(
        corruption
    )


    # ========================================================
    # LOAD TEST SET
    # ========================================================

    dataset = datasets.ImageFolder(
        TEST_DIR,
        transform=transform
    )


    # ========================================================
    # VERIFY CLASS MAPPING
    # ========================================================

    if (
        dataset.class_to_idx.get(
            "Normal"
        ) != 0

        or

        dataset.class_to_idx.get(
            "stone"
        ) != 1
    ):

        raise RuntimeError(
            "Unexpected class mapping.\n"
            "Expected Normal=0, stone=1."
        )


    # ========================================================
    # EVALUATE
    # ========================================================

    metrics = evaluate_dataset(
        dataset,
        condition_name
    )


    # ========================================================
    # SAVE ORIGINAL PERFORMANCE
    # ========================================================

    if condition_name == "Original":

        original_metrics = (
            metrics.copy()
        )


    # ========================================================
    # PERFORMANCE DROP
    # ========================================================

    accuracy_drop = (
        original_metrics[
            "accuracy"
        ]
        - metrics[
            "accuracy"
        ]
    )


    recall_drop = (
        original_metrics[
            "stone_recall"
        ]
        - metrics[
            "stone_recall"
        ]
    )


    f1_drop = (
        original_metrics[
            "f1"
        ]
        - metrics[
            "f1"
        ]
    )


    # ========================================================
    # RESULT ROW
    # ========================================================

    row = {

        "condition":
            condition_name,

        "accuracy":
            metrics[
                "accuracy"
            ],

        "precision":
            metrics[
                "precision"
            ],

        "stone_recall":
            metrics[
                "stone_recall"
            ],

        "specificity":
            metrics[
                "specificity"
            ],

        "f1":
            metrics[
                "f1"
            ],

        "auc":
            metrics[
                "auc"
            ],

        "accuracy_drop":
            accuracy_drop,

        "recall_drop":
            recall_drop,

        "f1_drop":
            f1_drop,

        "TN":
            metrics[
                "TN"
            ],

        "FP":
            metrics[
                "FP"
            ],

        "FN":
            metrics[
                "FN"
            ],

        "TP":
            metrics[
                "TP"
            ]
    }


    results.append(
        row
    )


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print(
        "\nRESULTS"
    )

    print(
        f"Accuracy       : "
        f"{metrics['accuracy'] * 100:.2f}%"
    )

    print(
        f"Precision      : "
        f"{metrics['precision'] * 100:.2f}%"
    )

    print(
        f"Stone Recall   : "
        f"{metrics['stone_recall'] * 100:.2f}%"
    )

    print(
        f"Specificity    : "
        f"{metrics['specificity'] * 100:.2f}%"
    )

    print(
        f"F1-score       : "
        f"{metrics['f1'] * 100:.2f}%"
    )

    print(
        f"ROC-AUC        : "
        f"{metrics['auc']:.4f}"
    )

    print(
        f"Accuracy drop  : "
        f"{accuracy_drop * 100:.2f}%"
    )

    print(
        f"Recall drop    : "
        f"{recall_drop * 100:.2f}%"
    )


    print(
        "\nConfusion Matrix:"
    )

    print(
        np.array(
            [
                [
                    metrics["TN"],
                    metrics["FP"]
                ],
                [
                    metrics["FN"],
                    metrics["TP"]
                ]
            ]
        )
    )


# ============================================================
# SAVE RESULTS AS CSV
# ============================================================

results_df = pd.DataFrame(
    results
)


CSV_FILE = (
    OUTPUT_DIR
    / "resnet50_robustness_results.csv"
)


results_df.to_csv(
    CSV_FILE,
    index=False
)


# ============================================================
# SAVE JSON
# ============================================================

JSON_FILE = (
    OUTPUT_DIR
    / "resnet50_robustness_results.json"
)


with open(
    JSON_FILE,
    "w"
) as file:

    json.dump(
        results,
        file,
        indent=4,
        default=lambda value:
            value.item()
            if isinstance(
                value,
                np.generic
            )
            else value
    )


# ============================================================
# SUMMARY TABLE
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "RESNET50 ROBUSTNESS SUMMARY"
)

print(
    "=" * 100
)


display_df = (
    results_df.copy()
)


percentage_columns = [

    "accuracy",

    "precision",

    "stone_recall",

    "specificity",

    "f1",

    "auc",

    "accuracy_drop",

    "recall_drop",

    "f1_drop"
]


for column in percentage_columns:

    display_df[column] = (
        display_df[column]
        * 100
    ).round(
        2
    )


print(
    display_df[
        [
            "condition",
            "accuracy",
            "stone_recall",
            "specificity",
            "f1",
            "auc",
            "accuracy_drop",
            "recall_drop"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# FIND WORST CONDITIONS
# ============================================================

worst_accuracy = (
    results_df
    .sort_values(
        "accuracy"
    )
    .iloc[0]
)


worst_recall = (
    results_df
    .sort_values(
        "stone_recall"
    )
    .iloc[0]
)


largest_accuracy_drop = (
    results_df
    .sort_values(
        "accuracy_drop",
        ascending=False
    )
    .iloc[0]
)


# ============================================================
# FINDINGS
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "ROBUSTNESS FINDINGS"
)

print(
    "=" * 80
)


print(
    f"\nOriginal accuracy:"
    f"\n{original_metrics['accuracy'] * 100:.2f}%"
)


print(
    f"\nOriginal stone recall:"
    f"\n{original_metrics['stone_recall'] * 100:.2f}%"
)


print(
    f"\nWorst accuracy:"
    f"\n{worst_accuracy['condition']}"
    f" -> "
    f"{worst_accuracy['accuracy'] * 100:.2f}%"
)


print(
    f"\nWorst stone recall:"
    f"\n{worst_recall['condition']}"
    f" -> "
    f"{worst_recall['stone_recall'] * 100:.2f}%"
)


print(
    f"\nLargest accuracy degradation:"
    f"\n{largest_accuracy_drop['condition']}"
    f" -> "
    f"{largest_accuracy_drop['accuracy_drop'] * 100:.2f} percentage points"
)


# ============================================================
# FINISHED
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "RESNET50 ROBUSTNESS TEST COMPLETE"
)

print(
    "=" * 80
)


print(
    f"\nCSV saved to:\n"
    f"{CSV_FILE}"
)


print(
    f"\nJSON saved to:\n"
    f"{JSON_FILE}"
)


print(
    "\nDone."
)