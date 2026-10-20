from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import (
    resnet50,
    ResNet50_Weights
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
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
    r"D:\Kidney\kidney US images\results_resnet50_fast_cpu\test_results"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

IMAGE_SIZE = 224

BATCH_SIZE = 32

# Safe for Windows CPU
NUM_WORKERS = 0

CLASS_NAMES = [
    "Normal",
    "stone"
]

NUM_CLASSES = 2


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 80)
print("RESNET50 FINAL TEST EVALUATION")
print("=" * 80)

print(
    f"\nDevice: {DEVICE}"
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
# CHECK FILES
# ============================================================

if not TEST_DIR.exists():

    raise FileNotFoundError(
        f"\nTest dataset not found:\n"
        f"{TEST_DIR}"
    )

if not CHECKPOINT.exists():

    raise FileNotFoundError(
        f"\nCheckpoint not found:\n"
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
# TEST PREPROCESSING
# ============================================================

test_transform = transforms.Compose(
    [
        # Stored images are grayscale.
        # Convert into 3 identical channels because
        # pretrained ResNet50 expects 3-channel input.

        transforms.Grayscale(
            num_output_channels=3
        ),

        # 512 x 512
        # ↓
        # 224 x 224

        transforms.Resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE
            )
        ),

        transforms.ToTensor(),

        # ImageNet normalization

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


# ============================================================
# LOAD TEST DATASET
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "LOADING TEST DATASET"
)

print(
    "=" * 80
)


test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=test_transform
)


print(
    "\nClass mapping:"
)

print(
    test_dataset.class_to_idx
)


# Expected:
# Normal = 0
# stone = 1

if (
    test_dataset.class_to_idx.get(
        "Normal"
    ) != 0
    or
    test_dataset.class_to_idx.get(
        "stone"
    ) != 1
):

    raise RuntimeError(
        "Unexpected class mapping.\n"
        "Expected Normal=0 and stone=1."
    )


print(
    f"\nTest images: "
    f"{len(test_dataset)}"
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(
        DEVICE.type == "cuda"
    )
)


print(
    f"Test batches: "
    f"{len(test_loader)}"
)


# ============================================================
# LOAD RESNET50 FEATURE EXTRACTOR
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "LOADING IMAGENET-PRETRAINED RESNET50"
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


print(
    f"\nFeature dimension: "
    f"{FEATURE_DIMENSION}"
)


# Remove ImageNet classifier.
#
# ResNet50 now outputs:
#
# [batch_size, 2048]

backbone.fc = nn.Identity()


# Freeze backbone

for parameter in backbone.parameters():

    parameter.requires_grad = False


backbone = backbone.to(
    DEVICE
)

backbone.eval()


print(
    "Frozen ResNet50 backbone ready."
)


# ============================================================
# LOAD TRAINED LINEAR CLASSIFIER
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "LOADING TRAINED NORMAL / STONE CLASSIFIER"
)

print(
    "=" * 80
)


classifier = nn.Linear(
    FEATURE_DIMENSION,
    NUM_CLASSES
)


state_dict = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=True
)


classifier.load_state_dict(
    state_dict
)


classifier = classifier.to(
    DEVICE
)

classifier.eval()


print(
    "\nClassifier loaded successfully."
)

print(
    "2048 features -> 2 classes"
)

print(
    "\n0 = Normal"
)

print(
    "1 = Stone"
)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# FINAL TEST
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "RUNNING FINAL HELD-OUT TEST"
)

print(
    "=" * 80
)


targets_all = []

predictions_all = []

probabilities_all = []

loss_total = 0.0


progress = tqdm(
    test_loader,
    desc="Test",
    unit="batch",
    dynamic_ncols=True
)


with torch.inference_mode():

    for images, targets in progress:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        targets = targets.to(
            DEVICE,
            non_blocking=True
        )


        # ====================================================
        # FORWARD PASS
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

                loss = criterion(
                    outputs,
                    targets
                )

        else:

            features = backbone(
                images
            )

            outputs = classifier(
                features
            )

            loss = criterion(
                outputs,
                targets
            )


        loss_total += (
            loss.item()
            * images.size(0)
        )


        # ====================================================
        # PROBABILITIES
        # ====================================================

        probabilities = torch.softmax(
            outputs,
            dim=1
        )


        stone_probabilities = (
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
            stone_probabilities
            .cpu()
            .numpy()
        )


        batch_accuracy = (
            predictions == targets
        ).float().mean().item()


        progress.set_postfix(
            {
                "loss":
                    f"{loss.item():.4f}",

                "batch_acc":
                    f"{batch_accuracy * 100:.1f}%"
            }
        )


# ============================================================
# CONVERT RESULTS
# ============================================================

targets_all = np.array(
    targets_all
)

predictions_all = np.array(
    predictions_all
)

probabilities_all = np.array(
    probabilities_all
)


test_loss = (
    loss_total
    / len(test_dataset)
)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    targets_all,
    predictions_all
)


precision = precision_score(
    targets_all,
    predictions_all,
    zero_division=0
)


# Stone recall / sensitivity
sensitivity = recall_score(
    targets_all,
    predictions_all,
    zero_division=0
)


f1 = f1_score(
    targets_all,
    predictions_all,
    zero_division=0
)


auc = roc_auc_score(
    targets_all,
    probabilities_all
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
)


# ============================================================
# PRINT RESULTS
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "FINAL RESNET50 TEST RESULTS"
)

print(
    "=" * 80
)


print(
    f"\nTest Loss       : "
    f"{test_loss:.4f}"
)

print(
    f"Accuracy        : "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Precision       : "
    f"{precision * 100:.2f}%"
)

print(
    f"Sensitivity     : "
    f"{sensitivity * 100:.2f}%"
)

print(
    f"Specificity     : "
    f"{specificity * 100:.2f}%"
)

print(
    f"F1-score        : "
    f"{f1 * 100:.2f}%"
)

print(
    f"ROC-AUC         : "
    f"{auc:.4f}"
)


print(
    "\nConfusion Matrix:"
)

print(
    cm
)


print(
    "\nDetailed:"
)

print(
    f"TN - Normal -> Normal : "
    f"{tn}"
)

print(
    f"FP - Normal -> Stone  : "
    f"{fp}"
)

print(
    f"FN - Stone -> Normal  : "
    f"{fn}"
)

print(
    f"TP - Stone -> Stone   : "
    f"{tp}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    targets_all,
    predictions_all,
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0
)


print(
    "\n"
    + "=" * 80
)

print(
    "CLASSIFICATION REPORT"
)

print(
    "=" * 80
)

print(
    report
)


REPORT_FILE = (
    OUTPUT_DIR
    / "classification_report.txt"
)


with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        report
    )


# ============================================================
# SAVE PER-IMAGE PREDICTIONS
# ============================================================

rows = []


for index, (
    image_path,
    true_label
) in enumerate(
    test_dataset.samples
):

    predicted_label = int(
        predictions_all[
            index
        ]
    )

    stone_probability = float(
        probabilities_all[
            index
        ]
    )


    rows.append(
        {
            "image_path":
                image_path,

            "true_label":
                true_label,

            "true_class":
                CLASS_NAMES[
                    true_label
                ],

            "predicted_label":
                predicted_label,

            "predicted_class":
                CLASS_NAMES[
                    predicted_label
                ],

            "probability_stone":
                stone_probability,

            "correct":
                predicted_label
                == true_label
        }
    )


predictions_df = pd.DataFrame(
    rows
)


PREDICTIONS_FILE = (
    OUTPUT_DIR
    / "test_predictions.csv"
)


predictions_df.to_csv(
    PREDICTIONS_FILE,
    index=False
)


# ============================================================
# SAVE MISCLASSIFIED IMAGES ONLY
# ============================================================

misclassified_df = (
    predictions_df[
        predictions_df[
            "correct"
        ] == False
    ]
)


MISCLASSIFIED_FILE = (
    OUTPUT_DIR
    / "misclassified_images.csv"
)


misclassified_df.to_csv(
    MISCLASSIFIED_FILE,
    index=False
)


print(
    f"\nMisclassified images: "
    f"{len(misclassified_df)}"
)


# ============================================================
# SAVE FINAL METRICS
# ============================================================

final_metrics = {

    "model":
        "ResNet50",

    "pretrained":
        True,

    "pretraining":
        "ImageNet",

    "training_strategy":
        "Frozen ResNet50 + Linear Classifier",

    "stored_image_size":
        "512x512",

    "model_input_size":
        "224x224",

    "feature_dimension":
        FEATURE_DIMENSION,

    "test_images":
        len(test_dataset),

    "accuracy":
        float(
            accuracy
        ),

    "precision":
        float(
            precision
        ),

    "sensitivity":
        float(
            sensitivity
        ),

    "specificity":
        float(
            specificity
        ),

    "f1":
        float(
            f1
        ),

    "roc_auc":
        float(
            auc
        ),

    "TN":
        int(
            tn
        ),

    "FP":
        int(
            fp
        ),

    "FN":
        int(
            fn
        ),

    "TP":
        int(
            tp
        )
}


METRICS_FILE = (
    OUTPUT_DIR
    / "final_test_metrics.json"
)


with open(
    METRICS_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        final_metrics,
        file,
        indent=4
    )


# ============================================================
# CONFUSION MATRIX FIGURE
# ============================================================

plt.figure(
    figsize=(
        6,
        5
    )
)


plt.imshow(
    cm,
    interpolation="nearest"
)


plt.title(
    "ResNet50 - Confusion Matrix"
)


plt.colorbar()


plt.xticks(
    [
        0,
        1
    ],
    CLASS_NAMES
)


plt.yticks(
    [
        0,
        1
    ],
    CLASS_NAMES
)


plt.xlabel(
    "Predicted"
)


plt.ylabel(
    "Actual"
)


for i in range(2):

    for j in range(2):

        plt.text(
            j,
            i,
            cm[
                i,
                j
            ],
            ha="center",
            va="center"
        )


plt.tight_layout()


CONFUSION_MATRIX_FILE = (
    OUTPUT_DIR
    / "confusion_matrix.png"
)


plt.savefig(
    CONFUSION_MATRIX_FILE,
    dpi=200
)


plt.close()


# ============================================================
# ROC CURVE
# ============================================================

fpr, tpr, thresholds = roc_curve(
    targets_all,
    probabilities_all
)


plt.figure(
    figsize=(
        7,
        6
    )
)


plt.plot(
    fpr,
    tpr,
    label=(
        f"AUC = "
        f"{auc:.4f}"
    )
)


plt.plot(
    [
        0,
        1
    ],
    [
        0,
        1
    ],
    linestyle="--"
)


plt.xlabel(
    "False Positive Rate"
)


plt.ylabel(
    "True Positive Rate"
)


plt.title(
    "ResNet50 - ROC Curve"
)


plt.legend()


plt.grid(
    alpha=0.3
)


plt.tight_layout()


ROC_FILE = (
    OUTPUT_DIR
    / "roc_curve.png"
)


plt.savefig(
    ROC_FILE,
    dpi=200
)


plt.close()


# ============================================================
# FINAL SUMMARY
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "TEST EVALUATION COMPLETE"
)

print(
    "=" * 80
)


print(
    f"\nResults directory:\n"
    f"{OUTPUT_DIR}"
)


print(
    f"\nMetrics:\n"
    f"{METRICS_FILE}"
)


print(
    f"\nClassification report:\n"
    f"{REPORT_FILE}"
)


print(
    f"\nAll predictions:\n"
    f"{PREDICTIONS_FILE}"
)


print(
    f"\nMisclassified images:\n"
    f"{MISCLASSIFIED_FILE}"
)


print(
    f"\nConfusion matrix:\n"
    f"{CONFUSION_MATRIX_FILE}"
)


print(
    f"\nROC curve:\n"
    f"{ROC_FILE}"
)


print(
    "\nDone."
)