from pathlib import Path
import csv
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import EfficientNet_B0_Weights, Swin_T_Weights

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = Path(
    r"D:\Kidney\ECCKD_group_split"
)

EFFICIENTNET_CHECKPOINT = Path(
    r"D:\Kidney\results_ecckd_efficientnet_b0\best_efficientnet_b0_ecckd.pth"
)

SWIN_CHECKPOINT = Path(
    r"D:\Kidney\results_ecckd_swin_tiny\best_swin_tiny_ecckd.pth"
)

RESULTS_DIR = Path(
    r"D:\Kidney\results_ecckd_effnet_swin_ensemble"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

BATCH_SIZE = 8
NUM_WORKERS = 0

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# PRINT HEADER
# ============================================================

print()
print("=" * 75)
print("ECCKD EFFICIENTNET-B0 + SWIN-TINY ENSEMBLE")
print("=" * 75)

print(f"\nDevice: {device}")


# ============================================================
# PREPROCESSING
# ============================================================

image_transform = transforms.Compose(
    [
        transforms.Grayscale(
            num_output_channels=3
        ),

        transforms.Resize(
            (224, 224)
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


# ============================================================
# DATASETS
# ============================================================

val_dataset = datasets.ImageFolder(
    DATASET_DIR / "val",
    transform=image_transform
)

test_dataset = datasets.ImageFolder(
    DATASET_DIR / "test",
    transform=image_transform
)


print(
    "\nClass mapping:",
    val_dataset.class_to_idx
)

print(
    "Validation images:",
    len(val_dataset)
)

print(
    "Test images:",
    len(test_dataset)
)


expected_mapping = {
    "normal": 0,
    "stone": 1
}

if val_dataset.class_to_idx != expected_mapping:
    raise RuntimeError(
        f"Unexpected class mapping.\n"
        f"Found: {val_dataset.class_to_idx}\n"
        f"Expected: {expected_mapping}"
    )


# ============================================================
# DATA LOADERS
# ============================================================

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# LOAD EFFICIENTNET-B0
# ============================================================

print()
print("Loading EfficientNet-B0...")

efficientnet = models.efficientnet_b0(
    weights=None
)

efficientnet_features = (
    efficientnet.classifier[1].in_features
)

efficientnet.classifier[1] = nn.Linear(
    efficientnet_features,
    2
)

efficientnet.load_state_dict(
    torch.load(
        EFFICIENTNET_CHECKPOINT,
        map_location=device
    )
)

efficientnet = efficientnet.to(device)

efficientnet.eval()


# ============================================================
# LOAD SWIN-TINY
# ============================================================

print("Loading Swin-Tiny...")

swin = models.swin_t(
    weights=None
)

swin_features = swin.head.in_features

swin.head = nn.Linear(
    swin_features,
    2
)

swin.load_state_dict(
    torch.load(
        SWIN_CHECKPOINT,
        map_location=device
    )
)

swin = swin.to(device)

swin.eval()


print()
print("Both checkpoints loaded successfully.")


# ============================================================
# GET MODEL PROBABILITIES
# ============================================================

@torch.no_grad()
def get_probabilities(
    efficientnet,
    swin,
    loader,
    description=""
):

    labels_all = []

    efficientnet_probs = []
    swin_probs = []

    progress_bar = tqdm(
        loader,
        desc=description,
        unit="batch"
    )


    for images, labels in progress_bar:

        images = images.to(
            device,
            non_blocking=True
        )


        # EfficientNet output
        eff_outputs = efficientnet(
            images
        )

        eff_probs = torch.softmax(
            eff_outputs,
            dim=1
        )[:, 1]


        # Swin output
        swin_outputs = swin(
            images
        )

        swin_batch_probs = torch.softmax(
            swin_outputs,
            dim=1
        )[:, 1]


        labels_all.extend(
            labels.numpy()
        )

        efficientnet_probs.extend(
            eff_probs
            .cpu()
            .numpy()
        )

        swin_probs.extend(
            swin_batch_probs
            .cpu()
            .numpy()
        )


    return (
        np.array(labels_all),
        np.array(efficientnet_probs),
        np.array(swin_probs)
    )


# ============================================================
# VALIDATION PREDICTIONS
# ============================================================

print()
print("=" * 75)
print("VALIDATION ENSEMBLE WEIGHT SEARCH")
print("=" * 75)


val_labels, val_eff_probs, val_swin_probs = (
    get_probabilities(
        efficientnet,
        swin,
        val_loader,
        description="Validation"
    )
)


# ============================================================
# WEIGHT SEARCH
# ============================================================

# Swin is currently the stronger individual model,
# so search a broad range of weights.
#
# w_swin = 0.00 means only EfficientNet
# w_swin = 1.00 means only Swin
#
# EfficientNet weight = 1 - w_swin

weight_candidates = np.arange(
    0.0,
    1.01,
    0.05
)


weight_results = []

best_weight_swin = None
best_weight_eff = None
best_val_auc = -1.0
best_val_accuracy = -1.0


for w_swin in weight_candidates:

    w_eff = 1.0 - w_swin


    ensemble_probs = (
        w_eff * val_eff_probs
        +
        w_swin * val_swin_probs
    )


    ensemble_preds = (
        ensemble_probs >= 0.5
    ).astype(int)


    auc = roc_auc_score(
        val_labels,
        ensemble_probs
    )

    accuracy = accuracy_score(
        val_labels,
        ensemble_preds
    )

    sensitivity = recall_score(
        val_labels,
        ensemble_preds,
        zero_division=0
    )

    f1 = f1_score(
        val_labels,
        ensemble_preds,
        zero_division=0
    )


    weight_results.append(
        {
            "efficientnet_weight":
                w_eff,

            "swin_weight":
                w_swin,

            "val_accuracy":
                accuracy,

            "val_sensitivity":
                sensitivity,

            "val_f1":
                f1,

            "val_auc":
                auc
        }
    )


    print(
        f"EfficientNet={w_eff:.2f} | "
        f"Swin={w_swin:.2f} | "
        f"Acc={accuracy * 100:.2f}% | "
        f"Sens={sensitivity * 100:.2f}% | "
        f"F1={f1 * 100:.2f}% | "
        f"AUC={auc:.4f}"
    )


    # Primary selection metric = validation ROC-AUC
    # Tie-breaker = validation accuracy

    if (
        auc > best_val_auc
        or
        (
            np.isclose(
                auc,
                best_val_auc
            )
            and
            accuracy > best_val_accuracy
        )
    ):

        best_val_auc = auc
        best_val_accuracy = accuracy

        best_weight_swin = w_swin
        best_weight_eff = w_eff


# ============================================================
# SAVE WEIGHT SEARCH
# ============================================================

weight_file = (
    RESULTS_DIR /
    "validation_weight_search.csv"
)


with open(
    weight_file,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=weight_results[0].keys()
    )

    writer.writeheader()

    writer.writerows(
        weight_results
    )


# ============================================================
# BEST VALIDATION WEIGHT
# ============================================================

print()
print("=" * 75)
print("BEST VALIDATION WEIGHT")
print("=" * 75)

print(
    f"\nEfficientNet-B0 weight : "
    f"{best_weight_eff:.2f}"
)

print(
    f"Swin-Tiny weight       : "
    f"{best_weight_swin:.2f}"
)

print(
    f"Validation ROC-AUC      : "
    f"{best_val_auc:.4f}"
)

print(
    f"Validation Accuracy     : "
    f"{best_val_accuracy * 100:.2f}%"
)


# ============================================================
# TEST PREDICTIONS
# ============================================================

print()
print("=" * 75)
print("FINAL HELD-OUT TEST")
print("=" * 75)


test_labels, test_eff_probs, test_swin_probs = (
    get_probabilities(
        efficientnet,
        swin,
        test_loader,
        description="Testing"
    )
)


# ============================================================
# APPLY LOCKED VALIDATION WEIGHTS
# ============================================================

test_ensemble_probs = (
    best_weight_eff
    *
    test_eff_probs
    +
    best_weight_swin
    *
    test_swin_probs
)


test_predictions = (
    test_ensemble_probs >= 0.5
).astype(int)


# ============================================================
# TEST METRICS
# ============================================================

accuracy = accuracy_score(
    test_labels,
    test_predictions
)

precision = precision_score(
    test_labels,
    test_predictions,
    zero_division=0
)

sensitivity = recall_score(
    test_labels,
    test_predictions,
    zero_division=0
)

f1 = f1_score(
    test_labels,
    test_predictions,
    zero_division=0
)

roc_auc = roc_auc_score(
    test_labels,
    test_ensemble_probs
)

cm = confusion_matrix(
    test_labels,
    test_predictions
)

tn, fp, fn, tp = cm.ravel()

specificity = (
    tn / (tn + fp)
    if (tn + fp) > 0
    else 0
)


# ============================================================
# PRINT FINAL RESULTS
# ============================================================

print()

print(
    f"EfficientNet weight : "
    f"{best_weight_eff:.2f}"
)

print(
    f"Swin weight         : "
    f"{best_weight_swin:.2f}"
)


print()

print(
    f"Accuracy     : "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Precision    : "
    f"{precision * 100:.2f}%"
)

print(
    f"Sensitivity  : "
    f"{sensitivity * 100:.2f}%"
)

print(
    f"Specificity  : "
    f"{specificity * 100:.2f}%"
)

print(
    f"F1-score     : "
    f"{f1 * 100:.2f}%"
)

print(
    f"ROC-AUC      : "
    f"{roc_auc:.4f}"
)


print()
print("Confusion Matrix:")

print(
    cm
)


print()

print(
    f"TN = {tn}"
)

print(
    f"FP = {fp}"
)

print(
    f"FN = {fn}"
)

print(
    f"TP = {tp}"
)


print()
print(
    "Classification Report:"
)

print(
    classification_report(
        test_labels,
        test_predictions,
        target_names=[
            "Normal",
            "Stone"
        ],
        digits=4
    )
)


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

prediction_file = (
    RESULTS_DIR /
    "ensemble_test_predictions.csv"
)


with open(
    prediction_file,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)


    writer.writerow(
        [
            "image_path",
            "true_label",
            "efficientnet_probability",
            "swin_probability",
            "ensemble_probability",
            "ensemble_prediction"
        ]
    )


    for i in range(
        len(test_labels)
    ):

        image_path = (
            test_dataset.samples[i][0]
        )


        writer.writerow(
            [
                image_path,

                int(
                    test_labels[i]
                ),

                float(
                    test_eff_probs[i]
                ),

                float(
                    test_swin_probs[i]
                ),

                float(
                    test_ensemble_probs[i]
                ),

                int(
                    test_predictions[i]
                )
            ]
        )


# ============================================================
# SAVE FINAL RESULTS
# ============================================================

result_file = (
    RESULTS_DIR /
    "ensemble_results.txt"
)


with open(
    result_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "ECCKD EFFICIENTNET-B0 + SWIN-TINY ENSEMBLE\n"
    )

    f.write(
        "=" * 65
        + "\n\n"
    )


    f.write(
        "VALIDATION-SELECTED WEIGHTS\n"
    )

    f.write(
        f"EfficientNet-B0 weight: "
        f"{best_weight_eff:.2f}\n"
    )

    f.write(
        f"Swin-Tiny weight: "
        f"{best_weight_swin:.2f}\n"
    )

    f.write(
        f"Validation ROC-AUC: "
        f"{best_val_auc:.6f}\n\n"
    )


    f.write(
        "FINAL TEST RESULTS\n"
    )

    f.write(
        f"Accuracy: "
        f"{accuracy:.6f}\n"
    )

    f.write(
        f"Precision: "
        f"{precision:.6f}\n"
    )

    f.write(
        f"Sensitivity: "
        f"{sensitivity:.6f}\n"
    )

    f.write(
        f"Specificity: "
        f"{specificity:.6f}\n"
    )

    f.write(
        f"F1-score: "
        f"{f1:.6f}\n"
    )

    f.write(
        f"ROC-AUC: "
        f"{roc_auc:.6f}\n\n"
    )


    f.write(
        "Confusion Matrix:\n"
    )

    f.write(
        str(cm)
    )

    f.write(
        "\n\n"
    )

    f.write(
        f"TN={tn}, "
        f"FP={fp}, "
        f"FN={fn}, "
        f"TP={tp}\n"
    )


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 75)
print("ENSEMBLE COMPLETE")
print("=" * 75)

print(
    f"\nResults saved to:\n"
    f"{RESULTS_DIR}"
)