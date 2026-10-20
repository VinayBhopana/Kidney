from pathlib import Path
import csv

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
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
    r"D:\Kidney\results_ecckd_ablation"
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
# HEADER
# ============================================================

print()
print("=" * 75)
print("ECCKD ABLATION TESTING")
print("=" * 75)
print(f"\nDevice: {device}")


# ============================================================
# PREPROCESSING
# ============================================================

transform = transforms.Compose([
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
])


# ============================================================
# TEST DATASET
# ============================================================

test_dataset = datasets.ImageFolder(
    DATASET_DIR / "test",
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)


print(
    "\nClass mapping:",
    test_dataset.class_to_idx
)

print(
    "Test images:",
    len(test_dataset)
)


# ============================================================
# LOAD EFFICIENTNET-B0
# ============================================================

print("\nLoading EfficientNet-B0...")

efficientnet = models.efficientnet_b0(
    weights=None
)

num_features = (
    efficientnet.classifier[1].in_features
)

efficientnet.classifier[1] = nn.Linear(
    num_features,
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

num_features = swin.head.in_features

swin.head = nn.Linear(
    num_features,
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


print("Models loaded successfully.")


# ============================================================
# GET PROBABILITIES
# ============================================================

@torch.no_grad()
def get_probabilities():

    labels_all = []

    efficientnet_probs = []
    swin_probs = []


    for images, labels in tqdm(
        test_loader,
        desc="Running models",
        unit="batch"
    ):

        images = images.to(device)


        # EfficientNet
        eff_output = efficientnet(
            images
        )

        eff_prob = torch.softmax(
            eff_output,
            dim=1
        )[:, 1]


        # Swin
        swin_output = swin(
            images
        )

        swin_prob = torch.softmax(
            swin_output,
            dim=1
        )[:, 1]


        labels_all.extend(
            labels.numpy()
        )

        efficientnet_probs.extend(
            eff_prob.cpu().numpy()
        )

        swin_probs.extend(
            swin_prob.cpu().numpy()
        )


    return (
        np.array(labels_all),
        np.array(efficientnet_probs),
        np.array(swin_probs)
    )


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(
    y_true,
    probabilities
):

    predictions = (
        probabilities >= 0.5
    ).astype(int)


    accuracy = accuracy_score(
        y_true,
        predictions
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    sensitivity = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    auc = roc_auc_score(
        y_true,
        probabilities
    )


    cm = confusion_matrix(
        y_true,
        predictions
    )

    tn, fp, fn, tp = cm.ravel()


    specificity = (
        tn / (tn + fp)
    )


    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "F1": f1,
        "ROC_AUC": auc,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    }


# ============================================================
# GET BASE MODEL OUTPUTS
# ============================================================

labels, eff_probs, swin_probs = (
    get_probabilities()
)


# ============================================================
# ABLATION CONFIGURATIONS
# ============================================================

experiments = {
    "EfficientNet_B0_only":
        eff_probs,

    "Swin_Tiny_only":
        swin_probs,

    "Equal_50_50":
        (
            0.50 * eff_probs
            +
            0.50 * swin_probs
        ),

    "Weighted_35_65":
        (
            0.35 * eff_probs
            +
            0.65 * swin_probs
        )
}


# ============================================================
# RUN ABLATIONS
# ============================================================

results = []


print()
print("=" * 75)
print("ABLATION RESULTS")
print("=" * 75)


for name, probabilities in experiments.items():

    metrics = calculate_metrics(
        labels,
        probabilities
    )


    result = {
        "Variant": name,
        **metrics
    }

    results.append(
        result
    )


    print()
    print(name)
    print("-" * 50)

    print(
        f"Accuracy    : "
        f"{metrics['Accuracy'] * 100:.2f}%"
    )

    print(
        f"Precision   : "
        f"{metrics['Precision'] * 100:.2f}%"
    )

    print(
        f"Sensitivity : "
        f"{metrics['Sensitivity'] * 100:.2f}%"
    )

    print(
        f"Specificity : "
        f"{metrics['Specificity'] * 100:.2f}%"
    )

    print(
        f"F1-score    : "
        f"{metrics['F1'] * 100:.2f}%"
    )

    print(
        f"ROC-AUC     : "
        f"{metrics['ROC_AUC']:.4f}"
    )

    print(
        f"TN={metrics['TN']} "
        f"FP={metrics['FP']} "
        f"FN={metrics['FN']} "
        f"TP={metrics['TP']}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

csv_path = (
    RESULTS_DIR /
    "ablation_results.csv"
)

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=results[0].keys()
    )

    writer.writeheader()

    writer.writerows(
        results
    )


print()
print("=" * 75)
print("ABLATION COMPLETE")
print("=" * 75)

print(
    f"\nResults saved to:\n"
    f"{csv_path}"
)