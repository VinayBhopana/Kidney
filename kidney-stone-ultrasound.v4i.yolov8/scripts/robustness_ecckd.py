from pathlib import Path
import csv

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.transforms import functional as TF

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from tqdm import tqdm

from PIL import ImageFilter


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
    r"D:\Kidney\results_ecckd_robustness"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

BATCH_SIZE = 8
NUM_WORKERS = 0

EFF_WEIGHT = 0.35
SWIN_WEIGHT = 0.65


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


print()
print("=" * 75)
print("ECCKD ROBUSTNESS TESTING")
print("=" * 75)

print(f"\nDevice: {device}")


# ============================================================
# CUSTOM TRANSFORMS
# ============================================================

class AddGaussianNoise:

    def __init__(
        self,
        mean=0.0,
        std=0.03
    ):

        self.mean = mean
        self.std = std


    def __call__(
        self,
        tensor
    ):

        noise = torch.randn_like(
            tensor
        ) * self.std + self.mean

        tensor = tensor + noise

        return torch.clamp(
            tensor,
            0.0,
            1.0
        )


class GaussianBlurPIL:

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


class DownsampleRestore:

    def __init__(
        self,
        small_size
    ):

        self.small_size = small_size


    def __call__(
        self,
        image
    ):

        image = TF.resize(
            image,
            [
                self.small_size,
                self.small_size
            ]
        )

        image = TF.resize(
            image,
            [
                224,
                224
            ]
        )

        return image


# ============================================================
# NORMALIZATION
# ============================================================

normalize = transforms.Normalize(
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


# ============================================================
# ROBUSTNESS TRANSFORMS
# ============================================================

robustness_transforms = {


    # --------------------------------------------------------
    # ORIGINAL
    # --------------------------------------------------------

    "Original":

        transforms.Compose([
            transforms.Grayscale(
                num_output_channels=3
            ),

            transforms.Resize(
                (224, 224)
            ),

            transforms.ToTensor(),

            normalize
        ]),


    # --------------------------------------------------------
    # GAUSSIAN NOISE
    # --------------------------------------------------------

    "Gaussian_Noise_Mild":

        transforms.Compose([
            transforms.Grayscale(
                num_output_channels=3
            ),

            transforms.Resize(
                (224, 224)
            ),

            transforms.ToTensor(),

            AddGaussianNoise(
                std=0.02
            ),

            normalize
        ]),


    "Gaussian_Noise_Moderate":

        transforms.Compose([
            transforms.Grayscale(
                num_output_channels=3
            ),

            transforms.Resize(
                (224, 224)
            ),

            transforms.ToTensor(),

            AddGaussianNoise(
                std=0.05
            ),

            normalize
        ]),


    # --------------------------------------------------------
    # BLUR
    # --------------------------------------------------------

    "Blur_Mild":

        transforms.Compose([
            transforms.Grayscale(
                num_output_channels=3
            ),

            GaussianBlurPIL(
                radius=1
            ),

            transforms.Resize(
                (224, 224)
            ),

            transforms.ToTensor(),

            normalize
        ]),


    "Blur_Moderate":

        transforms.Compose([
            transforms.Grayscale(
                num_output_channels=3
            ),

            GaussianBlurPIL(
                radius=2
            ),

            transforms.Resize(
                (224, 224)
            ),

            transforms.ToTensor(),

            normalize
        ]),


    # --------------------------------------------------------
    # BRIGHTNESS
    # --------------------------------------------------------

    "Brightness_Low":

        transforms.Compose([
            transforms.Grayscale(
                num_output_channels=3
            ),

            transforms.Resize(
                (224, 224)
            ),

            transforms.ColorJitter(
                brightness=(0.70, 0.70)
            ),

            transforms.ToTensor(),

            normalize
        ]),


    "Brightness_High":

        transforms.Compose([
            transforms.Grayscale(
                num_output_channels=3
            ),

            transforms.Resize(
                (224, 224)
            ),

            transforms.ColorJitter(
                brightness=(1.30, 1.30)
            ),

            transforms.ToTensor(),

            normalize
        ]),


    # --------------------------------------------------------
    # CONTRAST
    # --------------------------------------------------------

    "Contrast_Low":

        transforms.Compose([
            transforms.Grayscale(
                num_output_channels=3
            ),

            transforms.Resize(
                (224, 224)
            ),

            transforms.ColorJitter(
                contrast=(0.70, 0.70)
            ),

            transforms.ToTensor(),

            normalize
        ]),


    "Contrast_High":

        transforms.Compose([
            transforms.Grayscale(
                num_output_channels=3
            ),

            transforms.Resize(
                (224, 224)
            ),

            transforms.ColorJitter(
                contrast=(1.30, 1.30)
            ),

            transforms.ToTensor(),

            normalize
        ]),


    # --------------------------------------------------------
    # SMALL ROTATION
    # --------------------------------------------------------

    "Rotation_Plus_5":

        transforms.Compose([
            transforms.Grayscale(
                num_output_channels=3
            ),

            transforms.Resize(
                (224, 224)
            ),

            transforms.Lambda(
                lambda img:
                TF.rotate(
                    img,
                    angle=5
                )
            ),

            transforms.ToTensor(),

            normalize
        ]),


    "Rotation_Minus_5":

        transforms.Compose([
            transforms.Grayscale(
                num_output_channels=3
            ),

            transforms.Resize(
                (224, 224)
            ),

            transforms.Lambda(
                lambda img:
                TF.rotate(
                    img,
                    angle=-5
                )
            ),

            transforms.ToTensor(),

            normalize
        ]),


    # --------------------------------------------------------
    # RESOLUTION DEGRADATION
    # --------------------------------------------------------

    "Downsample_160":

        transforms.Compose([
            transforms.Grayscale(
                num_output_channels=3
            ),

            transforms.Resize(
                (224, 224)
            ),

            DownsampleRestore(
                160
            ),

            transforms.ToTensor(),

            normalize
        ]),


    "Downsample_112":

        transforms.Compose([
            transforms.Grayscale(
                num_output_channels=3
            ),

            transforms.Resize(
                (224, 224)
            ),

            DownsampleRestore(
                112
            ),

            transforms.ToTensor(),

            normalize
        ])
}


# ============================================================
# LOAD MODELS
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
# METRIC FUNCTION
# ============================================================

def calculate_metrics(
    labels,
    probabilities
):

    predictions = (
        probabilities >= 0.5
    ).astype(int)


    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision = precision_score(
        labels,
        predictions,
        zero_division=0
    )

    sensitivity = recall_score(
        labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0
    )

    auc = roc_auc_score(
        labels,
        probabilities
    )


    cm = confusion_matrix(
        labels,
        predictions
    )

    tn, fp, fn, tp = cm.ravel()


    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
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
# EVALUATION FUNCTION
# ============================================================

@torch.no_grad()
def evaluate_condition(
    condition_name,
    transform
):

    dataset = datasets.ImageFolder(
        DATASET_DIR / "test",
        transform=transform
    )


    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )


    labels_all = []
    ensemble_probs_all = []


    for images, labels in tqdm(
        loader,
        desc=condition_name,
        unit="batch"
    ):

        images = images.to(
            device
        )


        # EfficientNet probability
        eff_output = efficientnet(
            images
        )

        eff_prob = torch.softmax(
            eff_output,
            dim=1
        )[:, 1]


        # Swin probability
        swin_output = swin(
            images
        )

        swin_prob = torch.softmax(
            swin_output,
            dim=1
        )[:, 1]


        # Weighted hybrid
        ensemble_prob = (
            EFF_WEIGHT * eff_prob
            +
            SWIN_WEIGHT * swin_prob
        )


        labels_all.extend(
            labels.numpy()
        )

        ensemble_probs_all.extend(
            ensemble_prob
            .cpu()
            .numpy()
        )


    labels_all = np.array(
        labels_all
    )

    ensemble_probs_all = np.array(
        ensemble_probs_all
    )


    return calculate_metrics(
        labels_all,
        ensemble_probs_all
    )


# ============================================================
# RUN ALL ROBUSTNESS CONDITIONS
# ============================================================

results = []


print()
print("=" * 75)
print("ROBUSTNESS TESTS")
print("=" * 75)


for condition_name, transform in robustness_transforms.items():

    print()
    print(
        f"Testing: {condition_name}"
    )


    metrics = evaluate_condition(
        condition_name,
        transform
    )


    result = {
        "Condition":
            condition_name,

        **metrics
    }


    results.append(
        result
    )


    print(
        f"Accuracy    : "
        f"{metrics['Accuracy'] * 100:.2f}%"
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
        f"F1          : "
        f"{metrics['F1'] * 100:.2f}%"
    )

    print(
        f"ROC-AUC     : "
        f"{metrics['ROC_AUC']:.4f}"
    )

    print(
        f"FN={metrics['FN']} "
        f"FP={metrics['FP']}"
    )


# ============================================================
# CALCULATE PERFORMANCE DROP
# ============================================================

clean_result = results[0]

clean_accuracy = (
    clean_result["Accuracy"]
)

clean_sensitivity = (
    clean_result["Sensitivity"]
)

clean_auc = (
    clean_result["ROC_AUC"]
)


for result in results:

    result["Accuracy_Drop"] = (
        clean_accuracy
        -
        result["Accuracy"]
    )

    result["Sensitivity_Drop"] = (
        clean_sensitivity
        -
        result["Sensitivity"]
    )

    result["AUC_Drop"] = (
        clean_auc
        -
        result["ROC_AUC"]
    )


# ============================================================
# SAVE CSV
# ============================================================

csv_path = (
    RESULTS_DIR /
    "robustness_results.csv"
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


# ============================================================
# PRINT SUMMARY
# ============================================================

print()
print("=" * 75)
print("ROBUSTNESS SUMMARY")
print("=" * 75)


for result in results:

    print()

    print(
        result["Condition"]
    )

    print(
        f"Accuracy: "
        f"{result['Accuracy'] * 100:.2f}% "
        f"(drop "
        f"{result['Accuracy_Drop'] * 100:.2f}%)"
    )

    print(
        f"Sensitivity: "
        f"{result['Sensitivity'] * 100:.2f}% "
        f"(drop "
        f"{result['Sensitivity_Drop'] * 100:.2f}%)"
    )

    print(
        f"AUC: "
        f"{result['ROC_AUC']:.4f} "
        f"(drop "
        f"{result['AUC_Drop']:.4f})"
    )


print()
print("=" * 75)
print("ROBUSTNESS TESTING COMPLETE")
print("=" * 75)

print(
    f"\nResults saved to:\n"
    f"{csv_path}"
)