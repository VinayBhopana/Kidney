from pathlib import Path
import copy
import csv
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from tqdm.auto import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = Path(
    r"D:\Kidney\kidney US images\dataset_preprocessed_grayscale_512"
)

OUTPUT_DIR = Path(
    r"D:\Kidney\kidney US images\results_resnet50_fast_cpu"
)

CACHE_DIR = OUTPUT_DIR / "feature_cache"

IMAGE_SIZE = 224

# Image batches used only while extracting ResNet50 features
FEATURE_BATCH_SIZE = 32

# Features are tiny, so classifier batches can be large
CLASSIFIER_BATCH_SIZE = 512

# Full 10 epochs - these are fast because only classifier trains
EPOCHS = 10

CLASSIFIER_LR = 1e-3
WEIGHT_DECAY = 1e-4

RANDOM_SEED = 42

DEVICE = torch.device("cpu")


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred, y_probability):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    sensitivity = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    auc = roc_auc_score(
        y_true,
        y_probability
    )

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    tn, fp, fn, tp = cm.ravel()

    specificity = (
        tn / (tn + fp)
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "auc": auc,
        "cm": cm,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(
    model,
    loader,
    split_name,
    cache_file
):

    # --------------------------------------------------------
    # If already extracted, load cache
    # --------------------------------------------------------

    if cache_file.exists():

        print(
            f"\nCached {split_name} features found."
        )

        print(
            f"Loading:\n{cache_file}"
        )

        data = torch.load(
            cache_file,
            map_location="cpu"
        )

        return (
            data["features"],
            data["labels"]
        )

    # --------------------------------------------------------
    # Otherwise extract them
    # --------------------------------------------------------

    print(
        f"\nExtracting {split_name} features..."
    )

    model.eval()

    all_features = []
    all_labels = []

    progress_bar = tqdm(
        loader,
        desc=f"ResNet50 [{split_name}]",
        unit="batch",
        dynamic_ncols=True
    )

    # No gradients needed
    with torch.inference_mode():

        for images, labels in progress_bar:

            images = images.to(
                DEVICE
            )

            # ResNet50 produces 2048-D vectors
            features = model(
                images
            )

            all_features.append(
                features.cpu()
            )

            all_labels.append(
                labels.cpu()
            )

            processed = sum(
                tensor.shape[0]
                for tensor in all_labels
            )

            progress_bar.set_postfix(
                {
                    "images": processed
                }
            )

    features = torch.cat(
        all_features,
        dim=0
    )

    labels = torch.cat(
        all_labels,
        dim=0
    )

    # --------------------------------------------------------
    # Save so this expensive step NEVER needs repeating
    # --------------------------------------------------------

    torch.save(
        {
            "features": features,
            "labels": labels
        },
        cache_file
    )

    print(
        f"\n{split_name} features saved."
    )

    print(
        f"Feature shape: {features.shape}"
    )

    print(
        f"Cache:\n{cache_file}"
    )

    return (
        features,
        labels
    )


# ============================================================
# CLASSIFIER TRAINING
# ============================================================

def train_classifier_epoch(
    classifier,
    loader,
    criterion,
    optimizer,
    epoch
):

    classifier.train()

    running_loss = 0.0

    correct = 0
    total = 0

    progress_bar = tqdm(
        loader,
        desc=f"Epoch {epoch}/{EPOCHS} [CLASSIFIER]",
        unit="batch",
        dynamic_ncols=True
    )

    for features, labels in progress_bar:

        optimizer.zero_grad(
            set_to_none=True
        )

        outputs = classifier(
            features
        )

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        batch_size = labels.size(0)

        running_loss += (
            loss.item()
            * batch_size
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += batch_size

        progress_bar.set_postfix(
            {
                "loss":
                    f"{running_loss / total:.4f}",

                "acc":
                    f"{correct / total * 100:.2f}%"
            }
        )

    return (
        running_loss / total,
        correct / total
    )


# ============================================================
# CLASSIFIER EVALUATION
# ============================================================

def evaluate_classifier(
    classifier,
    loader,
    phase
):

    classifier.eval()

    y_true = []
    y_pred = []
    y_prob = []

    progress_bar = tqdm(
        loader,
        desc=f"[{phase}]",
        unit="batch",
        dynamic_ncols=True
    )

    with torch.inference_mode():

        for features, labels in progress_bar:

            outputs = classifier(
                features
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            y_true.extend(
                labels.numpy()
            )

            y_pred.extend(
                predictions.numpy()
            )

            # Stone = positive class = 1
            y_prob.extend(
                probabilities[:, 1].numpy()
            )

    y_true = np.array(
        y_true
    )

    y_pred = np.array(
        y_pred
    )

    y_prob = np.array(
        y_prob
    )

    return calculate_metrics(
        y_true,
        y_pred,
        y_prob
    )


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(
        RANDOM_SEED
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        "=" * 80
    )

    print(
        "FAST CPU RESNET50 KIDNEY ULTRASOUND BASELINE"
    )

    print(
        "=" * 80
    )

    print(
        "\nStrategy:"
    )

    print(
        "Frozen ImageNet-pretrained ResNet50"
    )

    print(
        "        ↓"
    )

    print(
        "Extract 2048-D features once"
    )

    print(
        "        ↓"
    )

    print(
        "Cache features"
    )

    print(
        "        ↓"
    )

    print(
        "Train 2-class linear head for 10 epochs"
    )

    print(
        "\nDevice: CPU"
    )

    print(
        f"CPU logical cores: {os.cpu_count()}"
    )


    # ========================================================
    # 1. PREPROCESSING
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 1 — PREPROCESSING"
    )

    print(
        "=" * 80
    )

    transform = transforms.Compose(
        [
            # Force grayscale and replicate
            # into 3 identical channels

            transforms.Grayscale(
                num_output_channels=3
            ),

            # Stored files stay at 512x512.
            # Resize happens only here.

            transforms.Resize(
                (
                    IMAGE_SIZE,
                    IMAGE_SIZE
                )
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

    print(
        """
512x512 grayscale
        ↓
3 identical channels
        ↓
224x224
        ↓
ImageNet normalization
"""
    )


    # ========================================================
    # 2. LOAD DATA
    # ========================================================

    print(
        "=" * 80
    )

    print(
        "STEP 2 — LOADING DATASET"
    )

    print(
        "=" * 80
    )

    train_dataset = datasets.ImageFolder(
        DATASET_DIR / "train",
        transform=transform
    )

    val_dataset = datasets.ImageFolder(
        DATASET_DIR / "val",
        transform=transform
    )

    test_dataset = datasets.ImageFolder(
        DATASET_DIR / "test",
        transform=transform
    )

    print(
        "\nClass mapping:"
    )

    print(
        train_dataset.class_to_idx
    )

    if (
        train_dataset.class_to_idx.get(
            "Normal"
        ) != 0
        or
        train_dataset.class_to_idx.get(
            "stone"
        ) != 1
    ):

        raise RuntimeError(
            "Expected Normal=0 and stone=1"
        )

    print(
        f"\nTrain      : {len(train_dataset)}"
    )

    print(
        f"Validation : {len(val_dataset)}"
    )

    print(
        f"Test       : {len(test_dataset)}"
    )


    # ========================================================
    # CPU DATA LOADERS
    # ========================================================

    # Keep workers at 0 on Windows CPU.
    # PyTorch itself can use CPU threading.

    train_image_loader = DataLoader(
        train_dataset,
        batch_size=FEATURE_BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    val_image_loader = DataLoader(
        val_dataset,
        batch_size=FEATURE_BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    test_image_loader = DataLoader(
        test_dataset,
        batch_size=FEATURE_BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )


    # ========================================================
    # 3. LOAD RESNET50
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 3 — LOADING PRETRAINED RESNET50"
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

    feature_dimension = (
        backbone.fc.in_features
    )

    print(
        f"\nOriginal FC input: "
        f"{feature_dimension}"
    )

    # Remove final ImageNet classifier
    backbone.fc = nn.Identity()

    # Freeze everything
    for parameter in backbone.parameters():

        parameter.requires_grad = False

    backbone = backbone.to(
        DEVICE
    )

    backbone.eval()

    print(
        "\nResNet50 backbone frozen."
    )

    print(
        "Output feature dimension: 2048"
    )


    # ========================================================
    # 4. EXTRACT FEATURES
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 4 — EXTRACTING / LOADING CACHED FEATURES"
    )

    print(
        "=" * 80
    )

    feature_start = time.time()

    train_features, train_labels = extract_features(
        backbone,
        train_image_loader,
        "TRAIN",
        CACHE_DIR / "train_features.pt"
    )

    val_features, val_labels = extract_features(
        backbone,
        val_image_loader,
        "VALIDATION",
        CACHE_DIR / "val_features.pt"
    )

    test_features, test_labels = extract_features(
        backbone,
        test_image_loader,
        "TEST",
        CACHE_DIR / "test_features.pt"
    )

    feature_time = (
        time.time()
        - feature_start
    )

    print(
        f"\nFeature extraction time: "
        f"{feature_time / 60:.2f} minutes"
    )


    # ========================================================
    # Free large backbone
    # ========================================================

    del backbone


    # ========================================================
    # 5. FEATURE DATA LOADERS
    # ========================================================

    train_feature_dataset = TensorDataset(
        train_features,
        train_labels
    )

    val_feature_dataset = TensorDataset(
        val_features,
        val_labels
    )

    test_feature_dataset = TensorDataset(
        test_features,
        test_labels
    )

    train_loader = DataLoader(
        train_feature_dataset,
        batch_size=CLASSIFIER_BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_feature_dataset,
        batch_size=CLASSIFIER_BATCH_SIZE,
        shuffle=False
    )

    test_loader = DataLoader(
        test_feature_dataset,
        batch_size=CLASSIFIER_BATCH_SIZE,
        shuffle=False
    )


    # ========================================================
    # 6. CLASSIFICATION HEAD
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 5 — CREATING NORMAL / STONE CLASSIFIER"
    )

    print(
        "=" * 80
    )

    classifier = nn.Linear(
        feature_dimension,
        2
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=CLASSIFIER_LR,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=EPOCHS
        )
    )

    print(
        "\nClassifier:"
    )

    print(
        "2048 features → 2 classes"
    )

    print(
        "\n0 = Normal"
    )

    print(
        "1 = Stone"
    )

    print(
        f"\nClassifier parameters: "
        f"{sum(p.numel() for p in classifier.parameters()):,}"
    )


    # ========================================================
    # 7. TRAIN 10 FULL EPOCHS
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 6 — TRAINING CLASSIFIER FOR 10 FULL EPOCHS"
    )

    print(
        "=" * 80
    )

    best_auc = -1.0

    best_epoch = 0

    best_weights = copy.deepcopy(
        classifier.state_dict()
    )

    history = []

    training_start = time.time()

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        print(
            "\n"
            + "#" * 80
        )

        print(
            f"EPOCH {epoch}/{EPOCHS}"
        )

        print(
            "#" * 80
        )

        train_loss, train_accuracy = (
            train_classifier_epoch(
                classifier,
                train_loader,
                criterion,
                optimizer,
                epoch
            )
        )

        val_metrics = evaluate_classifier(
            classifier,
            val_loader,
            f"EPOCH {epoch} VALIDATION"
        )

        scheduler.step()

        print(
            "\n"
            + "-" * 70
        )

        print(
            f"EPOCH {epoch} RESULTS"
        )

        print(
            "-" * 70
        )

        print(
            f"Train Loss      : "
            f"{train_loss:.4f}"
        )

        print(
            f"Train Accuracy  : "
            f"{train_accuracy * 100:.2f}%"
        )

        print(
            f"\nVal Accuracy    : "
            f"{val_metrics['accuracy'] * 100:.2f}%"
        )

        print(
            f"Val Precision   : "
            f"{val_metrics['precision'] * 100:.2f}%"
        )

        print(
            f"Val Sensitivity : "
            f"{val_metrics['sensitivity'] * 100:.2f}%"
        )

        print(
            f"Val Specificity : "
            f"{val_metrics['specificity'] * 100:.2f}%"
        )

        print(
            f"Val F1-score    : "
            f"{val_metrics['f1'] * 100:.2f}%"
        )

        print(
            f"Val ROC-AUC     : "
            f"{val_metrics['auc']:.4f}"
        )

        print(
            "\nConfusion Matrix:"
        )

        print(
            val_metrics["cm"]
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss":
                    train_loss,
                "train_accuracy":
                    train_accuracy,
                "val_accuracy":
                    val_metrics["accuracy"],
                "val_precision":
                    val_metrics["precision"],
                "val_sensitivity":
                    val_metrics["sensitivity"],
                "val_specificity":
                    val_metrics["specificity"],
                "val_f1":
                    val_metrics["f1"],
                "val_auc":
                    val_metrics["auc"],
            }
        )

        # ----------------------------------------------------
        # SAVE BEST, BUT RUN ALL 10
        # ----------------------------------------------------

        if val_metrics["auc"] > best_auc:

            best_auc = (
                val_metrics["auc"]
            )

            best_epoch = (
                epoch
            )

            best_weights = (
                copy.deepcopy(
                    classifier.state_dict()
                )
            )

            torch.save(
                best_weights,
                OUTPUT_DIR
                / "best_resnet50_linear_head.pth"
            )

            print(
                "\n>>> NEW BEST CHECKPOINT"
            )

            print(
                f">>> Epoch: {best_epoch}"
            )

            print(
                f">>> AUC  : {best_auc:.4f}"
            )

        print(
            f"\nCompleted {epoch}/10"
        )

    training_time = (
        time.time()
        - training_start
    )


    # ========================================================
    # SAVE HISTORY
    # ========================================================

    history_file = (
        OUTPUT_DIR
        / "training_history.csv"
    )

    with open(
        history_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=history[0].keys()
        )

        writer.writeheader()

        writer.writerows(
            history
        )


    # ========================================================
    # 8. FINAL TEST
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 7 — FINAL TEST"
    )

    print(
        "=" * 80
    )

    classifier.load_state_dict(
        best_weights
    )

    test_metrics = evaluate_classifier(
        classifier,
        test_loader,
        "FINAL TEST"
    )


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "FINAL FAST-CPU RESNET50 RESULTS"
    )

    print(
        "=" * 80
    )

    print(
        f"\nBest epoch      : "
        f"{best_epoch}"
    )

    print(
        f"Best Val AUC    : "
        f"{best_auc:.4f}"
    )

    print(
        f"\nAccuracy        : "
        f"{test_metrics['accuracy'] * 100:.2f}%"
    )

    print(
        f"Precision       : "
        f"{test_metrics['precision'] * 100:.2f}%"
    )

    print(
        f"Sensitivity     : "
        f"{test_metrics['sensitivity'] * 100:.2f}%"
    )

    print(
        f"Specificity     : "
        f"{test_metrics['specificity'] * 100:.2f}%"
    )

    print(
        f"F1-score        : "
        f"{test_metrics['f1'] * 100:.2f}%"
    )

    print(
        f"ROC-AUC         : "
        f"{test_metrics['auc']:.4f}"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(
        test_metrics["cm"]
    )

    print(
        "\nTN =", test_metrics["tn"]
    )

    print(
        "FP =", test_metrics["fp"]
    )

    print(
        "FN =", test_metrics["fn"]
    )

    print(
        "TP =", test_metrics["tp"]
    )


    # ========================================================
    # SAVE RESULTS
    # ========================================================

    result_file = (
        OUTPUT_DIR
        / "fast_cpu_results.txt"
    )

    with open(
        result_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "FAST CPU RESNET50 BASELINE\n"
        )

        f.write(
            "=" * 60 + "\n\n"
        )

        f.write(
            "Method: Frozen ResNet50 feature extractor "
            "+ linear classifier\n\n"
        )

        f.write(
            "Input: 512x512 grayscale\n"
        )

        f.write(
            "Model input: 224x224, 3 identical channels\n"
        )

        f.write(
            "Backbone: ImageNet-pretrained ResNet50\n"
        )

        f.write(
            "Feature dimension: 2048\n"
        )

        f.write(
            "Classifier: Linear 2048 -> 2\n\n"
        )

        f.write(
            f"Best epoch: {best_epoch}\n"
        )

        f.write(
            f"Best Val AUC: {best_auc:.6f}\n\n"
        )

        f.write(
            f"Accuracy: "
            f"{test_metrics['accuracy']:.6f}\n"
        )

        f.write(
            f"Precision: "
            f"{test_metrics['precision']:.6f}\n"
        )

        f.write(
            f"Sensitivity: "
            f"{test_metrics['sensitivity']:.6f}\n"
        )

        f.write(
            f"Specificity: "
            f"{test_metrics['specificity']:.6f}\n"
        )

        f.write(
            f"F1-score: "
            f"{test_metrics['f1']:.6f}\n"
        )

        f.write(
            f"ROC-AUC: "
            f"{test_metrics['auc']:.6f}\n\n"
        )

        f.write(
            "Confusion Matrix:\n"
        )

        f.write(
            str(test_metrics["cm"])
        )


    # ========================================================
    # DONE
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        f"\nFeature extraction time : "
        f"{feature_time / 60:.2f} min"
    )

    print(
        f"Classifier training time: "
        f"{training_time / 60:.2f} min"
    )

    print(
        f"\nResults:\n{result_file}"
    )

    print(
        f"\nFeature cache:\n{CACHE_DIR}"
    )


if __name__ == "__main__":
    main()