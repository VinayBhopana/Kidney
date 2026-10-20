from pathlib import Path
import time
import copy
import csv

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import EfficientNet_B0_Weights

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

RESULTS_DIR = Path(
    r"D:\Kidney\results_ecckd_efficientnet_b0"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# CPU-friendly settings
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
NUM_WORKERS = 0
SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print()
print("=" * 75)
print("ECCKD EFFICIENTNET-B0 BASELINE")
print("=" * 75)

print(f"\nDevice: {device}")

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )
else:
    print(
        "Running in CPU mode."
    )


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
# LOAD DATASETS
# ============================================================

train_dataset = datasets.ImageFolder(
    DATASET_DIR / "train",
    transform=image_transform
)

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
    train_dataset.class_to_idx
)

print(
    "Train images:",
    len(train_dataset)
)

print(
    "Validation images:",
    len(val_dataset)
)

print(
    "Test images:",
    len(test_dataset)
)


# ============================================================
# VERIFY CLASS MAPPING
# ============================================================

expected_mapping = {
    "normal": 0,
    "stone": 1
}

if train_dataset.class_to_idx != expected_mapping:

    raise RuntimeError(
        f"\nUnexpected class mapping.\n"
        f"Found: {train_dataset.class_to_idx}\n"
        f"Expected: {expected_mapping}"
    )


# ============================================================
# DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

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
# LOAD PRETRAINED EFFICIENTNET-B0
# ============================================================

print()
print(
    "Loading ImageNet-pretrained EfficientNet-B0..."
)

weights = EfficientNet_B0_Weights.DEFAULT

model = models.efficientnet_b0(
    weights=weights
)


# ============================================================
# FREEZE BACKBONE
# ============================================================

for parameter in model.parameters():
    parameter.requires_grad = False


# ============================================================
# REPLACE CLASSIFICATION HEAD
# ============================================================

# EfficientNet-B0 classifier:
#
# classifier[0] = Dropout
# classifier[1] = Linear
#
# Replace final linear layer with 2 outputs.

num_features = (
    model.classifier[1].in_features
)

model.classifier[1] = nn.Linear(
    num_features,
    2
)

model = model.to(device)


print(
    f"\nClassifier input features: "
    f"{num_features}"
)

print(
    "Backbone frozen: YES"
)

print(
    "Trainable layer: final classifier only"
)


# ============================================================
# MODEL PARAMETER INFORMATION
# ============================================================

total_parameters = sum(
    p.numel()
    for p in model.parameters()
)

trainable_parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

frozen_parameters = (
    total_parameters
    -
    trainable_parameters
)


print(
    f"\nTotal parameters: "
    f"{total_parameters:,}"
)

print(
    f"Trainable parameters: "
    f"{trainable_parameters:,}"
)

print(
    f"Frozen parameters: "
    f"{frozen_parameters:,}"
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

train_targets = np.array(
    train_dataset.targets
)

class_counts = np.bincount(
    train_targets
)

total_samples = len(
    train_targets
)

class_weights = (
    total_samples
    /
    (
        len(class_counts)
        *
        class_counts
    )
)

class_weights_tensor = torch.tensor(
    class_weights,
    dtype=torch.float32,
    device=device
)


print(
    "\nTraining class counts:",
    class_counts
)

print(
    "Class weights:",
    class_weights
)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights_tensor
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.classifier[1].parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# FUNCTION: RUN ONE EPOCH
# ============================================================

def run_epoch(
    model,
    loader,
    training=False,
    description=""
):

    if training:
        model.train()
    else:
        model.eval()


    running_loss = 0.0

    all_labels = []
    all_predictions = []
    all_probabilities = []


    progress_bar = tqdm(
        loader,
        desc=description,
        unit="batch",
        leave=False
    )


    for images, labels in progress_bar:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )


        if training:
            optimizer.zero_grad()


        with torch.set_grad_enabled(
            training
        ):

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                labels
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )[:, 1]

            predictions = torch.argmax(
                outputs,
                dim=1
            )


            if training:

                loss.backward()

                optimizer.step()


        running_loss += (
            loss.item()
            *
            images.size(0)
        )


        all_labels.extend(
            labels
            .detach()
            .cpu()
            .numpy()
        )

        all_predictions.extend(
            predictions
            .detach()
            .cpu()
            .numpy()
        )

        all_probabilities.extend(
            probabilities
            .detach()
            .cpu()
            .numpy()
        )


        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )


    epoch_loss = (
        running_loss
        /
        len(loader.dataset)
    )


    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )


    try:

        auc = roc_auc_score(
            all_labels,
            all_probabilities
        )

    except ValueError:

        auc = float("nan")


    return {

        "loss": epoch_loss,

        "accuracy": accuracy,

        "auc": auc,

        "labels": np.array(
            all_labels
        ),

        "predictions": np.array(
            all_predictions
        ),

        "probabilities": np.array(
            all_probabilities
        )
    }


# ============================================================
# TRAINING
# ============================================================

best_val_auc = -1.0

best_epoch = -1

best_model_state = None

history = []


print()
print("=" * 75)
print("TRAINING")
print("=" * 75)


training_start = time.time()


for epoch in range(
    1,
    NUM_EPOCHS + 1
):

    epoch_start = time.time()


    print()
    print(
        f"Epoch {epoch}/{NUM_EPOCHS}"
    )


    train_result = run_epoch(
        model,
        train_loader,
        training=True,
        description=(
            f"Epoch {epoch} Train"
        )
    )


    val_result = run_epoch(
        model,
        val_loader,
        training=False,
        description=(
            f"Epoch {epoch} Val"
        )
    )


    epoch_time = (
        time.time()
        -
        epoch_start
    )


    print(
        f"Train Loss : "
        f"{train_result['loss']:.4f}"
    )

    print(
        f"Train Acc  : "
        f"{train_result['accuracy'] * 100:.2f}%"
    )

    print(
        f"Train AUC  : "
        f"{train_result['auc']:.4f}"
    )

    print(
        f"Val Loss   : "
        f"{val_result['loss']:.4f}"
    )

    print(
        f"Val Acc    : "
        f"{val_result['accuracy'] * 100:.2f}%"
    )

    print(
        f"Val AUC    : "
        f"{val_result['auc']:.4f}"
    )

    print(
        f"Epoch Time : "
        f"{epoch_time / 60:.2f} minutes"
    )


    history.append(
        {

            "epoch":
                epoch,

            "train_loss":
                train_result[
                    "loss"
                ],

            "train_accuracy":
                train_result[
                    "accuracy"
                ],

            "train_auc":
                train_result[
                    "auc"
                ],

            "val_loss":
                val_result[
                    "loss"
                ],

            "val_accuracy":
                val_result[
                    "accuracy"
                ],

            "val_auc":
                val_result[
                    "auc"
                ]
        }
    )


    # ========================================================
    # SAVE BEST MODEL BASED ON VALIDATION AUC
    # ========================================================

    if (
        val_result["auc"]
        >
        best_val_auc
    ):

        best_val_auc = (
            val_result["auc"]
        )

        best_epoch = epoch

        best_model_state = copy.deepcopy(
            model.state_dict()
        )


        torch.save(
            best_model_state,
            RESULTS_DIR /
            "best_efficientnet_b0_ecckd.pth"
        )


        print(
            "Best model updated."
        )


# ============================================================
# TOTAL TRAINING TIME
# ============================================================

total_training_time = (
    time.time()
    -
    training_start
)


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_file = (
    RESULTS_DIR /
    "training_history.csv"
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


# ============================================================
# LOAD BEST MODEL
# ============================================================

model.load_state_dict(
    best_model_state
)


# ============================================================
# FINAL HELD-OUT TEST
# ============================================================

print()
print("=" * 75)
print("FINAL HELD-OUT TEST")
print("=" * 75)


test_result = run_epoch(
    model,
    test_loader,
    training=False,
    description="Testing"
)


y_true = test_result[
    "labels"
]

y_pred = test_result[
    "predictions"
]

y_prob = test_result[
    "probabilities"
]


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

tn, fp, fn, tp = cm.ravel()


# ============================================================
# TEST METRICS
# ============================================================

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

specificity = (
    tn / (tn + fp)
    if (tn + fp) > 0
    else 0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_true,
    y_prob
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()

print(
    f"Best epoch: "
    f"{best_epoch}"
)

print(
    f"Best validation AUC: "
    f"{best_val_auc:.4f}"
)


print()

print(
    f"Test Loss    : "
    f"{test_result['loss']:.4f}"
)

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

print(
    "Confusion Matrix:"
)

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
        y_true,
        y_pred,
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
    "test_predictions.csv"
)


with open(
    prediction_file,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(
        f
    )

    writer.writerow(
        [
            "true_label",
            "predicted_label",
            "stone_probability"
        ]
    )


    for true_label, pred, prob in zip(
        y_true,
        y_pred,
        y_prob
    ):

        writer.writerow(
            [
                int(true_label),
                int(pred),
                float(prob)
            ]
        )


# ============================================================
# SAVE FINAL RESULTS
# ============================================================

result_file = (
    RESULTS_DIR /
    "efficientnet_b0_results.txt"
)


with open(
    result_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "ECCKD EFFICIENTNET-B0 BASELINE\n"
    )

    f.write(
        "=" * 60
        + "\n\n"
    )

    f.write(
        f"Best epoch: "
        f"{best_epoch}\n"
    )

    f.write(
        f"Best validation AUC: "
        f"{best_val_auc:.6f}\n\n"
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

    f.write(
        f"\nTotal parameters: "
        f"{total_parameters:,}\n"
    )

    f.write(
        f"Trainable parameters: "
        f"{trainable_parameters:,}\n"
    )

    f.write(
        f"Training time: "
        f"{total_training_time / 60:.2f} minutes\n"
    )


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 75)
print("TRAINING COMPLETE")
print("=" * 75)

print(
    f"\nTotal training time: "
    f"{total_training_time / 60:.2f} minutes"
)

print(
    f"\nResults saved to:\n"
    f"{RESULTS_DIR}"
)