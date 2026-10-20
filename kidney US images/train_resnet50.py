from pathlib import Path
import copy
import csv
import time
import random

import numpy as np

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
    confusion_matrix
)

from tqdm.auto import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = Path(
    r"D:\Kidney\kidney US images\dataset_preprocessed_grayscale_512"
)

OUTPUT_DIR = Path(
    r"D:\Kidney\kidney US images\results_resnet50"
)

# Stored dataset = 512 x 512
# ResNet50 input = 224 x 224
IMAGE_SIZE = 224

BATCH_SIZE = 32

# FULL 10 epochs
EPOCHS = 10

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

RANDOM_SEED = 42


# ============================================================
# PRINT STAGE
# ============================================================

def print_stage(number, total, message):

    print("\n" + "=" * 80)

    print(
        f"STEP {number}/{total} — {message}"
    )

    print("=" * 80)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    use_amp,
    epoch,
    total_epochs
):

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0

    # --------------------------------------------------------
    # TQDM PROGRESS BAR
    # --------------------------------------------------------

    progress_bar = tqdm(
        loader,
        desc=(
            f"Epoch {epoch}/{total_epochs} "
            f"[TRAIN]"
        ),
        unit="batch",
        dynamic_ncols=True,
        leave=True
    )

    for images, labels in progress_bar:

        # ----------------------------------------------------
        # MOVE TO GPU / CPU
        # ----------------------------------------------------

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        # ----------------------------------------------------
        # RESET GRADIENTS
        # ----------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )

        # ----------------------------------------------------
        # FORWARD PASS
        # ----------------------------------------------------

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp
        ):

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

        # ----------------------------------------------------
        # BACKPROPAGATION
        # ----------------------------------------------------

        scaler.scale(
            loss
        ).backward()

        scaler.step(
            optimizer
        )

        scaler.update()

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        batch_size = images.size(0)

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

        current_loss = (
            running_loss / total
        )

        current_accuracy = (
            correct / total
        )

        # ----------------------------------------------------
        # LIVE PROGRESS BAR INFO
        # ----------------------------------------------------

        progress_bar.set_postfix(
            {
                "loss": f"{current_loss:.4f}",
                "acc": (
                    f"{current_accuracy * 100:.2f}%"
                )
            }
        )

    epoch_loss = (
        running_loss
        / len(loader.dataset)
    )

    epoch_accuracy = (
        correct
        / len(loader.dataset)
    )

    return (
        epoch_loss,
        epoch_accuracy
    )


# ============================================================
# VALIDATION / TEST FUNCTION
# ============================================================

def evaluate(
    model,
    loader,
    criterion,
    device,
    use_amp,
    phase_name
):

    model.eval()

    running_loss = 0.0

    targets = []

    predictions = []

    probabilities = []

    correct = 0

    total = 0

    # --------------------------------------------------------
    # PROGRESS BAR
    # --------------------------------------------------------

    progress_bar = tqdm(
        loader,
        desc=f"[{phase_name}]",
        unit="batch",
        dynamic_ncols=True,
        leave=True
    )

    with torch.no_grad():

        for images, labels in progress_bar:

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            # ------------------------------------------------
            # FORWARD PASS
            # ------------------------------------------------

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp
            ):

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )

            batch_size = images.size(0)

            running_loss += (
                loss.item()
                * batch_size
            )

            # ------------------------------------------------
            # PROBABILITY
            # ------------------------------------------------

            probs = torch.softmax(
                outputs,
                dim=1
            )

            predicted = torch.argmax(
                outputs,
                dim=1
            )

            # ------------------------------------------------
            # RUNNING ACCURACY
            # ------------------------------------------------

            correct += (
                predicted == labels
            ).sum().item()

            total += batch_size

            current_accuracy = (
                correct / total
            )

            # ------------------------------------------------
            # SAVE RESULTS
            # ------------------------------------------------

            targets.extend(
                labels.cpu().numpy()
            )

            predictions.extend(
                predicted.cpu().numpy()
            )

            # Stone = positive class = class 1
            probabilities.extend(
                probs[:, 1]
                .cpu()
                .numpy()
            )

            progress_bar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "acc": (
                        f"{current_accuracy * 100:.2f}%"
                    )
                }
            )

    average_loss = (
        running_loss
        / len(loader.dataset)
    )

    return (
        average_loss,
        np.array(targets),
        np.array(predictions),
        np.array(probabilities)
    )


# ============================================================
# CALCULATE METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
    y_probability
):

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
        "tp": tp
    }


# ============================================================
# MAIN
# ============================================================

def main():

    TOTAL_STEPS = 9

    # ========================================================
    # STEP 1
    # ========================================================

    print_stage(
        1,
        TOTAL_STEPS,
        "INITIALIZING ENVIRONMENT"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    set_seed(
        RANDOM_SEED
    )

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    use_amp = (
        device.type == "cuda"
    )

    print(
        f"\nPyTorch version : "
        f"{torch.__version__}"
    )

    print(
        f"Device          : "
        f"{device}"
    )

    print(
        f"CUDA available  : "
        f"{torch.cuda.is_available()}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU             : "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"CUDA version    : "
            f"{torch.version.cuda}"
        )

        gpu_properties = (
            torch.cuda.get_device_properties(0)
        )

        gpu_memory = (
            gpu_properties.total_memory
            / (1024 ** 3)
        )

        print(
            f"GPU VRAM        : "
            f"{gpu_memory:.2f} GB"
        )

    else:

        print(
            "GPU             : CPU only"
        )

    print(
        f"Mixed precision : "
        f"{use_amp}"
    )


    # ========================================================
    # STEP 2
    # ========================================================

    print_stage(
        2,
        TOTAL_STEPS,
        "VERIFYING DATASET"
    )

    print(
        f"\nDataset directory:\n"
        f"{DATASET_DIR}"
    )

    if not DATASET_DIR.exists():

        raise FileNotFoundError(
            f"\nDataset does not exist:\n"
            f"{DATASET_DIR}"
        )

    required_splits = [
        "train",
        "val",
        "test"
    ]

    for split in required_splits:

        split_path = (
            DATASET_DIR
            / split
        )

        if not split_path.exists():

            raise FileNotFoundError(
                f"Missing dataset split:\n"
                f"{split_path}"
            )

    print(
        "\nDataset structure verified."
    )


    # ========================================================
    # STEP 3
    # ========================================================

    print_stage(
        3,
        TOTAL_STEPS,
        "CREATING PREPROCESSING PIPELINE"
    )

    # --------------------------------------------------------
    # ImageNet normalization
    # --------------------------------------------------------

    imagenet_mean = [
        0.485,
        0.456,
        0.406
    ]

    imagenet_std = [
        0.229,
        0.224,
        0.225
    ]

    # --------------------------------------------------------
    # TRAIN TRANSFORM
    # --------------------------------------------------------

    train_transform = transforms.Compose(
        [
            # 512 x 512 grayscale
            #
            # Make 3 identical channels

            transforms.Grayscale(
                num_output_channels=3
            ),

            # Resize only when loading
            #
            # Stored images remain 512 x 512

            transforms.Resize(
                (
                    IMAGE_SIZE,
                    IMAGE_SIZE
                )
            ),

            # Mild augmentation

            transforms.RandomHorizontalFlip(
                p=0.5
            ),

            transforms.RandomRotation(
                degrees=5
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=imagenet_mean,
                std=imagenet_std
            )
        ]
    )

    # --------------------------------------------------------
    # VALIDATION / TEST
    # --------------------------------------------------------

    eval_transform = transforms.Compose(
        [
            transforms.Grayscale(
                num_output_channels=3
            ),

            transforms.Resize(
                (
                    IMAGE_SIZE,
                    IMAGE_SIZE
                )
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=imagenet_mean,
                std=imagenet_std
            )
        ]
    )

    print(
        """
Preprocessing pipeline:

512 x 512 grayscale ultrasound
            ↓
3 identical grayscale channels
            ↓
224 x 224
            ↓
ImageNet normalization
            ↓
ResNet50
"""
    )


    # ========================================================
    # STEP 4
    # ========================================================

    print_stage(
        4,
        TOTAL_STEPS,
        "LOADING DATASETS"
    )

    print(
        "\nLoading TRAIN dataset..."
    )

    train_dataset = datasets.ImageFolder(
        DATASET_DIR / "train",
        transform=train_transform
    )

    print(
        "TRAIN loaded."
    )

    print(
        "\nLoading VALIDATION dataset..."
    )

    val_dataset = datasets.ImageFolder(
        DATASET_DIR / "val",
        transform=eval_transform
    )

    print(
        "VALIDATION loaded."
    )

    print(
        "\nLoading TEST dataset..."
    )

    test_dataset = datasets.ImageFolder(
        DATASET_DIR / "test",
        transform=eval_transform
    )

    print(
        "TEST loaded."
    )

    print(
        "\nClass mapping:"
    )

    print(
        train_dataset.class_to_idx
    )

    # --------------------------------------------------------
    # SAFETY CHECK
    # --------------------------------------------------------

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
            "Expected class mapping:\n"
            "Normal = 0\n"
            "stone = 1"
        )

    print(
        "\nClass mapping verified:"
    )

    print(
        "0 = Normal"
    )

    print(
        "1 = Stone"
    )

    print(
        "\nDataset sizes:"
    )

    print(
        f"Train      : "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation : "
        f"{len(val_dataset)}"
    )

    print(
        f"Test       : "
        f"{len(test_dataset)}"
    )


    # ========================================================
    # STEP 5
    # ========================================================

    print_stage(
        5,
        TOTAL_STEPS,
        "CREATING DATA LOADERS"
    )

    # Windows safe because main() is protected.
    #
    # GPU:
    # 4 workers
    #
    # CPU:
    # 0 workers

    num_workers = (
        4
        if torch.cuda.is_available()
        else 0
    )

    print(
        f"\nBatch size : "
        f"{BATCH_SIZE}"
    )

    print(
        f"Workers    : "
        f"{num_workers}"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    print(
        "\nNumber of batches:"
    )

    print(
        f"Train      : "
        f"{len(train_loader)}"
    )

    print(
        f"Validation : "
        f"{len(val_loader)}"
    )

    print(
        f"Test       : "
        f"{len(test_loader)}"
    )


    # ========================================================
    # STEP 6
    # ========================================================

    print_stage(
        6,
        TOTAL_STEPS,
        "LOADING IMAGENET-PRETRAINED RESNET50"
    )

    print(
        "\nLoading pretrained weights..."
    )

    weights = (
        ResNet50_Weights.DEFAULT
    )

    model = resnet50(
        weights=weights
    )

    print(
        "Pretrained ResNet50 loaded."
    )

    # --------------------------------------------------------
    # Replace ImageNet head
    # --------------------------------------------------------

    number_features = (
        model.fc.in_features
    )

    print(
        f"\nResNet50 FC features: "
        f"{number_features}"
    )

    model.fc = nn.Linear(
        number_features,
        2
    )

    model = model.to(
        device
    )

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"\nTotal parameters     : "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters : "
        f"{trainable_parameters:,}"
    )

    print(
        "\nTraining strategy:"
    )

    print(
        "FULL NETWORK FINE-TUNING"
    )


    # ========================================================
    # STEP 7
    # ========================================================

    print_stage(
        7,
        TOTAL_STEPS,
        "CONFIGURING TRAINING"
    )

    criterion = (
        nn.CrossEntropyLoss()
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=EPOCHS
        )
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp
    )

    print(
        "\nConfiguration:"
    )

    print(
        f"Epochs         : "
        f"{EPOCHS}"
    )

    print(
        "Early stopping : DISABLED"
    )

    print(
        "Training mode  : FULL 10 EPOCHS"
    )

    print(
        f"Batch size     : "
        f"{BATCH_SIZE}"
    )

    print(
        "Optimizer      : AdamW"
    )

    print(
        f"Learning rate  : "
        f"{LEARNING_RATE}"
    )

    print(
        f"Weight decay   : "
        f"{WEIGHT_DECAY}"
    )

    print(
        "Loss           : CrossEntropyLoss"
    )

    print(
        "Scheduler      : CosineAnnealingLR"
    )

    print(
        f"Mixed precision: "
        f"{use_amp}"
    )


    # ========================================================
    # STEP 8
    # ========================================================

    print_stage(
        8,
        TOTAL_STEPS,
        "TRAINING RESNET50 — FULL 10 EPOCHS"
    )

    best_val_auc = -1.0

    best_epoch = 0

    best_model_weights = copy.deepcopy(
        model.state_dict()
    )

    history = []

    training_start = time.time()

    # ========================================================
    # FIXED 10 EPOCH LOOP
    # ========================================================

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

        epoch_start = time.time()

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        (
            train_loss,
            train_accuracy
        ) = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
            epoch=epoch,
            total_epochs=EPOCHS
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        (
            val_loss,
            y_val,
            val_pred,
            val_prob
        ) = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
            phase_name=(
                f"EPOCH {epoch}/{EPOCHS} VALIDATION"
            )
        )

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        val_metrics = calculate_metrics(
            y_val,
            val_pred,
            val_prob
        )

        scheduler.step()

        epoch_time = (
            time.time()
            - epoch_start
        )

        current_lr = (
            optimizer
            .param_groups[0]["lr"]
        )

        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        print(
            "\n"
            + "-" * 80
        )

        print(
            f"EPOCH {epoch} RESULTS"
        )

        print(
            "-" * 80
        )

        print(
            f"Train Loss       : "
            f"{train_loss:.4f}"
        )

        print(
            f"Train Accuracy   : "
            f"{train_accuracy * 100:.2f}%"
        )

        print(
            f"\nValidation Loss  : "
            f"{val_loss:.4f}"
        )

        print(
            f"Val Accuracy     : "
            f"{val_metrics['accuracy'] * 100:.2f}%"
        )

        print(
            f"Val Precision    : "
            f"{val_metrics['precision'] * 100:.2f}%"
        )

        print(
            f"Val Sensitivity  : "
            f"{val_metrics['sensitivity'] * 100:.2f}%"
        )

        print(
            f"Val Specificity  : "
            f"{val_metrics['specificity'] * 100:.2f}%"
        )

        print(
            f"Val F1-score     : "
            f"{val_metrics['f1'] * 100:.2f}%"
        )

        print(
            f"Val ROC-AUC      : "
            f"{val_metrics['auc']:.4f}"
        )

        print(
            f"\nLearning Rate    : "
            f"{current_lr:.8f}"
        )

        print(
            f"Epoch Time       : "
            f"{epoch_time / 60:.2f} minutes"
        )

        print(
            "\nValidation Confusion Matrix:"
        )

        print(
            val_metrics["cm"]
        )

        # ----------------------------------------------------
        # HISTORY
        # ----------------------------------------------------

        history.append(
            {
                "epoch": epoch,

                "train_loss":
                    train_loss,

                "train_accuracy":
                    train_accuracy,

                "val_loss":
                    val_loss,

                "val_accuracy":
                    val_metrics[
                        "accuracy"
                    ],

                "val_precision":
                    val_metrics[
                        "precision"
                    ],

                "val_sensitivity":
                    val_metrics[
                        "sensitivity"
                    ],

                "val_specificity":
                    val_metrics[
                        "specificity"
                    ],

                "val_f1":
                    val_metrics[
                        "f1"
                    ],

                "val_auc":
                    val_metrics[
                        "auc"
                    ],

                "learning_rate":
                    current_lr,

                "epoch_seconds":
                    epoch_time
            }
        )

        # ----------------------------------------------------
        # SAVE BEST MODEL
        #
        # IMPORTANT:
        # THIS DOES NOT STOP TRAINING.
        # ----------------------------------------------------

        if (
            val_metrics["auc"]
            > best_val_auc
        ):

            previous_best = (
                best_val_auc
            )

            best_val_auc = (
                val_metrics["auc"]
            )

            best_epoch = epoch

            best_model_weights = (
                copy.deepcopy(
                    model.state_dict()
                )
            )

            checkpoint_path = (
                OUTPUT_DIR
                / "best_resnet50.pth"
            )

            torch.save(
                best_model_weights,
                checkpoint_path
            )

            print(
                "\n>>> NEW BEST MODEL"
            )

            if previous_best >= 0:

                print(
                    f">>> Previous AUC : "
                    f"{previous_best:.4f}"
                )

            print(
                f">>> New AUC      : "
                f"{best_val_auc:.4f}"
            )

            print(
                f">>> Best epoch   : "
                f"{best_epoch}"
            )

            print(
                f">>> Saved:\n"
                f"{checkpoint_path}"
            )

        else:

            print(
                "\n>>> Validation AUC did not improve."
            )

            print(
                f">>> Current AUC : "
                f"{val_metrics['auc']:.4f}"
            )

            print(
                f">>> Best AUC    : "
                f"{best_val_auc:.4f}"
            )

            print(
                f">>> Best epoch  : "
                f"{best_epoch}"
            )

        # ----------------------------------------------------
        # NEVER INTERRUPT
        # ----------------------------------------------------

        print(
            f"\n>>> Epoch "
            f"{epoch}/{EPOCHS} COMPLETE"
        )

        print(
            f">>> Remaining epochs: "
            f"{EPOCHS - epoch}"
        )

    # ========================================================
    # TRAINING COMPLETE
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "ALL 10 EPOCHS COMPLETED"
    )

    print(
        "=" * 80
    )

    print(
        f"\nBest epoch          : "
        f"{best_epoch}"
    )

    print(
        f"Best Validation AUC : "
        f"{best_val_auc:.4f}"
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
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=history[0].keys()
        )

        writer.writeheader()

        writer.writerows(
            history
        )

    print(
        f"\nTraining history saved:\n"
        f"{history_file}"
    )


    # ========================================================
    # STEP 9
    # ========================================================

    print_stage(
        9,
        TOTAL_STEPS,
        "FINAL TEST USING BEST MODEL"
    )

    print(
        f"\nLoading model from "
        f"best epoch {best_epoch}..."
    )

    model.load_state_dict(
        best_model_weights
    )

    print(
        "Best model loaded."
    )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    (
        test_loss,
        y_true,
        y_pred,
        y_probability
    ) = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        use_amp=use_amp,
        phase_name="FINAL TEST"
    )

    test_metrics = calculate_metrics(
        y_true,
        y_pred,
        y_probability
    )


    # ========================================================
    # FINAL RESULTS
    # ========================================================

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
        f"\nBest Epoch      : "
        f"{best_epoch}"
    )

    print(
        f"Best Val AUC    : "
        f"{best_val_auc:.4f}"
    )

    print(
        f"\nTest Loss       : "
        f"{test_loss:.4f}"
    )

    print(
        f"Accuracy        : "
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
        "\nDetailed:"
    )

    print(
        f"TN — Normal → Normal : "
        f"{test_metrics['tn']}"
    )

    print(
        f"FP — Normal → Stone  : "
        f"{test_metrics['fp']}"
    )

    print(
        f"FN — Stone → Normal  : "
        f"{test_metrics['fn']}"
    )

    print(
        f"TP — Stone → Stone   : "
        f"{test_metrics['tp']}"
    )


    # ========================================================
    # SAVE MODEL
    # ========================================================

    model_path = (
        OUTPUT_DIR
        / "best_resnet50.pth"
    )

    torch.save(
        model.state_dict(),
        model_path
    )


    # ========================================================
    # SAVE RESULTS
    # ========================================================

    result_file = (
        OUTPUT_DIR
        / "resnet50_results.txt"
    )

    with open(
        result_file,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "RESNET50 KIDNEY ULTRASOUND RESULTS\n"
        )

        file.write(
            "=" * 60
            + "\n\n"
        )

        file.write(
            "PIPELINE\n"
        )

        file.write(
            "512x512 grayscale\n"
        )

        file.write(
            "-> 3 identical channels\n"
        )

        file.write(
            "-> 224x224\n"
        )

        file.write(
            "-> ImageNet normalization\n"
        )

        file.write(
            "-> ImageNet pretrained ResNet50\n"
        )

        file.write(
            "-> Full fine-tuning\n"
        )

        file.write(
            "-> Normal / Stone\n\n"
        )

        file.write(
            "TRAINING\n"
        )

        file.write(
            f"Epochs: {EPOCHS}\n"
        )

        file.write(
            "Early stopping: Disabled\n"
        )

        file.write(
            f"Batch size: "
            f"{BATCH_SIZE}\n"
        )

        file.write(
            f"Learning rate: "
            f"{LEARNING_RATE}\n"
        )

        file.write(
            "Optimizer: AdamW\n"
        )

        file.write(
            "Loss: CrossEntropyLoss\n\n"
        )

        file.write(
            "DATASET\n"
        )

        file.write(
            f"Train: "
            f"{len(train_dataset)}\n"
        )

        file.write(
            f"Validation: "
            f"{len(val_dataset)}\n"
        )

        file.write(
            f"Test: "
            f"{len(test_dataset)}\n\n"
        )

        file.write(
            "CLASS MAPPING\n"
        )

        file.write(
            "Normal = 0\n"
        )

        file.write(
            "Stone = 1\n\n"
        )

        file.write(
            "MODEL SELECTION\n"
        )

        file.write(
            f"Best epoch: "
            f"{best_epoch}\n"
        )

        file.write(
            f"Best validation AUC: "
            f"{best_val_auc:.6f}\n\n"
        )

        file.write(
            "FINAL TEST RESULTS\n"
        )

        file.write(
            f"Accuracy: "
            f"{test_metrics['accuracy']:.6f}\n"
        )

        file.write(
            f"Precision: "
            f"{test_metrics['precision']:.6f}\n"
        )

        file.write(
            f"Sensitivity: "
            f"{test_metrics['sensitivity']:.6f}\n"
        )

        file.write(
            f"Specificity: "
            f"{test_metrics['specificity']:.6f}\n"
        )

        file.write(
            f"F1-score: "
            f"{test_metrics['f1']:.6f}\n"
        )

        file.write(
            f"ROC-AUC: "
            f"{test_metrics['auc']:.6f}\n\n"
        )

        file.write(
            "CONFUSION MATRIX\n"
        )

        file.write(
            str(
                test_metrics["cm"]
            )
        )

        file.write(
            "\n\n"
        )

        file.write(
            f"TN: "
            f"{test_metrics['tn']}\n"
        )

        file.write(
            f"FP: "
            f"{test_metrics['fp']}\n"
        )

        file.write(
            f"FN: "
            f"{test_metrics['fn']}\n"
        )

        file.write(
            f"TP: "
            f"{test_metrics['tp']}\n"
        )


    # ========================================================
    # FINISH
    # ========================================================

    total_training_time = (
        time.time()
        - training_start
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "RESNET50 EXPERIMENT COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        f"\nAll epochs completed : "
        f"{EPOCHS}/{EPOCHS}"
    )

    print(
        f"Best epoch           : "
        f"{best_epoch}"
    )

    print(
        f"Best Validation AUC  : "
        f"{best_val_auc:.4f}"
    )

    print(
        f"Total training time  : "
        f"{total_training_time / 60:.2f} minutes"
    )

    print(
        f"\nBest model:\n"
        f"{model_path}"
    )

    print(
        f"\nFinal results:\n"
        f"{result_file}"
    )

    print(
        f"\nTraining history:\n"
        f"{history_file}"
    )


# ============================================================
# WINDOWS MULTIPROCESSING SAFETY
# ============================================================

if __name__ == "__main__":

    main()