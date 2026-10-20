"""Generate a comparable ROC curve for the two kidney ultrasound datasets."""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import auc, roc_curve


PROJECT_DIR = Path(__file__).resolve().parents[2]

DEFAULT_DATASETS = {
    "Original dataset": {
        "path": PROJECT_DIR
        / "kidney US images"
        / "results_resnet50_fast_cpu"
        / "test_results"
        / "test_predictions.csv",
        "score_column": "probability_stone",
    },
    "ECCKD group split": {
        "path": PROJECT_DIR
        / "results_ecckd_resnet50"
        / "test_predictions.csv",
        "score_column": "stone_probability",
    },
}


def load_predictions(path: Path, score_column: str) -> tuple[pd.Series, pd.Series]:
    """Load binary labels and stone probabilities from a prediction CSV."""
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    predictions = pd.read_csv(path)
    required_columns = {"true_label", score_column}
    missing_columns = required_columns.difference(predictions.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{path} is missing required column(s): {missing}")

    labels = pd.to_numeric(predictions["true_label"], errors="raise")
    scores = pd.to_numeric(predictions[score_column], errors="raise")

    if not labels.isin([0, 1]).all():
        raise ValueError(f"{path} contains labels other than 0 and 1")

    return labels, scores


def plot_roc_curves(dataset_config: dict[str, dict[str, object]], output_path: Path) -> None:
    """Plot and save ROC curves for all configured datasets."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    auc_rows = []

    plt.figure(figsize=(8, 7))

    for dataset_name, config in dataset_config.items():
        labels, scores = load_predictions(
            Path(config["path"]),
            str(config["score_column"]),
        )
        false_positive_rate, true_positive_rate, _ = roc_curve(labels, scores)
        roc_auc = auc(false_positive_rate, true_positive_rate)

        plt.plot(
            false_positive_rate,
            true_positive_rate,
            linewidth=2,
            label=f"{dataset_name} (AUC = {roc_auc:.4f})",
        )
        auc_rows.append({"dataset": dataset_name, "roc_auc": roc_auc})

    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random classifier (AUC = 0.5000)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC-AUC Comparison")
    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.25)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    auc_path = output_path.with_name(f"{output_path.stem}_values.csv")
    pd.DataFrame(auc_rows).to_csv(auc_path, index=False)
    print(f"ROC graph saved to: {output_path}")
    print(f"AUC values saved to: {auc_path}")
    for row in auc_rows:
        print(f"{row['dataset']}: {row['roc_auc']:.6f}")


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--original-csv",
        type=Path,
        default=DEFAULT_DATASETS["Original dataset"]["path"],
        help="Original dataset test-prediction CSV.",
    )
    parser.add_argument(
        "--ecckd-csv",
        type=Path,
        default=DEFAULT_DATASETS["ECCKD group split"]["path"],
        help="ECCKD group-split test-prediction CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "roc_auc_comparison.png",
        help="Output PNG path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    datasets = {
        "Original dataset": {
            "path": args.original_csv,
            "score_column": "probability_stone",
        },
        "ECCKD group split": {
            "path": args.ecckd_csv,
            "score_column": "stone_probability",
        },
    }
    plot_roc_curves(datasets, args.output)


if __name__ == "__main__":
    main()