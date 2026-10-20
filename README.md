# Kidney Ultrasound Experiments

This repository contains the preprocessing, dataset-audit, training, evaluation,
robustness, and ensemble scripts used for kidney ultrasound experiments.

## Repository contents

- `kidney US images/`: preprocessing and ResNet experiment scripts
- `kidney-stone-ultrasound.v4i.yolov8/scripts/`: ECCKD training and evaluation scripts
- `cross_dataset_analysis/`: cross-dataset leakage and matching summaries
- `results_ecckd_*/`: lightweight experiment summaries and tabular results

## Data and model artifacts

Image datasets, generated image previews, archives, virtual environments, and
trained model checkpoints are intentionally excluded from Git because they are
large generated or externally sourced artifacts. The scripts expect the
corresponding datasets to be available locally and document their expected
paths in their configuration and source code.

The tracked CSV and text files preserve the lightweight manifests, metrics, and
experiment summaries needed to inspect the reported results without committing
the underlying image data.
