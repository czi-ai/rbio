import os
from pathlib import Path
from typing import Tuple

import click
import pandas as pd
from sklearn.metrics import roc_auc_score


def calculate_metrics(
    ground_truth: pd.Series, predictions: pd.Series
) -> Tuple[int, int, int, int, float]:
    """
    Calculate TP, FP, TN, FN and AUC ROC from ground truth and predictions.

    Args:
        ground_truth: Series containing ground truth labels (0 or 1)
        predictions: Series containing predicted labels (0 or 1)

    Returns:
        Tuple containing (TP, FP, TN, FN, AUC)
    """
    # Convert to boolean for easier comparison
    ground_truth_bool = ground_truth.astype(bool)
    predictions_bool = predictions.astype(bool)

    # Calculate confusion matrix elements
    true_positives = ((ground_truth_bool) & (predictions_bool)).sum()
    false_positives = ((~ground_truth_bool) & (predictions_bool)).sum()
    true_negatives = ((~ground_truth_bool) & (~predictions_bool)).sum()
    false_negatives = ((ground_truth_bool) & (~predictions_bool)).sum()

    # Calculate AUC ROC
    try:
        auc_score = roc_auc_score(ground_truth, predictions)
    except ValueError:
        # Handle case where all predictions are the same
        auc_score = 0.5

    return true_positives, false_positives, true_negatives, false_negatives, auc_score


def process_benchmark_results(benchmark_csv_path: os.PathLike) -> None:
    """
    Process benchmark results from a CSV file and print metrics.

    Args:
        benchmark_csv_path: Path to the CSV file containing benchmark results
    """
    # Read the CSV file
    results_df = pd.read_csv(benchmark_csv_path)

    # Calculate metrics
    tp, fp, tn, fn, auc = calculate_metrics(
        results_df["ground_truth"], results_df["binary_answer"]
    )

    # Calculate additional metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    # Print results
    print("\nBenchmark Results:")
    print("-----------------")
    print(f"True Positives (TP): {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"True Negatives (TN): {tn}")
    print(f"False Negatives (FN): {fn}")
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC ROC: {auc:.4f}")


@click.command()
@click.option(
    "--results-csv",
    required=True,
    help="Path to the CSV file containing benchmark results",
    type=click.Path(exists=True, dir_okay=False),
)
def main(results_csv: str) -> None:
    """
    Process benchmark results from a CSV file and display metrics.
    """
    process_benchmark_results(results_csv)


if __name__ == "__main__":
    main()
