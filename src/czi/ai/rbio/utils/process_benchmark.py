import os
from pathlib import Path
from statistics import mean
from typing import Tuple

import click
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_metrics(
    ground_truth: pd.Series, predictions: pd.Series
) -> Tuple[int, int, int, int, float, float, float, float, float, float]:

    # Convert to boolean for easier comparison
    ground_truth_bool = ground_truth.astype(bool)
    predictions_bool = predictions.astype(bool)

    # Calculate confusion matrix elements
    true_positives = ((ground_truth_bool) & (predictions_bool)).sum()
    false_positives = ((~ground_truth_bool) & (predictions_bool)).sum()
    true_negatives = ((~ground_truth_bool) & (~predictions_bool)).sum()
    false_negatives = ((ground_truth_bool) & (~predictions_bool)).sum()

    # Main classification metrics
    tpr = true_positives / (true_positives + false_negatives)
    tnr = true_negatives / (true_negatives + false_positives)
    balanced_accuracy = (tpr + tnr) / 2
    mcc = matthews_corrcoef(ground_truth.to_list(), predictions.to_list())

    # Calculate AUC ROC
    try:
        auc_score = roc_auc_score(ground_truth, predictions)
    except ValueError:
        # Handle case where all predictions are the same
        auc_score = 0.5

    # Calculate additional metrics
    accuracy = (true_positives + true_negatives) / (
        true_positives + true_negatives + false_positives + false_negatives
    )
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0
    )
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )
    specificity = (
        true_negatives / (true_negatives + false_positives)
        if (true_negatives + false_positives) > 0
        else 0
    )

    return {
        "TP": int(true_positives),
        "FP": int(false_positives),
        "TN": int(true_negatives),
        "FN": int(false_negatives),
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "AUC-ROC": auc_score,
        "Specificity": specificity,
        "TPR": tpr,
        "TNR": tnr,
        "Balanced Accuracy": balanced_accuracy,
        "MCC": mcc,
    }


@click.command()
@click.option(
    "--results-csv",
    required=True,
    help="Path to the CSV file containing benchmark results",
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--group-by-target",
    default=False,
    help="Whether the stats should be grouped by target gene and then averaged",
    type=bool,
)
def main(results_csv: str, group_by_target: bool) -> None:
    # Read the CSV file
    all_results = pd.read_csv(results_csv)
    all_results = all_results[~all_results["answer"].isna()]

    # Check for nan values in answer, which would otherwise get converted to a positive
    assert all_results["answer"].isnull().any() == False

    metrics_all = []

    if group_by_target:
        targets = all_results["gene_monitored"].unique()

        for target in targets:
            # Calculate metrics
            metrics = calculate_metrics(
                all_results[all_results["gene_monitored"] == target]["ground_truth"],
                all_results[all_results["gene_monitored"] == target]["binary_answer"],
            )

            metrics_all.append(metrics)
    else:
        metrics = calculate_metrics(
            all_results["ground_truth"],
            all_results["answer"],
        )
        metrics_all.append(metrics)

    metrics_keys = metrics_all[0].keys()
    cum_metrics = {m: [] for m in metrics_keys}
    for m_dict in metrics_all:
        for m, m_val in m_dict.items():
            cum_metrics[m].append(m_val)
    avg_metrics = {m: np.nanmean(cum_metrics[m]) for m in metrics_keys}
    sum_metrics = {m: np.sum(cum_metrics[m]) for m in metrics_keys}

    # Print results
    print("\nBenchmark Results:")
    print("-----------------")

    print(f"True Positives (TP): {sum_metrics['TP']}")
    print(f"False Positives (FP): {sum_metrics['FP']}")
    print(f"True Negatives (TN): {sum_metrics['TN']}")
    print(f"False Negatives (FN): {sum_metrics['FN']}")
    print(f"\nAccuracy: {avg_metrics['Accuracy']:.4f}")
    print(f"Precision: {avg_metrics['Precision']:.4f}")
    print(f"Recall: {avg_metrics['Recall']:.4f}")
    print(f"F1 Score: {avg_metrics['F1-score']:.4f}")
    print(f"AUC ROC: {avg_metrics['AUC-ROC']:.4f}")
    print(f"Specificity: {avg_metrics['Specificity']:.4f}")
    print(f"TPR: {avg_metrics['TPR']:.4f}")
    print(f"TNR: {avg_metrics['TNR']:.4f}")
    print(f"Balanced Accuracy: {avg_metrics['Balanced Accuracy']:.4f}")
    print(f"MCC: {avg_metrics['MCC']:.4f}")
    
    print("\nMetrics in single line:")
    print(
        f"{sum_metrics['TP']}|{sum_metrics['FP']}|{sum_metrics['TN']}|{sum_metrics['FN']}|{avg_metrics['Accuracy']:.4f}|{avg_metrics['Precision']:.4f}|{avg_metrics['Recall']:.4f}|{avg_metrics['Specificity']:.4f}|{avg_metrics['AUC-ROC']:.4f}|{avg_metrics['TPR']:.4f}|{avg_metrics['TNR']:.4f}|{avg_metrics['Balanced Accuracy']:.4f}|{avg_metrics['MCC']:.4f}"
    )


if __name__ == "__main__":
    main()
