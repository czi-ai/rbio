import click
import pandas as pd
import numpy as np
from typing import Dict
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    # Confusion matrix (labels must be [0, 1] for consistent order)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Main classification metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_pred)
    except ValueError:
        auc = float("nan")

    return {
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "AUC ROC": auc,
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
    is_flag=True,
    help="Use per-gene AUC and classification metrics (as in the baseline paper)",
)
def main(results_csv: str, group_by_target: bool) -> None:
    all_results = pd.read_csv(results_csv)
    all_results = all_results[all_results["binary_answer"] != -1]

    y_true_all = all_results["ground_truth"].values
    y_pred_all = all_results["binary_answer"].values

    metrics_list = []

    if group_by_target:
        for gene, group in all_results.groupby("gene_monitored"):
            y_true = group["ground_truth"].values
            y_pred = group["binary_answer"].values

            # Skip degenerate groups
            if len(np.unique(y_true)) < 2:
                continue

            metrics = compute_metrics(y_true, y_pred)
            metrics_list.append(metrics)
    else:
        metrics = compute_metrics(y_true_all, y_pred_all)
        metrics_list.append(metrics)

    # Average metrics across groups if grouped
    avg_metrics = {k: np.nanmean([m[k] for m in metrics_list]) for k in metrics_list[0]}

    # Print metrics
    print("\nBenchmark Results:")
    print("-----------------")
    print(f"TP: {int(np.nansum([m['TP'] for m in metrics_list]))}")
    print(f"FP: {int(np.nansum([m['FP'] for m in metrics_list]))}")
    print(f"TN: {int(np.nansum([m['TN'] for m in metrics_list]))}")
    print(f"FN: {int(np.nansum([m['FN'] for m in metrics_list]))}")
    print(f"Accuracy: {avg_metrics['Accuracy']:.4f}")
    print(f"Precision: {avg_metrics['Precision']:.4f}")
    print(f"Recall: {avg_metrics['Recall']:.4f}")
    print(f"F1 Score: {avg_metrics['F1-score']:.4f}")
    print(f"AUC ROC: {avg_metrics['AUC ROC']:.4f}")


if __name__ == "__main__":
    main()
