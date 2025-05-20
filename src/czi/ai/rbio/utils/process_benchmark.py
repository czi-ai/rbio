from typing import Dict

import click
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None
) -> Dict[str, float]:
    # Confusion matrix (labels must be [0, 1] for consistent order)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Main classification metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # AUC with probabilities (if available), otherwise use binary
    try:
        if y_prob is not None:
            auc = roc_auc_score(y_true, y_prob)
        else:
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

    # Ensure answer column is numeric if present
    if "answer" in all_results.columns:
        all_results["answer"] = pd.to_numeric(all_results["answer"], errors="coerce")

    y_true_all = all_results["ground_truth"].values
    y_pred_all = all_results["binary_answer"].values
    y_prob_all = (
        all_results["answer"].values if "answer" in all_results.columns else None
    )

    metrics_list = []

    if group_by_target:
        for gene, group in all_results.groupby("gene_monitored"):
            y_true = group["ground_truth"].values
            y_pred = group["binary_answer"].values

            # Attempt to get numeric probabilities
            if "answer" in group.columns:
                y_prob = pd.to_numeric(group["answer"], errors="coerce").values
            else:
                y_prob = None

            if len(np.unique(y_true)) < 2:
                continue  # skip degenerate groups

            metrics = compute_metrics(y_true, y_pred, y_prob)
            metrics_list.append(metrics)
    else:
        metrics = compute_metrics(y_true_all, y_pred_all, y_prob_all)
        metrics_list.append(metrics)

    avg_metrics = {k: np.nanmean([m[k] for m in metrics_list]) for k in metrics_list[0]}

    tp = int(np.nansum([m["TP"] for m in metrics_list]))
    fp = int(np.nansum([m["FP"] for m in metrics_list]))
    tn = int(np.nansum([m["TN"] for m in metrics_list]))
    fn = int(np.nansum([m["FN"] for m in metrics_list]))
    print("\nBenchmark Results:")
    print("-----------------")
    print(f"TP: {tp}")
    print(f"FP: {fp}")
    print(f"TN: {tn}")
    print(f"FN: {fn}")
    print(f"Accuracy: {avg_metrics['Accuracy']:.4f}")
    print(f"Precision: {avg_metrics['Precision']:.4f}")
    print(f"Recall: {avg_metrics['Recall']:.4f}")
    print(f"F1 Score: {avg_metrics['F1-score']:.4f}")
    print(f"AUC ROC: {avg_metrics['AUC ROC']:.4f}")

    row = (
        f"{tp}|{fp}|{tn}|{fn}|{avg_metrics['Accuracy']:.4f}"
        f"|{avg_metrics['Precision']:.4f}|{avg_metrics['Recall']:.4f}"
        f"|{avg_metrics['F1-score']:.4f}|{avg_metrics['AUC ROC']:.4f}"
    )

    print(row)


if __name__ == "__main__":
    main()
