import os
from pathlib import Path
from statistics import mean
from typing import Tuple

import click
import pandas as pd
from sklearn.metrics import roc_auc_score


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

    return (
        true_positives,
        false_positives,
        true_negatives,
        false_negatives,
        auc_score,
        accuracy,
        precision,
        recall,
        f1,
        specificity,
    )


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

    tps = []
    fps = []
    tns = []
    fns = []
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    aucs = []
    specificities = []

    if group_by_target:
        targets = all_results["gene_monitored"].unique()

        for target in targets:
            # Calculate metrics
            tp, fp, tn, fn, auc, accuracy, precision, recall, f1, specificity = (
                calculate_metrics(
                    all_results[all_results["gene_monitored"] == target][
                        "ground_truth"
                    ],
                    all_results[all_results["gene_monitored"] == target][
                        "binary_answer"
                    ],
                )
            )

            tps.append(tp)
            fps.append(fp)
            tns.append(tn)
            fns.append(fn)
            aucs.append(auc)
            accuracies.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            specificities.append(specificity)
    else:
        tp, fp, tn, fn, auc, accuracy, precision, recall, f1, specificity = (
            calculate_metrics(
                all_results["ground_truth"],
                all_results["binary_answer"],
            )
        )

        tps.append(tp)
        fps.append(fp)
        tns.append(tn)
        fns.append(fn)
        aucs.append(auc)
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        specificities.append(specificity)

    # Print results
    print("\nBenchmark Results:")
    print("-----------------")
    print(f"True Positives (TP): {sum(tps)}")
    print(f"False Positives (FP): {sum(fps)}")
    print(f"True Negatives (TN): {sum(tns)}")
    print(f"False Negatives (FN): {sum(fns)}")
    print(f"\nAccuracy: {mean(accuracies):.4f}")
    print(f"Precision: {mean(precisions):.4f}")
    print(f"Recall: {mean(recalls):.4f}")
    print(f"F1 Score: {mean(f1s):.4f}")
    print(f"AUC ROC: {mean(aucs):.4f}")
    print(f"Specificity: {mean(specificities):.4f}")

    # Print metrics in single line with specified format
    print("\nMetrics in single line:")
    print(
        f"{sum(tps)}|{sum(fps)}|{sum(tns)}|{sum(fns)}|{mean(accuracies):.4f}|{mean(precisions):.4f}|{mean(recalls):.4f}|{mean(f1s):.4f}|{mean(specificities):.4f}|{mean(aucs):.4f}"
    )


if __name__ == "__main__":
    main()
