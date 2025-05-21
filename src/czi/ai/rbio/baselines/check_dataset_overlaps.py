import os

import click
import pandas as pd


@click.command()
@click.option(
    "--train-dataset-path", help="Dataset CSV file path", required=True, multiple=False
)
@click.option(
    "--test-dataset-path", help="Dataset CSV file path", required=True, multiple=False
)
def check_gene_pair_overlap(
    train_dataset_path: os.PathLike, test_dataset_path: os.PathLike
) -> None:
    """
    Check for overlapping (gene_perturbed, gene_monitored) pairs in training and test datasets.
    """
    # Load datasets
    df_train = pd.read_csv(train_dataset_path)
    df_test = pd.read_csv(test_dataset_path)

    # Extract pairs
    train_pairs = set(zip(df_train["gene_perturbed"], df_train["gene_monitored"]))
    test_pairs = set(zip(df_test["gene_perturbed"], df_test["gene_monitored"]))

    # Find intersections
    overlap = train_pairs.intersection(test_pairs)

    if overlap:
        click.secho(
            f"⚠️ WARNING: Found {len(overlap)} overlapping (gene_perturbed, gene_monitored) pairs!",
            fg="yellow",
        )
        for pair in sorted(overlap):
            print(f" - {pair}")
    else:
        click.secho(
            "✅ No overlapping gene perturbation-monitoring pairs found between training and test sets.",
            fg="green",
        )


if __name__ == "__main__":
    check_gene_pair_overlap()
