import logging
from typing import Optional

import mlflow
import torch
import torch.distributed as dist
from transformers.integrations.integration_utils import is_mlflow_available


class MetricsCollector:
    """
    A class to collect and store metrics for a distributed PyTorch environment using MLFlow.
    Each GPU process will have their own instance of this class, and they will communicate with each other to log metrics.
    """

    def __init__(self):
        self._world_size: Optional[int] = None
        self._rank: Optional[int] = None
        self._disabled = not is_mlflow_available()

    def _setup_metrics_collector(self):
        """
        This function needs to be called after the distributed process group is initialized.
        """
        if dist.is_initialized():
            """
            We have multiple GPUs, so rank zero needs to send the active run id to all other ranks.
            """
            self._rank = dist.get_rank()
            self._world_size = dist.get_world_size()
        else:
            """
            We have only one GPU, so we can just use the active run id.
            """
            self._rank = 0
            self._world_size = 1

        if self._rank == 0:
            active_run = mlflow.active_run()
            if active_run is None:
                raise RuntimeError(
                    "MLFlow is not active. Please start an MLFlow run before logging metrics."
                )

    def _convert_to_tensor(
        self, metrics_batch: list[dict[str, float]], metrics_keys: list[str]
    ) -> torch.Tensor:
        """
        Convert a batch of metrics into a tensor for fast processing.
        """
        tensor_metrics = torch.tensor(
            [
                [metrics_row[key] for key in metrics_keys]
                for metrics_row in metrics_batch
            ]
        ).to(torch.cuda.current_device())
        return tensor_metrics

    def _convert_from_tensor(
        self, tensor: torch.Tensor, metrics_keys: list[str]
    ) -> list[dict[str, float]]:
        """
        Convert a tensor back into a list of metrics dictionaries.
        """
        metrics_batch = []
        for i in range(tensor.shape[0]):
            metrics_row = {}
            for j, key in enumerate(metrics_keys):
                metrics_row[key] = tensor[i][j].item()
            metrics_batch.append(metrics_row)
        return metrics_batch

    def log_metrics(self, metrics_batch: list[dict[str, float]], step: int):
        """
        Log metrics to MLflow. If the process is distributed, it will average the metrics across all ranks.
        All metrics for a single step should be passed in as a list of dictionaries.
        Logging as a batch to reduce the overhead of distributed communication.
        """

        if self._disabled:
            logging.warning("MLFlow is not available, metrics will not be logged.")
            return

        if not metrics_batch:
            logging.warning("No metrics to log, skipping.")
            return

        if self._rank is None:
            self._setup_metrics_collector()

        metrics_keys = sorted(metrics_batch[0].keys())

        if self._world_size > 1:
            if self._rank > 0:
                """
                We are not rank zero, so we need to send our metrics to rank zero.
                """
                sent_metrics = self._convert_to_tensor(metrics_batch, metrics_keys)
                dist.send(sent_metrics, dst=0)
            else:
                """
                We are rank zero, so we need to receive metrics from all other ranks, average them, and send to MLFlow.
                """
                self_metrics = self._convert_to_tensor(metrics_batch, metrics_keys)
                all_metrics = [self_metrics]

                for recv_rank in range(1, self._world_size):
                    recv_metrics = torch.zeros(all_metrics[0].shape).to(
                        torch.cuda.current_device()
                    )
                    dist.recv(recv_metrics, src=recv_rank)
                    all_metrics.append(recv_metrics)

                assert (
                    len(all_metrics) == self._world_size
                ), "Not all ranks sent their metrics"

                stacked_tensors = torch.stack(all_metrics).to(
                    torch.cuda.current_device()
                )
                averaged_tensors = torch.mean(stacked_tensors, dim=0)

                averaged_metrics = self._convert_from_tensor(
                    averaged_tensors, metrics_keys
                )

                for metrics_row in averaged_metrics:
                    mlflow.log_metrics(metrics_row, step=step)
        else:
            """
            Single GPU case, just log the metrics directly.
            """
            for metrics_row in metrics_batch:
                mlflow.log_metrics(metrics_row, step=step)
