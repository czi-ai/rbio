import os
from collections import defaultdict

import mlflow
import torch.distributed as dist


class MetricsCollector:
    """
    A class to collect and store metrics for a distributed PyTorch environment using MLFlow.
    Each GPU process will have their own instance of this class, and they will communicate with each other to log metrics.
    """

    def __init__(self):
        self._mlflow_run_id: str = None
        self._world_size: int = None
        self._rank: int = None

    def setup_mlflow_run(self):
        """
        This function needs to be called after the distributed process group is initialized.
        """
        if dist.is_initialized():
            """
            We have multiple GPUs, so rank zero needs to send the active run id to all other ranks.
            """
            self._rank = dist.get_rank()
            self._world_size = dist.get_world_size()

            if self._rank == 0:
                """
                We are rank zero, so we need to get the active run id and send it to all other ranks.
                """
                active_run = mlflow.active_run()
                if active_run and active_run.info:
                    run_id = active_run.info.run_id
                else:
                    run_id = None

                assert (
                    run_id is not None
                ), "Could not determine MLFlow run id in rank zero"

                for recv_rank in range(1, self._world_size):
                    dist.send_object_list([run_id], dst=recv_rank)

                self._mlflow_run_id = run_id
            else:
                """
                We are not rank zero, so we need to receive the active run id from rank zero.
                """
                sent_objects = [None]
                dist.recv_object_list(sent_objects, src=0)
                self._mlflow_run_id = sent_objects[0]
                mlflow.start_run(run_id=self._mlflow_run_id)
        else:
            """
            We have only one GPU, so we can just use the active run id.
            """
            self._rank = 0
            self._world_size = 1

            active_run = mlflow.active_run()
            if active_run and active_run.info:
                run_id = active_run.info.run_id
            else:
                run_id = None

            assert run_id is not None, "Could not determine MLFlow run id in rank zero"
            self._mlflow_run_id = run_id

    def log_metrics(self, metrics_batch: list[dict[str, float]], step: int):
        """
        Log metrics to MLflow. If the process is distributed, it will average the metrics across all ranks.
        All metrics for a single step should be passed in as a list of dictionaries.
        Logging as a batch to reduce the overhead of distributed communication.
        """

        if self._mlflow_run_id is None:
            self.setup_mlflow_run()

        if self._world_size > 1:
            if self._rank > 0:
                """
                We are not rank zero, so we need to send our metrics to rank zero.
                """
                dist.send_object_list(metrics_batch, dst=0)
            else:
                """
                We are rank zero, so we need to receive metrics from all other ranks, average them, and send to MLFlow.
                """
                all_metrics = [metrics_batch]

                for recv_rank in range(1, self._world_size):
                    sent_objects = [None] * len(metrics_batch)
                    dist.recv_object_list(sent_objects, src=recv_rank)
                    all_metrics.append(sent_objects)

                assert (
                    len(all_metrics) == self._world_size
                ), "Not all ranks sent their metrics"

                for index in range(len(metrics_batch)):
                    averaged_metrics = defaultdict(float)

                    for rank in range(self._world_size):
                        for key in metrics_batch[0].keys():
                            averaged_metrics[key] += all_metrics[rank][index][key]

                    for key in metrics_batch[0].keys():
                        averaged_metrics[key] /= self._world_size

                    mlflow.log_metrics(averaged_metrics, step=step)
        else:
            """
            Single GPU case, just log the metrics directly.
            """
            for metrics_row in metrics_batch:
                mlflow.log_metrics(metrics_row, step=step)
