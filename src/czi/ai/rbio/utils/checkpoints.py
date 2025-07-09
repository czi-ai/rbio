import glob
import json
import logging
import os
import shutil
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import torch.distributed as dist
from transformers import TrainerCallback

logger = logging.getLogger(__name__)


def remove_incomplete_checkpoints(output_dir: str):
    """
    Removes incomplete checkpoint directories that do not contain a '_SUCCESS' file.
    """
    checkpoint_files = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    for checkpoint_path in checkpoint_files:
        success_file = os.path.join(checkpoint_path, "_SUCCESS")
        if not os.path.exists(success_file):
            logger.warning(f"Removing incomplete checkpoint: {checkpoint_path}")
            shutil.rmtree(checkpoint_path)


def get_last_checkpoint_path(output_dir: str) -> str:
    """
    Returns the path to the last complete checkpoint directory.
    If no complete checkpoint is found, returns None.
    """
    checkpoint_files = sorted(
        glob.glob(os.path.join(output_dir, "checkpoint-*")),
        key=lambda path: int(os.path.basename(path).split("-")[1]),
    )

    if checkpoint_files:
        return checkpoint_files[-1]

    return None


@contextmanager
def checkpoint_recovery(output_dir: str):
    """
    Determine if training should continue from the last complete checkpoint.
    """
    if dist.is_initialized():
        global_rank = dist.get_rank()
    else:
        global_rank = 0

    is_global_rank_zero = global_rank == 0
    running_flag_path = os.path.join(output_dir, "running.json")

    if is_global_rank_zero:
        remove_incomplete_checkpoints(output_dir)

    # Ensure all processes have waited until incomplete checkpoints are removed
    if dist.is_initialized():
        dist.barrier()

    last_checkpoint_path = get_last_checkpoint_path(output_dir)

    # This prevents the case where a different rank reads the running flag after ranks 0 has written it,
    if dist.is_initialized():
        dist.barrier()

    if os.path.exists(running_flag_path) and last_checkpoint_path:
        logger.info(
            f"[RANK {global_rank}] Automatically resuming from checkpoint: {last_checkpoint_path}"
        )
        recover_from_checkpoint = True
    else:
        if is_global_rank_zero:
            logger.info(
                f"[RANK {global_rank}] Writing running flag to {running_flag_path}"
            )
            with open(running_flag_path, "w") as f:
                job_name = os.environ.get("JOB_NAME")
                start_time = datetime.now(ZoneInfo("US/Pacific")).isoformat()
                payload = {"job_name": job_name, "start_time": start_time}
                json.dump(payload, f)
        recover_from_checkpoint = False
    try:
        yield recover_from_checkpoint
    finally:
        if is_global_rank_zero and os.path.exists(running_flag_path):
            os.remove(running_flag_path)


class MarkCheckpointCompleteCallback(TrainerCallback):
    """
    Sometimes a training is interrupted while saving a checkpoint.
    We don't want to resume from an incomplete checkpoint, as that will cause the training to fail.
    The solution is to mark a checkpoint as complete by writing a '_SUCCESS' file to the checkpoint directory.
    """

    def on_save(self, args, state, control, **kwargs):
        checkpoint_files = sorted(
            glob.glob(os.path.join(args.output_dir, "checkpoint-*")),
            key=lambda path: int(os.path.basename(path).split("-")[1]),
        )
        last_checkpoint_path = checkpoint_files[-1]
        success_file = Path(os.path.join(last_checkpoint_path, "_SUCCESS"))
        success_file.touch(exist_ok=True)
