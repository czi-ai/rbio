#!/usr/bin/env bash

MACHINE_RANK=${RANK:-0}
NUM_MACHINES=${PET_NNODES:-1}
MAIN_PROCESS_IP=${MASTER_ADDR:-""}
MAIN_PROCESS_PORT=${MASTER_PORT:-9090}

if [ $NUM_MACHINES -gt 1 ]; then
    SAME_NETWORK="false"
    NUM_PROCESSES=${WORLD_SIZE:-0}
else
    SAME_NETWORK="true"
    NUM_PROCESSES=${RUNAI_NUM_OF_GPUS:-0}
fi

config_file=$(cat <<EOF
compute_environment: LOCAL_MACHINE
debug: true
distributed_type: MULTI_GPU
downcast_bf16: 'no'
enable_cpu_affinity: false
gpu_ids: all
machine_rank: ${MACHINE_RANK}
main_process_ip: '${MAIN_PROCESS_IP}'
main_process_port: ${MAIN_PROCESS_PORT}
main_training_function: main
mixed_precision: 'no'
num_machines: ${NUM_MACHINES}
num_processes: ${NUM_PROCESSES}
rdzv_backend: static
same_network: ${SAME_NETWORK}
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF
)

echo "$config_file"