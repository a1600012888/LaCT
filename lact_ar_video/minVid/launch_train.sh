#!/usr/bin/bash

umask 007

#2. pluto configs (do not change)
export NUM_GPUS=${NUM_OF_GPUS}
export NUM_NODES=${WORLD_SIZE}
export NODE_RANK=${RANK}
#the following can be changed.
export GPU_PRE_FLIGHT_TEST=false
export SOFTWARE_FAILURE_AUTO_RECOVERY=false
export USE_PROFILER=false
export USE_LINEAGE=false
export EXP_MANAGER=false
export DISABLE_WANDB=true

if [ ! -z "$RUNAI_NUM_OF_GPUS" ]; then
  export NUM_GPUS=${RUNAI_NUM_OF_GPUS}
else
  export NUM_GPUS=${NUM_OF_GPUS}
fi
export NUM_NODES=${WORLD_SIZE}
export NODE_RANK=${RANK}
export NODE_LAUNCH_TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%S.%3N")+00:00 # same as python datetime.now(timezone.utc).isoformat()


# Decide single nodes or multiple nodes.
if [ -n "$MASTER_PORT" ]; then
  echo "Using run.ai distributed: $WORLD_SIZE, $RANK, $MASTER_ADDR, $MASTER_PORT"
  job_n=$WORLD_SIZE
  job_id=$RANK
  master_addr=$MASTER_ADDR
  master_port=$MASTER_PORT
  num_gpu=$RUNAI_NUM_OF_GPUS
else
  echo "Using runai localhost: 1, 0, localhost"
  job_n=1
  job_id=0
  master_addr=localhost
  master_port=12345
  if [ -n "$RUNAI_NUM_OF_GPUS" ]; then
    num_gpu=$RUNAI_NUM_OF_GPUS
  else
    num_gpu=8
  fi
fi


# No NCCL DEBUG info
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=eth0

# Run
torchrun \
--nproc_per_node=$num_gpu --nnodes=$job_n \
--rdzv-id=${JOB_UUID} \
--rdzv-backend=c10d \
--rdzv-endpoint=${master_addr}:${master_port} \
train.py ${@:1}
