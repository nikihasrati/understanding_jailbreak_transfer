#!/bin/bash
#SBATCH --job-name=cross_model_evaluate
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --mem=50G
#SBATCH --output=slurm-%j_%a.out

SOURCE_MODEL_PATH=${1:?source model path required}
TARGET_MODEL_PATH=${2:?target model path required}
MODEL_CACHE_ROOT=${3:?model cache root required}
CHUNK_ID=${SLURM_ARRAY_TASK_ID}
NUM_GPUS=${NUM_GPUS:-4}

export HF_HOME="$MODEL_CACHE_ROOT/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_P2P_DISABLE=1

python -m pipeline.cross_model.evaluate_completions   --source_model_path "$SOURCE_MODEL_PATH"   --target_model_path "$TARGET_MODEL_PATH"   --chunk_id "$CHUNK_ID"   --num_gpus "$NUM_GPUS"
