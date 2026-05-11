#!/bin/bash
#SBATCH --job-name=cross_model_generate
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=40G
#SBATCH --output=slurm-%j_%a.out

SOURCE_MODEL_PATH=${1:?source model path required}
TARGET_MODEL_PATH=${2:?target model path required}
MODEL_CACHE_ROOT=${3:?model cache root required}
CHUNK_ID=${SLURM_ARRAY_TASK_ID}

export HF_HOME="$MODEL_CACHE_ROOT/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_P2P_DISABLE=1

python -m pipeline cross-model generate   --source-model-path "$SOURCE_MODEL_PATH"   --target-model-path "$TARGET_MODEL_PATH"   --chunk-id "$CHUNK_ID"
