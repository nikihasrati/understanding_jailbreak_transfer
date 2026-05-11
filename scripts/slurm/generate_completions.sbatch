#!/bin/bash
#SBATCH --job-name=generate_completions
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=40G
#SBATCH --output=slurm-%j_%a.out

PROFILE=${1:?profile required: general or preempt}
case "$PROFILE" in
  general|preempt) ;;
  *) echo "PROFILE must be 'general' or 'preempt'" >&2; exit 2 ;;
esac

MODEL_PATH=${2:?model path required}
MODEL_CACHE_ROOT=${3:?model cache root required}
FLAG_TYPE=${4:-multi-seed}
CHUNK_ID=${SLURM_ARRAY_TASK_ID}
NUM_CHUNKS=${SLURM_ARRAY_TASK_COUNT:-1}

export HF_HOME="$MODEL_CACHE_ROOT/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_P2P_DISABLE=1

python -m pipeline.generation.generate_completions   --model-path "$MODEL_PATH"   --chunk-id "$CHUNK_ID"   --num-chunks "$NUM_CHUNKS"   --resume   --"$FLAG_TYPE"
