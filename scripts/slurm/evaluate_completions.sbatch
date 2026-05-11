#!/bin/bash
#SBATCH --job-name=evaluate_completions
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --mem=50G
#SBATCH --output=slurm-%j_%a.out

PROFILE=${1:?profile required: general or preempt}
case "$PROFILE" in
  general|preempt) ;;
  *) echo "PROFILE must be 'general' or 'preempt'" >&2; exit 2 ;;
esac

MODEL_PATH=${2:?model path required}
MODEL_CACHE_ROOT=${3:?model cache root required}
FLAG_TYPE=${4:-multi_seed}
CHUNK_ID=${SLURM_ARRAY_TASK_ID}
NUM_GPUS=${NUM_GPUS:-4}

export HF_HOME="$MODEL_CACHE_ROOT/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_P2P_DISABLE=1

python -m pipeline.evaluation.evaluate_completions   --model_path "$MODEL_PATH"   --"$FLAG_TYPE"   --chunk_id "$CHUNK_ID"   --num_gpus "$NUM_GPUS"
