#!/bin/bash
#SBATCH --job-name=save_activations
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --mem=40G
#SBATCH --output=slurm-%j.out

PROFILE=${1:?profile required: general or preempt}
case "$PROFILE" in
  general|preempt) ;;
  *) echo "PROFILE must be 'general' or 'preempt'" >&2; exit 2 ;;
esac

MODEL_PATH=${2:?model path required}
MODEL_CACHE_ROOT=${3:?model cache root required}
INPUT_KIND=${4:-multi_seed_jailbreak}
NUM_CHUNKS=${NUM_CHUNKS:-32}
BATCH_SIZE=${BATCH_SIZE:-100}

export HF_HOME="$MODEL_CACHE_ROOT/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_P2P_DISABLE=1

python -m pipeline.activations.save_activations   --model-path "$MODEL_PATH"   --input-kind "$INPUT_KIND"   --num-chunks "$NUM_CHUNKS"   --batch-size "$BATCH_SIZE"
