#!/bin/bash
#SBATCH --job-name=cross_model_generate
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=40G
#SBATCH --output=slurm-%j_%a.out

PROFILE=${1:?profile required: general or preempt}
case "$PROFILE" in
  general|preempt) ;;
  *) echo "PROFILE must be 'general' or 'preempt'" >&2; exit 2 ;;
esac

SOURCE_MODEL_PATH=${2:?source model path required}
TARGET_MODEL_PATH=${3:?target model path required}
MODEL_CACHE_ROOT=${4:?model cache root required}
CHUNK_ID=${SLURM_ARRAY_TASK_ID}

export HF_HOME="$MODEL_CACHE_ROOT/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_P2P_DISABLE=1

python -m pipeline.cross_model.generate_completions   --source_model_path "$SOURCE_MODEL_PATH"   --target_model_path "$TARGET_MODEL_PATH"   --chunk_id "$CHUNK_ID"
