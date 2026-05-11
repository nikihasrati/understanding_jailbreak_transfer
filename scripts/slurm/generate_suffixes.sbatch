#!/bin/bash
#SBATCH --job-name=generate_suffixes
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=50G
#SBATCH --output=slurm-%j_%a.out

MODEL_ID=${1:?model id required}
MODEL_CACHE_ROOT=${2:?model cache root required}
RESULTS_DIR=${3:-data/gcg_results/raw}
NUM_STEPS=${NUM_STEPS:-500}
END_SEED=${END_SEED:-99}
INDEX=${SLURM_ARRAY_TASK_ID}

export HF_HOME="$MODEL_CACHE_ROOT/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"

python -m pipeline suffixes run-gcg   --model-id "$MODEL_ID"   --index "$INDEX"   --num-steps "$NUM_STEPS"   --end-seed "$END_SEED"   --results-dir "$RESULTS_DIR"
