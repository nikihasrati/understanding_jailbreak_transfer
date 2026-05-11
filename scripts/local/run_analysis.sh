#!/usr/bin/env bash
set -euo pipefail
python -m pipeline.analysis.multi_seed_data_analysis "$@"
