#!/usr/bin/env bash
set -euo pipefail
python -m pipeline.suffix_generation.run_gcg "$@"
