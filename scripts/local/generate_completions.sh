#!/usr/bin/env bash
set -euo pipefail
python -m pipeline.generation.generate_completions "$@"
