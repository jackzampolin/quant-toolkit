#!/bin/bash
set -e

cd "$(dirname "$0")/.."
source .venv/bin/activate

python tools/build_glm52_nvfp4_routed_experts.py "$@"
