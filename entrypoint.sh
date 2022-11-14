#!/bin/bash
set -e
source "/opt/conda/etc/profile.d/conda.sh"
conda activate dreamdispersal
uvicorn scripts.api:app --host 0.0.0.0 --port 9090
