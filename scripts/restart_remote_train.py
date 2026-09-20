#!/usr/bin/env python3
"""Upload fixed files and restart FT-CNN training on AutoDL."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import paramiko

HOST = "connect.nmb2.seetacloud.com"
PORT = 28950
USER = "root"
PASSWORD = os.environ.get("AUTODL_PASSWORD", "")
REMOTE = "/root/autodl-tmp/FT-CNN"
LOCAL = Path(__file__).resolve().parents[1]


def main() -> int:
    if not PASSWORD:
        print("Set AUTODL_PASSWORD", file=sys.stderr)
        return 1

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=30)
    sftp = client.open_sftp()

    # Sync critical source files
    files = [
        "src/utils/config.py",
        "src/train.py",
        "src/eval.py",
        "src/models/ft_cnn.py",
        "src/models/losses.py",
        "src/data/wfdb_loader.py",
        "src/data/preprocess.py",
        "src/data/augment.py",
        "src/data/aami_map.py",
        "src/data/splits.py",
        "scripts/build_cache.py",
        # do NOT overwrite remote config/default.yaml (cloud data_root)
    ]
    for rel in files:
        local = LOCAL / rel
        remote = f"{REMOTE}/{rel}"
        print("put", rel)
        sftp.put(str(local), remote)
    sftp.close()

    # Kill old pipeline if any, then restart with pipefail
    cmd = f"""bash -lc '
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
cd {REMOTE}
pkill -f "python scripts/build_cache.py" 2>/dev/null || true
pkill -f "python -m src.train" 2>/dev/null || true
mkdir -p artifacts logs
# quick import check
python -c "from src.utils.config import load_config; print(load_config(\"config/default.yaml\")[\"paths\"][\"data_root\"])"
export TF_CPP_MIN_LOG_LEVEL=2
nohup bash -c "
set -euo pipefail
cd {REMOTE}
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
echo START_CACHE \\$(date) | tee logs/pipeline.log
python scripts/build_cache.py 2>&1 | tee -a logs/build_cache.log | tee -a logs/pipeline.log
echo START_TRAIN \\$(date) | tee -a logs/pipeline.log
python -m src.train 2>&1 | tee logs/train.log | tee -a logs/pipeline.log
echo START_EVAL \\$(date) | tee -a logs/pipeline.log
python -m src.eval --checkpoint artifacts/checkpoints/best.h5 2>&1 | tee logs/eval.log | tee -a logs/pipeline.log
echo ALL_DONE \\$(date) | tee -a logs/pipeline.log
" >/dev/null 2>&1 &
echo PIPELINE_PID=\\$!
sleep 3
tail -n 40 logs/pipeline.log || true
'
"""
    _, stdout, stderr = client.exec_command(cmd, get_pty=True, timeout=120)
    out = stdout.read().decode("utf-8", "replace")
    print(out.encode("ascii", "replace").decode("ascii"))
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
