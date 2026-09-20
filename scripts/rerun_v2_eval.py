#!/usr/bin/env python3
"""Upload fixed eval/cnn_lstm and re-run v2 evaluation on AutoDL."""

from __future__ import annotations

import os
import time

import paramiko

HOST = "connect.nmb2.seetacloud.com"
PORT = int(os.environ.get("AUTODL_PORT", "14616"))
PASSWORD = os.environ["AUTODL_PASSWORD"]
REMOTE = "/root/autodl-tmp/FT-CNN"
LOCAL = os.path.join(os.path.dirname(__file__), "..")


def main() -> None:
    from pathlib import Path

    local = Path(LOCAL).resolve()
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username="root", password=PASSWORD, timeout=30)
    sftp = c.open_sftp()
    for rel in ["src/eval.py", "src/models/cnn_lstm.py"]:
        sftp.put(str(local / rel), f"{REMOTE}/{rel}")
        print("uploaded", rel)
    sftp.close()

    cmd = r"""
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
cd /root/autodl-tmp/FT-CNN
export TF_CPP_MIN_LOG_LEVEL=2
export TF_FORCE_GPU_ALLOW_GROWTH=true
mkdir -p logs/05_cnn_lstm_v2_svdb artifacts/cnn_lstm_v2
python -m src.eval --config config/cnn_lstm_v2.yaml \
  --checkpoint artifacts/cnn_lstm_v2/checkpoints/best.h5 \
  --cache-dir cache_ctx5_svdb \
  --artifact-dir artifacts/cnn_lstm_v2 \
  2>&1 | tee logs/05_cnn_lstm_v2_svdb/cnn_lstm_v2_eval.log
echo EVAL_EXIT:$?
"""
    _, o, e = c.exec_command(cmd, timeout=600)
    print(o.read().decode("utf-8", "replace")[-8000:])
    err = e.read().decode("utf-8", "replace")
    if err.strip():
        print("ERR", err[-2000:])
    c.close()


if __name__ == "__main__":
    main()
