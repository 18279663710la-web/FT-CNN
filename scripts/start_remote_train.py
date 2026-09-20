#!/usr/bin/env python3
"""Start FT-CNN cache build + training on AutoDL in background."""

from __future__ import annotations

import os
import sys
import time

import paramiko

HOST = "connect.nmb2.seetacloud.com"
PORT = 28950
USER = "root"
PASSWORD = os.environ.get("AUTODL_PASSWORD", "")
REMOTE = "/root/autodl-tmp/FT-CNN"


def run(client, cmd, timeout=60):
    print("$", cmd[:200], "..." if len(cmd) > 200 else "")
    _, stdout, stderr = client.exec_command(cmd, get_pty=True, timeout=timeout)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    code = stdout.channel.recv_exit_status()
    text = (out + err).encode("ascii", "replace").decode("ascii")
    print(text[-4000:] if len(text) > 4000 else text)
    return code, out


def main() -> int:
    if not PASSWORD:
        print("Set AUTODL_PASSWORD", file=sys.stderr)
        return 1

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=30)
    print("Connected")

    # Sanity
    run(
        client,
        "bash -lc 'source /root/miniconda3/etc/profile.d/conda.sh && conda activate base && "
        f"ls {REMOTE}/config/default.yaml && ls /root/autodl-tmp/mit_bih_dataset/*.dat | wc -l && "
        "python -c \"import tensorflow as tf; print(tf.__version__); "
        "print(tf.config.list_physical_devices(\\\"GPU\\\"))\"'",
    )

    # Launch cache+train as one nohup job
    job = f"""bash -lc '
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
cd {REMOTE}
mkdir -p artifacts logs
export TF_CPP_MIN_LOG_LEVEL=2
nohup bash -c "
set -e
echo START_CACHE \\$(date)
python scripts/build_cache.py 2>&1 | tee logs/build_cache.log
echo START_TRAIN \\$(date)
python -m src.train 2>&1 | tee logs/train.log
echo START_EVAL \\$(date)
python -m src.eval --checkpoint artifacts/checkpoints/best.h5 2>&1 | tee logs/eval.log
echo ALL_DONE \\$(date)
" > logs/pipeline.log 2>&1 &
echo PIPELINE_PID=\\$!
sleep 2
head -n 30 logs/pipeline.log || true
tail -n 5 logs/build_cache.log 2>/dev/null || true
'
"""
    code, out = run(client, job, timeout=120)
    client.close()
    return 0 if code == 0 else code


if __name__ == "__main__":
    raise SystemExit(main())
