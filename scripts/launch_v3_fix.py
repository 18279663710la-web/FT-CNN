#!/usr/bin/env python3
"""Fix v3 launch: correct cache check + detach train."""
from __future__ import annotations

import os
import time
from pathlib import Path

import paramiko
import yaml

HOST = "connect.nmb2.seetacloud.com"
PORT = int(os.environ.get("AUTODL_PORT", "14616"))
PASSWORD = os.environ["AUTODL_PASSWORD"]
REMOTE = "/root/autodl-tmp/FT-CNN"
LOCAL = Path(__file__).resolve().parents[1]


def ensure_dir(sftp, remote: str) -> None:
    parts = remote.strip("/").split("/")
    cur = ""
    for p in parts:
        cur += "/" + p
        try:
            sftp.stat(cur)
        except FileNotFoundError:
            sftp.mkdir(cur)


def main() -> None:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username="root", password=PASSWORD, timeout=60)
    sftp = c.open_sftp()

    for rel in [
        "src/models/cnn_lstm.py",
        "src/train.py",
        "src/eval.py",
        "config/cnn_lstm_v3.yaml",
    ]:
        ensure_dir(sftp, f"{REMOTE}/{Path(rel).parent.as_posix()}")
        sftp.put(str(LOCAL / rel), f"{REMOTE}/{rel}")
        print("up", rel)

    cfg = yaml.safe_load((LOCAL / "config/cnn_lstm_v3.yaml").read_text(encoding="utf-8"))
    cfg["paths"]["data_root"] = "/root/autodl-tmp/mit_bih_dataset"
    cfg["paths"]["cache_dir"] = "cache_ctx5_svdb"
    cfg["paths"]["artifact_dir"] = "artifacts/cnn_lstm_v3"
    cfg["svdb"]["data_root"] = "/root/autodl-tmp/svdb"
    with sftp.file(f"{REMOTE}/config/cnn_lstm_v3.yaml", "w") as rf:
        rf.write(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))

    ensure_dir(sftp, f"{REMOTE}/logs/06_cnn_lstm_v3_rr_branch")
    ensure_dir(sftp, f"{REMOTE}/artifacts/cnn_lstm_v3/checkpoints")
    ensure_dir(sftp, f"{REMOTE}/scripts")

    sh = """#!/bin/bash
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
cd /root/autodl-tmp/FT-CNN
export TF_CPP_MIN_LOG_LEVEL=2
export TF_FORCE_GPU_ALLOW_GROWTH=true
LOGDIR=logs/06_cnn_lstm_v3_rr_branch
mkdir -p "$LOGDIR" artifacts/cnn_lstm_v3/checkpoints
echo "===== TRAIN cnn_lstm_v3 =====" | tee "$LOGDIR/cnn_lstm_v3.log"
ls -la cache_ctx5_svdb/ | tee -a "$LOGDIR/cnn_lstm_v3.log"
test -f cache_ctx5_svdb/train_fit.npz
python -m src.train --config config/cnn_lstm_v3.yaml --focal-gamma 0 --use-class-weight false \\
  2>&1 | tee "$LOGDIR/cnn_lstm_v3_train.log"
echo "===== EVAL =====" | tee -a "$LOGDIR/cnn_lstm_v3.log"
python -m src.eval --config config/cnn_lstm_v3.yaml \\
  --checkpoint artifacts/cnn_lstm_v3/checkpoints/best.h5 \\
  --cache-dir cache_ctx5_svdb --artifact-dir artifacts/cnn_lstm_v3 \\
  2>&1 | tee "$LOGDIR/cnn_lstm_v3_eval.log"
echo ALL_DONE $(date) | tee -a "$LOGDIR/cnn_lstm_v3.log"
"""
    with sftp.file(f"{REMOTE}/scripts/run_cnn_lstm_v3.sh", "w") as rf:
        rf.write(sh)
    sftp.chmod(f"{REMOTE}/scripts/run_cnn_lstm_v3.sh", 0o755)

    # Write a tiny launcher that backgrounds itself
    launch = """#!/bin/bash
cd /root/autodl-tmp/FT-CNN
nohup bash scripts/run_cnn_lstm_v3.sh > logs/06_cnn_lstm_v3_rr_branch/cnn_lstm_v3_nohup.out 2>&1 &
echo $! > logs/06_cnn_lstm_v3_rr_branch/cnn_lstm_v3.pid
echo STARTED:$(cat logs/06_cnn_lstm_v3_rr_branch/cnn_lstm_v3.pid)
"""
    with sftp.file(f"{REMOTE}/scripts/launch_v3.sh", "w") as rf:
        rf.write(launch)
    sftp.chmod(f"{REMOTE}/scripts/launch_v3.sh", 0o755)
    sftp.close()

    # Use get_pty so background survives SSH close better
    transport = c.get_transport()
    chan = transport.open_session()
    chan.get_pty()
    chan.exec_command("bash /root/autodl-tmp/FT-CNN/scripts/launch_v3.sh")
    time.sleep(3)
    out = b""
    while chan.recv_ready():
        out += chan.recv(4096)
    print(out.decode("utf-8", "replace"))
    chan.close()

    time.sleep(20)
    _, o, e = c.exec_command(
        "cat /root/autodl-tmp/FT-CNN/logs/06_cnn_lstm_v3_rr_branch/cnn_lstm_v3.pid; "
        "echo ---; ps -p $(cat /root/autodl-tmp/FT-CNN/logs/06_cnn_lstm_v3_rr_branch/cnn_lstm_v3.pid) -o pid,cmd || true; "
        "echo ---; pgrep -af 'src.train|run_cnn_lstm_v3' || echo NO_PROC; "
        "echo ---; tail -n 80 /root/autodl-tmp/FT-CNN/logs/06_cnn_lstm_v3_rr_branch/cnn_lstm_v3_nohup.out"
    )
    print(o.read().decode("utf-8", "replace"))
    err = e.read().decode("utf-8", "replace")
    if err.strip():
        print("ERR", err[:1000])
    c.close()


if __name__ == "__main__":
    main()
