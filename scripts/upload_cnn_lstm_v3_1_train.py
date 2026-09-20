#!/usr/bin/env python3
"""Upload dual-RR v3.1 and start train+eval on AutoDL (reuse cache_ctx5_svdb)."""

from __future__ import annotations

import os
import time
from pathlib import Path

import paramiko
import yaml

HOST = os.environ.get("AUTODL_HOST", "connect.nmb2.seetacloud.com")
PORT = int(os.environ.get("AUTODL_PORT", "37448"))
PASSWORD = os.environ["AUTODL_PASSWORD"]
REMOTE = "/root/autodl-tmp/FT-CNN"
LOCAL = Path(__file__).resolve().parents[1]

CODE_FILES = [
    "src/models/cnn_lstm.py",
    "src/train.py",
    "src/eval.py",
    "src/models/losses.py",
    "src/utils/config.py",
    "src/utils/seed.py",
    "config/cnn_lstm_v3_1.yaml",
]


def ensure_dir(sftp: paramiko.SFTPClient, remote: str) -> None:
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

    for rel in CODE_FILES:
        lp = LOCAL / rel
        if not lp.exists():
            print("missing", rel)
            continue
        ensure_dir(sftp, f"{REMOTE}/{Path(rel).parent.as_posix()}")
        sftp.put(str(lp), f"{REMOTE}/{rel}")
        print("up", rel)

    cfg = yaml.safe_load((LOCAL / "config/cnn_lstm_v3_1.yaml").read_text(encoding="utf-8"))
    cfg["paths"]["data_root"] = "/root/autodl-tmp/mit_bih_dataset"
    cfg["paths"]["cache_dir"] = "cache_ctx5_svdb"
    cfg["paths"]["artifact_dir"] = "artifacts/cnn_lstm_v3_1"
    cfg["svdb"]["data_root"] = "/root/autodl-tmp/svdb"
    with sftp.file(f"{REMOTE}/config/cnn_lstm_v3_1.yaml", "w") as rf:
        rf.write(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    print("patched config")

    ensure_dir(sftp, f"{REMOTE}/logs/07_cnn_lstm_v3_1_dual_rr")
    ensure_dir(sftp, f"{REMOTE}/artifacts/cnn_lstm_v3_1/checkpoints")
    ensure_dir(sftp, f"{REMOTE}/scripts")

    sh = """#!/bin/bash
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
cd /root/autodl-tmp/FT-CNN
export TF_CPP_MIN_LOG_LEVEL=2
export TF_FORCE_GPU_ALLOW_GROWTH=true
LOGDIR=logs/07_cnn_lstm_v3_1_dual_rr
mkdir -p "$LOGDIR" artifacts/cnn_lstm_v3_1/checkpoints
echo "===== TRAIN cnn_lstm_v3_1 dual-RR =====" | tee "$LOGDIR/cnn_lstm_v3_1.log"
ls -la cache_ctx5_svdb/ | tee -a "$LOGDIR/cnn_lstm_v3_1.log"
test -f cache_ctx5_svdb/train_fit.npz
python -c "from src.models.cnn_lstm import build_cnn_lstm_from_config; from src.utils.config import load_config; m=build_cnn_lstm_from_config(load_config('config/cnn_lstm_v3_1.yaml')); print('lstm_in', int(m.get_layer('lstm').input.shape[-1])); print('has_rr_branch', any('rr_branch' in l.name for l in m.layers)); print('has_fuse_step', any(l.name=='fuse_step' for l in m.layers))" | tee -a "$LOGDIR/cnn_lstm_v3_1.log"
python -m src.train --config config/cnn_lstm_v3_1.yaml --focal-gamma 0 --use-class-weight false \\
  2>&1 | tee "$LOGDIR/cnn_lstm_v3_1_train.log"
echo "===== EVAL =====" | tee -a "$LOGDIR/cnn_lstm_v3_1.log"
python -m src.eval --config config/cnn_lstm_v3_1.yaml \\
  --checkpoint artifacts/cnn_lstm_v3_1/checkpoints/best.h5 \\
  --cache-dir cache_ctx5_svdb --artifact-dir artifacts/cnn_lstm_v3_1 \\
  2>&1 | tee "$LOGDIR/cnn_lstm_v3_1_eval.log"
echo ALL_DONE $(date) | tee -a "$LOGDIR/cnn_lstm_v3_1.log"
"""
    with sftp.file(f"{REMOTE}/scripts/run_cnn_lstm_v3_1.sh", "w") as rf:
        rf.write(sh)
    sftp.chmod(f"{REMOTE}/scripts/run_cnn_lstm_v3_1.sh", 0o755)

    launch = """#!/bin/bash
cd /root/autodl-tmp/FT-CNN
nohup bash scripts/run_cnn_lstm_v3_1.sh > logs/07_cnn_lstm_v3_1_dual_rr/cnn_lstm_v3_1_nohup.out 2>&1 &
echo $! > logs/07_cnn_lstm_v3_1_dual_rr/cnn_lstm_v3_1.pid
echo STARTED:$(cat logs/07_cnn_lstm_v3_1_dual_rr/cnn_lstm_v3_1.pid)
"""
    with sftp.file(f"{REMOTE}/scripts/launch_v3_1.sh", "w") as rf:
        rf.write(launch)
    sftp.chmod(f"{REMOTE}/scripts/launch_v3_1.sh", 0o755)
    sftp.close()

    transport = c.get_transport()
    chan = transport.open_session()
    chan.get_pty()
    chan.exec_command("bash /root/autodl-tmp/FT-CNN/scripts/launch_v3_1.sh")
    time.sleep(3)
    out = b""
    while chan.recv_ready():
        out += chan.recv(4096)
    print(out.decode("utf-8", "replace"))
    chan.close()

    time.sleep(25)
    _, o, e = c.exec_command(
        "source /root/miniconda3/etc/profile.d/conda.sh && conda activate base; "
        "cat /root/autodl-tmp/FT-CNN/logs/07_cnn_lstm_v3_1_dual_rr/cnn_lstm_v3_1.pid; echo ---; "
        "pgrep -af 'src.train|run_cnn_lstm_v3_1' || echo NO_PROC; echo ---; "
        "tail -n 60 /root/autodl-tmp/FT-CNN/logs/07_cnn_lstm_v3_1_dual_rr/cnn_lstm_v3_1_nohup.out"
    )
    print(o.read().decode("utf-8", "replace"))
    err = e.read().decode("utf-8", "replace")
    if err.strip():
        print("ERR", err[:800])
    c.close()
    print("LAUNCHED_V3_1")


if __name__ == "__main__":
    main()
