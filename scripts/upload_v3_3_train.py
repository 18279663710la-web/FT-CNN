#!/usr/bin/env python3
"""Upload v3.3 (prem+seq RR, S RR aug) and train on AutoDL."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import paramiko
import yaml

HOST = os.environ.get("AUTODL_HOST", "connect.nmb2.seetacloud.com")
PORT = int(os.environ.get("AUTODL_PORT", "37448"))
PASSWORD = os.environ["AUTODL_PASSWORD"]
REMOTE = "/root/autodl-tmp/FT-CNN"
LOCAL = Path(__file__).resolve().parents[1]
LOGDIR = "logs/11_cnn_lstm_v3_3"

FILES = [
    "src/models/cnn_lstm.py",
    "src/train.py",
    "src/eval.py",
    "src/data/rr_augment.py",
    "config/cnn_lstm_v3_3.yaml",
]


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

    for rel in FILES:
        ensure_dir(sftp, f"{REMOTE}/{Path(rel).parent.as_posix()}")
        sftp.put(str(LOCAL / rel), f"{REMOTE}/{rel}")
        print("up", rel)

    cfg = yaml.safe_load((LOCAL / "config/cnn_lstm_v3_3.yaml").read_text(encoding="utf-8"))
    cfg["paths"]["data_root"] = "/root/autodl-tmp/mit_bih_dataset"
    cfg["paths"]["cache_dir"] = "cache_ctx5_svdb"
    cfg["paths"]["artifact_dir"] = "artifacts/cnn_lstm_v3_3"
    cfg["svdb"]["data_root"] = "/root/autodl-tmp/svdb"
    cfg["train"]["use_class_weight"] = False
    with sftp.file(f"{REMOTE}/config/cnn_lstm_v3_3.yaml", "w") as rf:
        rf.write(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))

    ensure_dir(sftp, f"{REMOTE}/{LOGDIR}")
    ensure_dir(sftp, f"{REMOTE}/artifacts/cnn_lstm_v3_3/checkpoints")
    ensure_dir(sftp, f"{REMOTE}/scripts")

    sh = f"""#!/bin/bash
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
cd {REMOTE}
export TF_CPP_MIN_LOG_LEVEL=2
export TF_FORCE_GPU_ALLOW_GROWTH=true
LOGDIR={LOGDIR}
mkdir -p "$LOGDIR" artifacts/cnn_lstm_v3_3/checkpoints
echo "===== TRAIN v3.3 prem+seq + S RR aug =====" | tee "$LOGDIR/run.log"
test -f cache_ctx5_svdb/train_fit.npz
python -c "from src.models.cnn_lstm import build_cnn_lstm_from_config; from src.utils.config import load_config; m=build_cnn_lstm_from_config(load_config('config/cnn_lstm_v3_3.yaml')); print('lstm_in', int(m.get_layer('lstm').input.shape[-1])); print([l.name for l in m.layers if 'rr' in l.name or 'prem' in l.name or l.name=='fuse_step'])" | tee -a "$LOGDIR/run.log"
python -m src.train --config config/cnn_lstm_v3_3.yaml --focal-gamma 0 --use-class-weight false \\
  2>&1 | tee "$LOGDIR/train.log"
echo "===== EVAL =====" | tee -a "$LOGDIR/run.log"
python -m src.eval --config config/cnn_lstm_v3_3.yaml \\
  --checkpoint artifacts/cnn_lstm_v3_3/checkpoints/best.h5 \\
  --cache-dir cache_ctx5_svdb --artifact-dir artifacts/cnn_lstm_v3_3 \\
  2>&1 | tee "$LOGDIR/eval.log"
echo ALL_DONE $(date) | tee -a "$LOGDIR/run.log"
"""
    with sftp.file(f"{REMOTE}/scripts/run_v3_3.sh", "w") as rf:
        rf.write(sh)
    sftp.chmod(f"{REMOTE}/scripts/run_v3_3.sh", 0o755)

    launch = f"""#!/bin/bash
cd {REMOTE}
pkill -f 'run_v3_2_hard|run_v3_3|src.train.*v3_3|src.train.*hard' 2>/dev/null || true
sleep 1
nohup bash scripts/run_v3_3.sh > {LOGDIR}/nohup.out 2>&1 &
echo $! > {LOGDIR}/run.pid
echo STARTED:$(cat {LOGDIR}/run.pid)
"""
    with sftp.file(f"{REMOTE}/scripts/launch_v3_3.sh", "w") as rf:
        rf.write(launch)
    sftp.chmod(f"{REMOTE}/scripts/launch_v3_3.sh", 0o755)
    sftp.close()

    transport = c.get_transport()
    chan = transport.open_session()
    chan.get_pty()
    chan.exec_command("bash /root/autodl-tmp/FT-CNN/scripts/launch_v3_3.sh")
    time.sleep(3)
    out = b""
    while chan.recv_ready():
        out += chan.recv(4096)
    print(out.decode("utf-8", "replace"))
    chan.close()

    time.sleep(40)
    # pull snip of log
    sftp = c.open_sftp()
    local = LOCAL / "artifacts" / "v3_3_nohup_snip.txt"
    local.parent.mkdir(exist_ok=True)
    try:
        sftp.get(f"{REMOTE}/{LOGDIR}/nohup.out", str(local))
    except Exception as e:
        print("get log failed", e)
    sftp.close()
    _, o, _ = c.exec_command("pgrep -af src.train || echo NO_PROC")
    print("PROC", o.read().decode("ascii", "replace"))
    if local.exists():
        text = local.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            if any(
                k in line
                for k in (
                    "RR_AUGMENT_S",
                    "TRAIN_CFG",
                    "lstm_in",
                    "rr_prem",
                    "Epoch 1/",
                    "Traceback",
                    "Error",
                )
            ) and "ETA" not in line:
                print(line[:240])
    c.close()
    print("LAUNCHED_V3_3")


if __name__ == "__main__":
    main()
