#!/usr/bin/env python3
"""Upload hard-mine fine-tune (train-only) and launch on AutoDL."""

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
LOGDIR = "logs/10_cnn_lstm_v3_2_hard"

FILES = [
    "src/data/hard_mine.py",
    "src/train.py",
    "src/models/cnn_lstm.py",
    "src/eval.py",
    "config/cnn_lstm_v3_2_hard.yaml",
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

    cfg = yaml.safe_load(
        (LOCAL / "config/cnn_lstm_v3_2_hard.yaml").read_text(encoding="utf-8")
    )
    cfg["paths"]["data_root"] = "/root/autodl-tmp/mit_bih_dataset"
    cfg["paths"]["cache_dir"] = "cache_ctx5_svdb"
    cfg["paths"]["artifact_dir"] = "artifacts/cnn_lstm_v3_2_hard"
    cfg["svdb"]["data_root"] = "/root/autodl-tmp/svdb"
    cfg["train"]["use_class_weight"] = False
    cfg["train"]["hard_mine"]["init_checkpoint"] = (
        "artifacts/cnn_lstm_v3_2/checkpoints/best.h5"
    )
    with sftp.file(f"{REMOTE}/config/cnn_lstm_v3_2_hard.yaml", "w") as rf:
        rf.write(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))

    ensure_dir(sftp, f"{REMOTE}/{LOGDIR}")
    ensure_dir(sftp, f"{REMOTE}/artifacts/cnn_lstm_v3_2_hard/checkpoints")
    ensure_dir(sftp, f"{REMOTE}/scripts")

    sh = f"""#!/bin/bash
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
cd {REMOTE}
export TF_CPP_MIN_LOG_LEVEL=2
export TF_FORCE_GPU_ALLOW_GROWTH=true
LOGDIR={LOGDIR}
mkdir -p "$LOGDIR" artifacts/cnn_lstm_v3_2_hard/checkpoints
echo "===== HARD FT: mine S↔N on train_fit only =====" | tee "$LOGDIR/run.log"
test -f cache_ctx5_svdb/train_fit.npz
test -f artifacts/cnn_lstm_v3_2/checkpoints/best.h5
python -m src.train --config config/cnn_lstm_v3_2_hard.yaml --focal-gamma 0 --use-class-weight false \\
  2>&1 | tee "$LOGDIR/train.log"
echo "===== EVAL (test untouched during mining) =====" | tee -a "$LOGDIR/run.log"
python -m src.eval --config config/cnn_lstm_v3_2_hard.yaml \\
  --checkpoint artifacts/cnn_lstm_v3_2_hard/checkpoints/best.h5 \\
  --cache-dir cache_ctx5_svdb --artifact-dir artifacts/cnn_lstm_v3_2_hard \\
  2>&1 | tee "$LOGDIR/eval.log"
echo ALL_DONE $(date) | tee -a "$LOGDIR/run.log"
"""
    with sftp.file(f"{REMOTE}/scripts/run_v3_2_hard.sh", "w") as rf:
        rf.write(sh)
    sftp.chmod(f"{REMOTE}/scripts/run_v3_2_hard.sh", 0o755)

    launch = f"""#!/bin/bash
cd {REMOTE}
pkill -f 'run_v3_2|src.train.*v3_2' 2>/dev/null || true
sleep 1
nohup bash scripts/run_v3_2_hard.sh > {LOGDIR}/nohup.out 2>&1 &
echo $! > {LOGDIR}/run.pid
echo STARTED:$(cat {LOGDIR}/run.pid)
"""
    with sftp.file(f"{REMOTE}/scripts/launch_v3_2_hard.sh", "w") as rf:
        rf.write(launch)
    sftp.chmod(f"{REMOTE}/scripts/launch_v3_2_hard.sh", 0o755)
    sftp.close()

    transport = c.get_transport()
    chan = transport.open_session()
    chan.get_pty()
    chan.exec_command("bash /root/autodl-tmp/FT-CNN/scripts/launch_v3_2_hard.sh")
    time.sleep(3)
    out = b""
    while chan.recv_ready():
        out += chan.recv(4096)
    print(out.decode("utf-8", "replace"))
    chan.close()

    time.sleep(35)
    _, o, e = c.exec_command(
        f"cat {REMOTE}/{LOGDIR}/run.pid; echo ---; "
        f"pgrep -af 'src.train|run_v3_2_hard' || echo NO_PROC; echo ---; "
        f"grep -E 'HARD_MINE|Loaded init|after_oversample|Epoch' {REMOTE}/{LOGDIR}/nohup.out | head -n 40; "
        f"echo ---; tail -n 30 {REMOTE}/{LOGDIR}/nohup.out"
    )
    print(o.read().decode("utf-8", "replace"))
    err = e.read().decode("utf-8", "replace")
    if err.strip():
        print("ERR", err[:500])
    c.close()
    print("LAUNCHED_HARD_FT")


if __name__ == "__main__":
    main()
