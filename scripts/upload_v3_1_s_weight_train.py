#!/usr/bin/env python3
"""Upload v3.1 + mild S weight and train on AutoDL."""

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

FILES = [
    "src/train.py",
    "src/models/cnn_lstm.py",
    "src/eval.py",
    "config/cnn_lstm_v3_1_s_weight.yaml",
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
        (LOCAL / "config/cnn_lstm_v3_1_s_weight.yaml").read_text(encoding="utf-8")
    )
    cfg["paths"]["data_root"] = "/root/autodl-tmp/mit_bih_dataset"
    cfg["paths"]["cache_dir"] = "cache_ctx5_svdb"
    cfg["paths"]["artifact_dir"] = "artifacts/cnn_lstm_v3_1_s_weight"
    cfg["svdb"]["data_root"] = "/root/autodl-tmp/svdb"
    with sftp.file(f"{REMOTE}/config/cnn_lstm_v3_1_s_weight.yaml", "w") as rf:
        rf.write(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))

    logdir = "logs/08_cnn_lstm_v3_1_s_weight"
    ensure_dir(sftp, f"{REMOTE}/{logdir}")
    ensure_dir(sftp, f"{REMOTE}/artifacts/cnn_lstm_v3_1_s_weight/checkpoints")
    ensure_dir(sftp, f"{REMOTE}/scripts")

    sh = f"""#!/bin/bash
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
cd {REMOTE}
export TF_CPP_MIN_LOG_LEVEL=2
export TF_FORCE_GPU_ALLOW_GROWTH=true
LOGDIR={logdir}
mkdir -p "$LOGDIR" artifacts/cnn_lstm_v3_1_s_weight/checkpoints
echo "===== TRAIN v3.1 + S weight 2.5 =====" | tee "$LOGDIR/run.log"
test -f cache_ctx5_svdb/train_fit.npz
# Do NOT pass --use-class-weight false; config enables manual weights
python -m src.train --config config/cnn_lstm_v3_1_s_weight.yaml --focal-gamma 0 \\
  2>&1 | tee "$LOGDIR/train.log"
echo "===== EVAL =====" | tee -a "$LOGDIR/run.log"
python -m src.eval --config config/cnn_lstm_v3_1_s_weight.yaml \\
  --checkpoint artifacts/cnn_lstm_v3_1_s_weight/checkpoints/best.h5 \\
  --cache-dir cache_ctx5_svdb --artifact-dir artifacts/cnn_lstm_v3_1_s_weight \\
  2>&1 | tee "$LOGDIR/eval.log"
echo ALL_DONE $(date) | tee -a "$LOGDIR/run.log"
"""
    with sftp.file(f"{REMOTE}/scripts/run_v3_1_s_weight.sh", "w") as rf:
        rf.write(sh)
    sftp.chmod(f"{REMOTE}/scripts/run_v3_1_s_weight.sh", 0o755)

    launch = f"""#!/bin/bash
cd {REMOTE}
# stop prior v3.1 job if still running
pkill -f 'run_cnn_lstm_v3_1|src.train.*cnn_lstm_v3_1' 2>/dev/null || true
sleep 1
nohup bash scripts/run_v3_1_s_weight.sh > {logdir}/nohup.out 2>&1 &
echo $! > {logdir}/run.pid
echo STARTED:$(cat {logdir}/run.pid)
"""
    with sftp.file(f"{REMOTE}/scripts/launch_v3_1_s_weight.sh", "w") as rf:
        rf.write(launch)
    sftp.chmod(f"{REMOTE}/scripts/launch_v3_1_s_weight.sh", 0o755)
    sftp.close()

    transport = c.get_transport()
    chan = transport.open_session()
    chan.get_pty()
    chan.exec_command("bash /root/autodl-tmp/FT-CNN/scripts/launch_v3_1_s_weight.sh")
    time.sleep(3)
    out = b""
    while chan.recv_ready():
        out += chan.recv(4096)
    print(out.decode("utf-8", "replace"))
    chan.close()

    time.sleep(20)
    _, o, e = c.exec_command(
        f"cat {REMOTE}/{logdir}/run.pid; echo ---; "
        f"pgrep -af 'src.train|run_v3_1_s_weight' || echo NO_PROC; echo ---; "
        f"tail -n 40 {REMOTE}/{logdir}/nohup.out"
    )
    print(o.read().decode("utf-8", "replace"))
    err = e.read().decode("utf-8", "replace")
    if err.strip():
        print("ERR", err[:500])
    c.close()
    print("LAUNCHED_S_WEIGHT")


if __name__ == "__main__":
    main()
