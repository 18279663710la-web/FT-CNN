#!/usr/bin/env python3
"""Upload v3.2 (explicit prematurity) and train on AutoDL."""

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
LOGDIR = "logs/09_cnn_lstm_v3_2_prem"

FILES = [
    "src/models/cnn_lstm.py",
    "src/train.py",
    "src/eval.py",
    "config/cnn_lstm_v3_2.yaml",
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

    cfg = yaml.safe_load((LOCAL / "config/cnn_lstm_v3_2.yaml").read_text(encoding="utf-8"))
    cfg["paths"]["data_root"] = "/root/autodl-tmp/mit_bih_dataset"
    cfg["paths"]["cache_dir"] = "cache_ctx5_svdb"
    cfg["paths"]["artifact_dir"] = "artifacts/cnn_lstm_v3_2"
    cfg["svdb"]["data_root"] = "/root/autodl-tmp/svdb"
    cfg["train"]["use_class_weight"] = False
    with sftp.file(f"{REMOTE}/config/cnn_lstm_v3_2.yaml", "w") as rf:
        rf.write(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))

    ensure_dir(sftp, f"{REMOTE}/{LOGDIR}")
    ensure_dir(sftp, f"{REMOTE}/artifacts/cnn_lstm_v3_2/checkpoints")
    ensure_dir(sftp, f"{REMOTE}/scripts")

    sh = f"""#!/bin/bash
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
cd {REMOTE}
export TF_CPP_MIN_LOG_LEVEL=2
export TF_FORCE_GPU_ALLOW_GROWTH=true
LOGDIR={LOGDIR}
mkdir -p "$LOGDIR" artifacts/cnn_lstm_v3_2/checkpoints
echo "===== TRAIN v3.2 center_prem (no class weight) =====" | tee "$LOGDIR/run.log"
test -f cache_ctx5_svdb/train_fit.npz
python -c "from src.models.cnn_lstm import build_cnn_lstm_from_config; from src.utils.config import load_config; m=build_cnn_lstm_from_config(load_config('config/cnn_lstm_v3_2.yaml')); print('lstm_in', int(m.get_layer('lstm').input.shape[-1])); print('layers', [l.name for l in m.layers if 'rr' in l.name or 'prem' in l.name or l.name=='fuse_step'])" | tee -a "$LOGDIR/run.log"
python -m src.train --config config/cnn_lstm_v3_2.yaml --focal-gamma 0 --use-class-weight false \\
  2>&1 | tee "$LOGDIR/train.log"
echo "===== EVAL =====" | tee -a "$LOGDIR/run.log"
python -m src.eval --config config/cnn_lstm_v3_2.yaml \\
  --checkpoint artifacts/cnn_lstm_v3_2/checkpoints/best.h5 \\
  --cache-dir cache_ctx5_svdb --artifact-dir artifacts/cnn_lstm_v3_2 \\
  2>&1 | tee "$LOGDIR/eval.log"
echo ALL_DONE $(date) | tee -a "$LOGDIR/run.log"
"""
    with sftp.file(f"{REMOTE}/scripts/run_v3_2.sh", "w") as rf:
        rf.write(sh)
    sftp.chmod(f"{REMOTE}/scripts/run_v3_2.sh", 0o755)

    launch = f"""#!/bin/bash
cd {REMOTE}
pkill -f 'run_v3_1_s_weight|src.train.*s_weight|run_v3_2|src.train.*v3_2' 2>/dev/null || true
sleep 1
nohup bash scripts/run_v3_2.sh > {LOGDIR}/nohup.out 2>&1 &
echo $! > {LOGDIR}/run.pid
echo STARTED:$(cat {LOGDIR}/run.pid)
"""
    with sftp.file(f"{REMOTE}/scripts/launch_v3_2.sh", "w") as rf:
        rf.write(launch)
    sftp.chmod(f"{REMOTE}/scripts/launch_v3_2.sh", 0o755)
    sftp.close()

    transport = c.get_transport()
    chan = transport.open_session()
    chan.get_pty()
    chan.exec_command("bash /root/autodl-tmp/FT-CNN/scripts/launch_v3_2.sh")
    time.sleep(3)
    out = b""
    while chan.recv_ready():
        out += chan.recv(4096)
    print(out.decode("utf-8", "replace"))
    chan.close()

    time.sleep(22)
    _, o, e = c.exec_command(
        f"cat {REMOTE}/{LOGDIR}/run.pid; echo ---; "
        f"pgrep -af 'src.train|run_v3_2' || echo NO_PROC; echo ---; "
        f"tail -n 50 {REMOTE}/{LOGDIR}/nohup.out"
    )
    print(o.read().decode("utf-8", "replace"))
    err = e.read().decode("utf-8", "replace")
    if err.strip():
        print("ERR", err[:500])
    c.close()
    print("LAUNCHED_V3_2")


if __name__ == "__main__":
    main()
