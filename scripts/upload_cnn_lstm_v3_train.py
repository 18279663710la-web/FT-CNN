#!/usr/bin/env python3
"""Upload CNN-LSTM v3 (RR branch) and train on AutoDL using existing SVDB cache."""

from __future__ import annotations

import os
import time
from pathlib import Path

import paramiko
import yaml

HOST = "connect.nmb2.seetacloud.com"
PORT = int(os.environ.get("AUTODL_PORT", "14616"))
PASSWORD = os.environ.get("AUTODL_PASSWORD") or ""
if not PASSWORD:
    raise SystemExit("Set AUTODL_PASSWORD")
REMOTE = "/root/autodl-tmp/FT-CNN"
REMOTE_MIT = "/root/autodl-tmp/mit_bih_dataset"
REMOTE_SVDB = "/root/autodl-tmp/svdb"
LOCAL = Path(r"C:\Users\安\Desktop\ECG\FT-CNN")

CODE_FILES = [
    "src/models/cnn_lstm.py",
    "src/train.py",
    "src/eval.py",
    "config/cnn_lstm_v3.yaml",
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


def main() -> int:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username="root", password=PASSWORD, timeout=30)
    sftp = c.open_sftp()

    print("Uploading code...")
    for rel in CODE_FILES:
        lp = LOCAL / rel
        upload_file = str(lp)
        remote = f"{REMOTE}/{rel}"
        ensure_dir(sftp, str(Path(remote).as_posix().rsplit("/", 1)[0]))
        sftp.put(upload_file, remote)
        print(" ", rel)

    cfg = yaml.safe_load((LOCAL / "config/cnn_lstm_v3.yaml").read_text(encoding="utf-8"))
    cfg["paths"]["data_root"] = REMOTE_MIT
    cfg["paths"]["cache_dir"] = "cache_ctx5_svdb"
    cfg["paths"]["artifact_dir"] = "artifacts/cnn_lstm_v3"
    cfg["svdb"]["data_root"] = REMOTE_SVDB
    with sftp.file(f"{REMOTE}/config/cnn_lstm_v3.yaml", "w") as rf:
        rf.write(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    print("patched cnn_lstm_v3.yaml")

    pipeline = f"""#!/bin/bash
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
cd {REMOTE}
export TF_CPP_MIN_LOG_LEVEL=2
export TF_FORCE_GPU_ALLOW_GROWTH=true
mkdir -p logs/06_cnn_lstm_v3_rr_branch artifacts/cnn_lstm_v3/checkpoints

echo "===== TRAIN cnn_lstm_v3 (RR branch, reuse cache_ctx5_svdb) =====" | tee logs/06_cnn_lstm_v3_rr_branch/cnn_lstm_v3.log
test -f cache_ctx5_svdb/train_fit.npz || {{ echo "MISSING cache_ctx5_svdb/train_fit.npz"; exit 1; }}
python -m src.train --config config/cnn_lstm_v3.yaml \\
  --focal-gamma 0 --use-class-weight false \\
  2>&1 | tee logs/06_cnn_lstm_v3_rr_branch/cnn_lstm_v3_train.log

echo "===== EVAL =====" | tee -a logs/06_cnn_lstm_v3_rr_branch/cnn_lstm_v3.log
python -m src.eval --config config/cnn_lstm_v3.yaml \\
  --checkpoint artifacts/cnn_lstm_v3/checkpoints/best.h5 \\
  --cache-dir cache_ctx5_svdb --artifact-dir artifacts/cnn_lstm_v3 \\
  2>&1 | tee logs/06_cnn_lstm_v3_rr_branch/cnn_lstm_v3_eval.log

echo ALL_DONE $(date) | tee -a logs/06_cnn_lstm_v3_rr_branch/cnn_lstm_v3.log
"""
    ensure_dir(sftp, f"{REMOTE}/scripts")
    ensure_dir(sftp, f"{REMOTE}/logs/06_cnn_lstm_v3_rr_branch")
    with sftp.file(f"{REMOTE}/scripts/run_cnn_lstm_v3.sh", "w") as rf:
        rf.write(pipeline)
    sftp.chmod(f"{REMOTE}/scripts/run_cnn_lstm_v3.sh", 0o755)

    starter = (
        f"cd {REMOTE} && "
        f"(nohup bash scripts/run_cnn_lstm_v3.sh "
        f"> logs/06_cnn_lstm_v3_rr_branch/cnn_lstm_v3_nohup.out 2>&1 & "
        f"echo $! > logs/06_cnn_lstm_v3_rr_branch/cnn_lstm_v3.pid); "
        f"sleep 1; cat logs/06_cnn_lstm_v3_rr_branch/cnn_lstm_v3.pid"
    )
    _, o, e = c.exec_command(starter, timeout=30)
    print("pid", o.read().decode().strip())
    err = e.read().decode().strip()
    if err:
        print("starter_err", err)
    time.sleep(15)
    _, o, _ = c.exec_command(
        "ps aux | grep -E '[s]rc.train|[r]un_cnn_lstm_v3' || echo NO_PROC; "
        "echo ---; "
        "tail -n 80 /root/autodl-tmp/FT-CNN/logs/06_cnn_lstm_v3_rr_branch/cnn_lstm_v3_nohup.out 2>/dev/null || true"
    )
    print(o.read().decode())
    sftp.close()
    c.close()
    print("LAUNCHED_V3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
