#!/usr/bin/env python3
"""Upload SVDB + CNN-LSTM v2 code, rebuild cache, start train on AutoDL."""

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
REMOTE_SVDB = "/root/autodl-tmp/svdb"
REMOTE_MIT = "/root/autodl-tmp/mit_bih_dataset"
LOCAL = Path(r"C:\Users\安\Desktop\ECG\FT-CNN")
LOCAL_SVDB = Path(
    r"C:\Users\安\Desktop\project\项目\项目\autodl-tmp\数据集\mit-bih-supraventricular-arrhythmia-database-1.0.0"
)

CODE_FILES = [
    "src/models/cnn_lstm.py",
    "src/data/svdb_loader.py",
    "src/data/context.py",
    "src/data/wfdb_loader.py",
    "src/data/augment.py",
    "src/data/preprocess.py",
    "src/data/aami_map.py",
    "src/data/splits.py",
    "src/train.py",
    "src/eval.py",
    "src/models/ft_cnn.py",
    "src/models/losses.py",
    "src/utils/config.py",
    "src/utils/seed.py",
    "scripts/build_cache.py",
    "config/cnn_lstm_v2.yaml",
    "config/cnn_lstm.yaml",
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


def upload_file(sftp, local: Path, remote: str) -> None:
    ensure_dir(sftp, str(Path(remote).as_posix().rsplit("/", 1)[0]))
    sftp.put(str(local), remote)


def main() -> int:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username="root", password=PASSWORD, timeout=30)
    sftp = c.open_sftp()

    print("Uploading code...")
    for rel in CODE_FILES:
        lp = LOCAL / rel
        if not lp.exists():
            print(" missing", rel)
            continue
        upload_file(sftp, lp, f"{REMOTE}/{rel}")
        print(" ", rel)

    # Patch remote config paths
    cfg = yaml.safe_load((LOCAL / "config/cnn_lstm_v2.yaml").read_text(encoding="utf-8"))
    cfg["paths"]["data_root"] = REMOTE_MIT
    cfg["paths"]["cache_dir"] = "cache_ctx5_svdb"
    cfg["paths"]["artifact_dir"] = "artifacts/cnn_lstm_v2"
    cfg["svdb"]["data_root"] = REMOTE_SVDB
    with sftp.file(f"{REMOTE}/config/cnn_lstm_v2.yaml", "w") as rf:
        rf.write(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    print("patched cnn_lstm_v2.yaml")

    # Upload SVDB essential files
    ensure_dir(sftp, REMOTE_SVDB)
    names = []
    if (LOCAL_SVDB / "RECORDS").exists():
        names.append("RECORDS")
    for p in sorted(LOCAL_SVDB.glob("*")):
        if p.suffix.lower() in {".dat", ".hea", ".atr"} or p.name == "RECORDS":
            names.append(p.name)
    names = sorted(set(names))
    print(f"Uploading SVDB ({len(names)} files)...")
    for i, name in enumerate(names, 1):
        lp = LOCAL_SVDB / name
        sftp.put(str(lp), f"{REMOTE_SVDB}/{name}")
        if i % 20 == 0 or i == len(names):
            print(f"  {i}/{len(names)}")

    pipeline = f"""#!/bin/bash
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
cd {REMOTE}
export TF_CPP_MIN_LOG_LEVEL=2
export TF_FORCE_GPU_ALLOW_GROWTH=true
mkdir -p logs artifacts/cnn_lstm_v2

echo "===== BUILD cache_ctx5_svdb =====" | tee logs/cnn_lstm_v2.log
python scripts/build_cache.py --config config/cnn_lstm_v2.yaml --skip-augment \\
  2>&1 | tee logs/cache_ctx5_svdb.log

echo "===== TRAIN cnn_lstm_v2 =====" | tee -a logs/cnn_lstm_v2.log
python -m src.train --config config/cnn_lstm_v2.yaml \\
  --focal-gamma 0 --use-class-weight false \\
  2>&1 | tee logs/cnn_lstm_v2_train.log

echo "===== EVAL =====" | tee -a logs/cnn_lstm_v2.log
python -m src.eval --config config/cnn_lstm_v2.yaml \\
  --checkpoint artifacts/cnn_lstm_v2/checkpoints/best.h5 \\
  --cache-dir cache_ctx5_svdb --artifact-dir artifacts/cnn_lstm_v2 \\
  2>&1 | tee logs/cnn_lstm_v2_eval.log

echo ALL_DONE $(date) | tee -a logs/cnn_lstm_v2.log
"""
    ensure_dir(sftp, f"{REMOTE}/scripts")
    with sftp.file(f"{REMOTE}/scripts/run_cnn_lstm_v2.sh", "w") as rf:
        rf.write(pipeline)
    sftp.chmod(f"{REMOTE}/scripts/run_cnn_lstm_v2.sh", 0o755)

    # start detached
    starter = (
        f"cd {REMOTE} && pkill -f 'run_cnn_lstm_v2|src.train.*cnn_lstm_v2' 2>/dev/null || true; "
        f"(nohup bash scripts/run_cnn_lstm_v2.sh > logs/cnn_lstm_v2_nohup.out 2>&1 & echo $! > logs/cnn_lstm_v2.pid); "
        f"cat logs/cnn_lstm_v2.pid"
    )
    transport = c.get_transport()
    chan = transport.open_session()
    chan.exec_command(starter)
    time.sleep(2)
    if chan.recv_ready():
        print("pid", chan.recv(64).decode().strip())
    chan.close()
    time.sleep(8)
    _, o, _ = c.exec_command(
        "ps aux | grep -E '[b]uild_cache|[s]rc.train|[r]un_cnn_lstm_v2' || echo NO_PROC; "
        "echo ---; tail -n 40 /root/autodl-tmp/FT-CNN/logs/cnn_lstm_v2_nohup.out 2>/dev/null || true"
    )
    print(o.read().decode())
    sftp.close()
    c.close()
    print("LAUNCHED_V2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
