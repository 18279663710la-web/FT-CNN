#!/usr/bin/env python3
"""Upload CNN+LSTM code (+ optional cache) and start training on AutoDL."""

from __future__ import annotations

import os
import stat
import sys
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
DATA = "/root/autodl-tmp/mit_bih_dataset"
LOCAL = Path(r"C:\Users\安\Desktop\ECG\FT-CNN")

UPLOAD_GLOBS = [
    "src/**/*.py",
    "config/*.yaml",
    "scripts/build_cache.py",
    "requirements.txt",
    "REPRO_NOTES.md",
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


def upload_file(sftp: paramiko.SFTPClient, local: Path, remote: str) -> None:
    ensure_dir(sftp, str(Path(remote).parent).replace("\\", "/"))
    sftp.put(str(local), remote)


def sh(client: paramiko.SSHClient, cmd: str, timeout: int = 120) -> tuple[int, str]:
    print(">>>", cmd[:160])
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    code = stdout.channel.recv_exit_status()
    text = (out + ("\n" + err if err.strip() else "")).encode("ascii", "replace").decode()
    if text.strip():
        print(text[-4000:] if len(text) > 4000 else text)
    return code, out


def main() -> int:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username="root", password=PASSWORD, timeout=30)
    sftp = c.open_sftp()

    # Upload source
    files: list[Path] = []
    for pattern in UPLOAD_GLOBS:
        files.extend(LOCAL.glob(pattern))
    files = sorted({f.resolve() for f in files if f.is_file()})
    print(f"Uploading {len(files)} source files...")
    for f in files:
        rel = f.relative_to(LOCAL).as_posix()
        remote = f"{REMOTE}/{rel}"
        upload_file(sftp, f, remote)
        print(" ", rel)
    print("source upload done")

    # Fix cnn_lstm.yaml data_root on remote
    cfg_local = LOCAL / "config" / "cnn_lstm.yaml"
    cfg = yaml.safe_load(cfg_local.read_text(encoding="utf-8"))
    cfg["paths"]["data_root"] = DATA
    cfg["paths"]["cache_dir"] = "cache_ctx5"
    cfg["paths"]["artifact_dir"] = "artifacts/cnn_lstm_ctx5"
    remote_cfg = f"{REMOTE}/config/cnn_lstm.yaml"
    with sftp.file(remote_cfg, "w") as rf:
        rf.write(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    print("patched remote config/cnn_lstm.yaml data_root")

    # Upload cache_ctx5 if present (faster than rebuild)
    cache_local = LOCAL / "cache_ctx5"
    if cache_local.is_dir() and (cache_local / "train_fit.npz").exists():
        ensure_dir(sftp, f"{REMOTE}/cache_ctx5")
        for name in ["train_fit.npz", "val.npz", "test.npz", "meta.json"]:
            lp = cache_local / name
            if lp.exists():
                print(f"Uploading cache_ctx5/{name} ({lp.stat().st_size / 1e6:.1f} MB)...")
                sftp.put(str(lp), f"{REMOTE}/cache_ctx5/{name}")
        print("cache upload done")
        need_build = False
    else:
        need_build = True

    # Pipeline script
    pipeline = f"""#!/bin/bash
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
cd {REMOTE}
export TF_CPP_MIN_LOG_LEVEL=2
export TF_FORCE_GPU_ALLOW_GROWTH=true
mkdir -p logs artifacts/cnn_lstm_ctx5

echo "===== CNN-LSTM CTX5 TRAIN =====" | tee logs/cnn_lstm.log
"""
    if need_build:
        pipeline += f"""
python scripts/build_cache.py --config config/cnn_lstm.yaml --aug-classes 1,2,3,4 \\
  2>&1 | tee logs/cache_ctx5.log
"""
    pipeline += f"""
python -m src.train --config config/cnn_lstm.yaml \\
  --focal-gamma 0 --use-class-weight false \\
  2>&1 | tee logs/cnn_lstm_train.log

python -m src.eval --config config/cnn_lstm.yaml \\
  --checkpoint artifacts/cnn_lstm_ctx5/checkpoints/best.h5 \\
  --cache-dir cache_ctx5 --artifact-dir artifacts/cnn_lstm_ctx5 \\
  2>&1 | tee logs/cnn_lstm_eval.log

echo ALL_DONE $(date) | tee -a logs/cnn_lstm.log
"""
    remote_sh = f"{REMOTE}/scripts/run_cnn_lstm.sh"
    ensure_dir(sftp, f"{REMOTE}/scripts")
    with sftp.file(remote_sh, "w") as rf:
        rf.write(pipeline)
    sftp.chmod(remote_sh, 0o755)

    # Kill old train if any, start detached
    sh(c, "pkill -f 'src.train|run_cnn_lstm' || true", timeout=30)
    time.sleep(1)
    start = (
        f"cd {REMOTE} && nohup bash scripts/run_cnn_lstm.sh "
        f"> logs/cnn_lstm_nohup.out 2>&1 & echo PID:$!"
    )
    sh(c, start, timeout=30)
    time.sleep(3)
    sh(
        c,
        "ps aux | grep -E 'src.train|build_cache|run_cnn_lstm' | grep -v grep || echo NO_PROC; "
        "tail -n 30 /root/autodl-tmp/FT-CNN/logs/cnn_lstm_nohup.out 2>/dev/null || true",
        timeout=30,
    )

    sftp.close()
    c.close()
    print("LAUNCHED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
