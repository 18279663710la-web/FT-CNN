#!/usr/bin/env python3
"""Upload fixed training code and retrain on existing AutoDL cache."""

from __future__ import annotations

import os
import time
from pathlib import Path

import paramiko
import yaml

HOST = "connect.nmb2.seetacloud.com"
PORT = 28950
PASSWORD = os.environ["AUTODL_PASSWORD"]
REMOTE = "/root/autodl-tmp/FT-CNN"
DATA = "/root/autodl-tmp/mit_bih_dataset"
LOCAL = Path(r"C:\Users\安\Desktop\ECG\FT-CNN")


def run(client, cmd, timeout=60):
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    code = stdout.channel.recv_exit_status()
    text = (out + err).encode("ascii", "replace").decode()
    print(text[-3000:] if len(text) > 3000 else text)
    return code, out


def main():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username="root", password=PASSWORD, timeout=30)
    sftp = c.open_sftp()

    for rel in [
        "src/train.py",
        "src/eval.py",
        "src/models/ft_cnn.py",
        "src/models/losses.py",
        "src/utils/config.py",
    ]:
        sftp.put(str(LOCAL / rel), f"{REMOTE}/{rel}")
        print("put", rel)

    # Merge local train flags into remote config without losing data_root
    local_cfg = yaml.safe_load((LOCAL / "config/default.yaml").read_text(encoding="utf-8"))
    with sftp.file(f"{REMOTE}/config/default.yaml", "r") as f:
        remote_cfg = yaml.safe_load(f.read().decode("utf-8"))
    remote_cfg["paths"]["data_root"] = DATA
    remote_cfg["train"]["use_class_weight"] = False
    remote_cfg["train"]["focal_gamma"] = local_cfg["train"]["focal_gamma"]
    with sftp.file(f"{REMOTE}/config/default.yaml", "w") as f:
        f.write(yaml.safe_dump(remote_cfg, sort_keys=False, allow_unicode=True))
    print("config: use_class_weight=false, data_root=", DATA)

    pipeline = f"""#!/bin/bash
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
cd {REMOTE}
export TF_CPP_MIN_LOG_LEVEL=2
export TF_FORCE_GPU_ALLOW_GROWTH=true
mkdir -p logs artifacts
echo START_RETRAIN $(date) | tee logs/retrain_pipeline.log
ls -lh cache/*.npz
python -m src.train 2>&1 | tee logs/retrain.log
echo START_EVAL $(date) | tee -a logs/retrain_pipeline.log
python -m src.eval --checkpoint artifacts/checkpoints/best.h5 2>&1 | tee logs/retrain_eval.log
echo ALL_DONE $(date) | tee -a logs/retrain_pipeline.log
"""
    with sftp.file(f"{REMOTE}/scripts/run_retrain.sh", "w") as f:
        f.write(pipeline)
    sftp.chmod(f"{REMOTE}/scripts/run_retrain.sh", 0o755)
    sftp.close()

    run(c, "pkill -f 'src.train' || true; pkill -f 'src.eval' || true; pkill -f run_retrain.sh || true; sleep 1")

    transport = c.get_transport()
    chan = transport.open_session()
    chan.settimeout(20)
    chan.exec_command(
        f"setsid /bin/bash {REMOTE}/scripts/run_retrain.sh "
        f"> {REMOTE}/logs/retrain_nohup.out 2>&1 < /dev/null & echo $!"
    )
    buf = b""
    try:
        while True:
            if chan.recv_ready():
                buf += chan.recv(4096)
            if chan.exit_status_ready():
                while chan.recv_ready():
                    buf += chan.recv(4096)
                break
            time.sleep(0.1)
    except Exception as e:
        print("start note:", type(e).__name__, e)
    print("PID:", buf.decode("utf-8", "replace").strip())

    time.sleep(12)
    run(c, "ps -ef | grep -E 'src.train|run_retrain' | grep -v grep || echo NO_PROC")
    run(c, f"tail -n 40 {REMOTE}/logs/retrain.log 2>/dev/null || tail -n 40 {REMOTE}/logs/retrain_nohup.out")
    c.close()


if __name__ == "__main__":
    main()
