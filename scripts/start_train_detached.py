#!/usr/bin/env python3
"""Reliably start remote training and print status."""

import os
import time
from pathlib import Path

import paramiko

HOST = "connect.nmb2.seetacloud.com"
PORT = 28950
PASSWORD = os.environ["AUTODL_PASSWORD"]
REMOTE = "/root/autodl-tmp/FT-CNN"
LOCAL = Path(r"C:\Users\安\Desktop\ECG\FT-CNN")


def read(client, cmd, timeout=30):
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    code = stdout.channel.recv_exit_status()
    text = (out + err).encode("ascii", "replace").decode()
    return code, text


def main():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username="root", password=PASSWORD, timeout=30)
    sftp = c.open_sftp()
    sftp.put(str(LOCAL / "src/utils/config.py"), f"{REMOTE}/src/utils/config.py")

    pipeline = f"""#!/bin/bash
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
cd {REMOTE}
export TF_CPP_MIN_LOG_LEVEL=2
mkdir -p logs artifacts cache
echo START_CACHE $(date) | tee logs/pipeline.log
python scripts/build_cache.py 2>&1 | tee logs/build_cache.log
echo START_TRAIN $(date) | tee -a logs/pipeline.log
python -m src.train 2>&1 | tee logs/train.log
echo START_EVAL $(date) | tee -a logs/pipeline.log
python -m src.eval --checkpoint artifacts/checkpoints/best.h5 2>&1 | tee logs/eval.log
echo ALL_DONE $(date) | tee -a logs/pipeline.log
"""
    with sftp.file(f"{REMOTE}/scripts/run_pipeline.sh", "w") as f:
        f.write(pipeline)
    sftp.chmod(f"{REMOTE}/scripts/run_pipeline.sh", 0o755)
    sftp.close()

    # verify future import present
    code, text = read(c, f"head -n 5 {REMOTE}/src/utils/config.py")
    print("config.py head:\n", text)

    read(c, "pkill -f run_pipeline.sh || true; pkill -f build_cache.py || true; pkill -f 'src.train' || true; sleep 1")

    # Detach properly: setsid + redirect stdin
    start_cmd = (
        f"setsid /bin/bash {REMOTE}/scripts/run_pipeline.sh "
        f"> {REMOTE}/logs/nohup.out 2>&1 < /dev/null & echo $!"
    )
    # Use a short-lived channel; don't wait forever
    transport = c.get_transport()
    chan = transport.open_session()
    chan.settimeout(15)
    chan.exec_command(start_cmd)
    pid_out = b""
    try:
        while True:
            if chan.recv_ready():
                pid_out += chan.recv(4096)
            if chan.exit_status_ready():
                while chan.recv_ready():
                    pid_out += chan.recv(4096)
                break
            time.sleep(0.1)
    except Exception as e:
        print("start channel note:", type(e).__name__, e)
    print("start out:", pid_out.decode("utf-8", "replace").strip())

    time.sleep(10)
    code, text = read(c, "ps aux | grep -E 'run_pipeline|build_cache|src.train' | grep -v grep || echo NO_PROC")
    print("procs:\n", text)
    code, text = read(
        c,
        f"tail -n 60 {REMOTE}/logs/nohup.out; echo '===='; "
        f"tail -n 40 {REMOTE}/logs/pipeline.log; echo '===='; "
        f"tail -n 40 {REMOTE}/logs/build_cache.log 2>/dev/null || true",
    )
    print("logs:\n", text)
    c.close()


if __name__ == "__main__":
    main()
