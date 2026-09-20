#!/usr/bin/env python3
import os
import time

import paramiko

HOST = "connect.nmb2.seetacloud.com"
PORT = int(os.environ.get("AUTODL_PORT", "14616"))
PASSWORD = os.environ.get("AUTODL_PASSWORD") or ""
if not PASSWORD:
    raise SystemExit("Set AUTODL_PASSWORD")
REMOTE = "/root/autodl-tmp/FT-CNN"

PIPELINE = f"""#!/bin/bash
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
cd {REMOTE}
export TF_CPP_MIN_LOG_LEVEL=2
export TF_FORCE_GPU_ALLOW_GROWTH=true
mkdir -p logs artifacts/cnn_lstm_ctx5

echo "===== CNN-LSTM CTX5 TRAIN =====" | tee logs/cnn_lstm.log
python -m src.train --config config/cnn_lstm.yaml \\
  --focal-gamma 0 --use-class-weight false \\
  2>&1 | tee logs/cnn_lstm_train.log

python -m src.eval --config config/cnn_lstm.yaml \\
  --checkpoint artifacts/cnn_lstm_ctx5/checkpoints/best.h5 \\
  --cache-dir cache_ctx5 --artifact-dir artifacts/cnn_lstm_ctx5 \\
  2>&1 | tee logs/cnn_lstm_eval.log

echo ALL_DONE $(date) | tee -a logs/cnn_lstm.log
"""


def main() -> None:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username="root", password=PASSWORD, timeout=30)
    sftp = c.open_sftp()

    # rewrite pipeline
    path = f"{REMOTE}/scripts/run_cnn_lstm.sh"
    with sftp.file(path, "w") as f:
        f.write(PIPELINE)
    sftp.chmod(path, 0o755)

    # check cache
    _, o, _ = c.exec_command(f"ls -lh {REMOTE}/cache_ctx5; ls {REMOTE}/src/models/cnn_lstm.py")
    print(o.read().decode())

    # start with nohup via bash -c that exits
    starter = (
        f"cd {REMOTE} && "
        f"(nohup bash scripts/run_cnn_lstm.sh > logs/cnn_lstm_nohup.out 2>&1 & echo $! > logs/cnn_lstm.pid) "
        f"&& cat logs/cnn_lstm.pid"
    )
    transport = c.get_transport()
    chan = transport.open_session()
    chan.exec_command(starter)
    # don't wait forever
    time.sleep(2)
    if chan.recv_ready():
        print("pid_out", chan.recv(4096).decode())
    chan.close()

    time.sleep(5)
    _, o, _ = c.exec_command(
        "ps aux | grep -E '[s]rc.train|[r]un_cnn_lstm' || echo NO_PROC; "
        "echo ---; tail -n 40 /root/autodl-tmp/FT-CNN/logs/cnn_lstm_nohup.out 2>/dev/null || echo no_log; "
        "echo ---; cat /root/autodl-tmp/FT-CNN/logs/cnn_lstm.pid 2>/dev/null"
    )
    print(o.read().decode())
    sftp.close()
    c.close()


if __name__ == "__main__":
    main()
