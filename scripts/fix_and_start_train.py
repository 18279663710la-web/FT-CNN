#!/usr/bin/env python3
import os
import sys
from pathlib import Path

import paramiko
import yaml

HOST = "connect.nmb2.seetacloud.com"
PORT = 28950
PASSWORD = os.environ["AUTODL_PASSWORD"]
REMOTE = "/root/autodl-tmp/FT-CNN"
DATA = "/root/autodl-tmp/mit_bih_dataset"
LOCAL = Path(r"C:\Users\安\Desktop\ECG\FT-CNN")


def sh(client, cmd, timeout=120):
    print(">>>", cmd[:120])
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    code = stdout.channel.recv_exit_status()
    text = (out + ("\n" + err if err.strip() else "")).encode("ascii", "replace").decode()
    print(text[-5000:] if len(text) > 5000 else text)
    return code, out


def main():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username="root", password=PASSWORD, timeout=30)

    # status
    sh(c, "ps aux | grep -E 'build_cache|src.train|run_pipeline' | grep -v grep || echo NO_TRAIN_PROC")
    sh(c, f"ls -la {REMOTE}/logs 2>/dev/null; tail -n 40 {REMOTE}/logs/pipeline.log 2>/dev/null || echo NO_PIPELINE_LOG")
    sh(c, f"grep -A1 data_root {REMOTE}/config/default.yaml || true")

    # ensure config + config.py + pipeline script
    sftp = c.open_sftp()
    sftp.put(str(LOCAL / "src/utils/config.py"), f"{REMOTE}/src/utils/config.py")
    with sftp.file(f"{REMOTE}/config/default.yaml", "r") as f:
        cfg = yaml.safe_load(f.read().decode("utf-8"))
    cfg["paths"]["data_root"] = DATA
    with sftp.file(f"{REMOTE}/config/default.yaml", "w") as f:
        f.write(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))

    pipeline = (
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        "source /root/miniconda3/etc/profile.d/conda.sh\n"
        "conda activate base\n"
        f"cd {REMOTE}\n"
        "export TF_CPP_MIN_LOG_LEVEL=2\n"
        "mkdir -p logs artifacts\n"
        'echo START_CACHE $(date) | tee logs/pipeline.log\n'
        "python scripts/build_cache.py 2>&1 | tee logs/build_cache.log\n"
        'echo START_TRAIN $(date) | tee -a logs/pipeline.log\n'
        "python -m src.train 2>&1 | tee logs/train.log\n"
        'echo START_EVAL $(date) | tee -a logs/pipeline.log\n'
        "python -m src.eval --checkpoint artifacts/checkpoints/best.h5 2>&1 | tee logs/eval.log\n"
        'echo ALL_DONE $(date) | tee -a logs/pipeline.log\n'
    )
    with sftp.file(f"{REMOTE}/scripts/run_pipeline.sh", "w") as f:
        f.write(pipeline)
    sftp.chmod(f"{REMOTE}/scripts/run_pipeline.sh", 0o755)
    sftp.close()

    # kill old, start new
    sh(
        c,
        "pkill -f build_cache.py || true; pkill -f 'src.train' || true; "
        "pkill -f 'src.eval' || true; pkill -f run_pipeline.sh || true; sleep 1",
    )
    sh(
        c,
        f"bash -lc 'source /root/miniconda3/etc/profile.d/conda.sh && conda activate base && "
        f"cd {REMOTE} && python -c \"from src.utils.config import load_config; "
        f"print(load_config(\\\"config/default.yaml\\\")[\\\"paths\\\"][\\\"data_root\\\"])\"'",
    )

    # start detached using nohup via absolute path
    start = (
        f"cd {REMOTE} && "
        f"nohup /bin/bash {REMOTE}/scripts/run_pipeline.sh "
        f"> {REMOTE}/logs/nohup.out 2>&1 & echo $!"
    )
    code, out = sh(c, start)
    pid = out.strip().splitlines()[-1] if out.strip() else "?"
    print("STARTED_PID", pid)

    import time
    time.sleep(8)
    sh(c, "ps aux | grep -E 'build_cache|run_pipeline|src.train' | grep -v grep || echo STILL_NO_PROC")
    sh(c, f"wc -l {REMOTE}/logs/nohup.out {REMOTE}/logs/pipeline.log {REMOTE}/logs/build_cache.log 2>/dev/null; "
       f"echo '---nohup---'; tail -n 50 {REMOTE}/logs/nohup.out 2>/dev/null; "
       f"echo '---pipeline---'; tail -n 30 {REMOTE}/logs/pipeline.log 2>/dev/null")
    c.close()


if __name__ == "__main__":
    main()
