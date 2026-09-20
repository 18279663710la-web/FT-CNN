#!/usr/bin/env python3
"""Upload latest code and run test-set evaluation on AutoDL."""

from __future__ import annotations

import json
import os
from pathlib import Path

import paramiko

HOST = "connect.nmb2.seetacloud.com"
PORT = 28950
PASSWORD = os.environ["AUTODL_PASSWORD"]
REMOTE = "/root/autodl-tmp/FT-CNN"
LOCAL = Path(r"C:\Users\安\Desktop\ECG\FT-CNN")


def run(client, cmd, timeout=600):
    print(">>>", cmd[:100])
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    code = stdout.channel.recv_exit_status()
    text = (out + err).encode("ascii", "replace").decode()
    print(text[-6000:] if len(text) > 6000 else text)
    return code, out


def main():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username="root", password=PASSWORD, timeout=30)
    sftp = c.open_sftp()
    for rel in [
        "src/utils/config.py",
        "src/train.py",
        "src/eval.py",
        "src/models/ft_cnn.py",
        "src/models/losses.py",
        "src/data/aami_map.py",
    ]:
        sftp.put(str(LOCAL / rel), f"{REMOTE}/{rel}")
        print("put", rel)
    sftp.close()

    cmd = (
        "bash -lc '"
        "source /root/miniconda3/etc/profile.d/conda.sh && conda activate base && "
        f"cd {REMOTE} && "
        "export TF_FORCE_GPU_ALLOW_GROWTH=true && "
        "export TF_CPP_MIN_LOG_LEVEL=2 && "
        "python -m src.eval --checkpoint artifacts/checkpoints/best.h5 "
        "2>&1 | tee logs/eval.log"
        "'"
    )
    code, out = run(c, cmd, timeout=600)
    _, metrics_out, _ = c.exec_command(f"cat {REMOTE}/artifacts/metrics.json", timeout=30)
    raw = metrics_out.read().decode("utf-8")
    Path(LOCAL / "artifacts").mkdir(exist_ok=True)
    (LOCAL / "artifacts" / "metrics.json").write_text(raw, encoding="utf-8")
    print("SAVED_LOCAL metrics.json")
    print(raw[:4000])
    c.close()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
