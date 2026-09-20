#!/usr/bin/env python3
"""Probe new AutoDL instance for FT-CNN readiness."""
from __future__ import annotations

import os
import paramiko

HOST = os.environ.get("AUTODL_HOST", "connect.nmb2.seetacloud.com")
PORT = int(os.environ.get("AUTODL_PORT", "37448"))
PASSWORD = os.environ["AUTODL_PASSWORD"]

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username="root", password=PASSWORD, timeout=60)
_, o, e = c.exec_command(
    r"""
echo HOST=$(hostname); pwd; df -h /root | tail -1
echo ===
ls -la /root/autodl-tmp 2>/dev/null || echo NO_autodl_tmp
echo ===
ls -la /root/autodl-tmp/FT-CNN 2>/dev/null | head -30 || echo NO_FTCNN
echo ===
ls -la /root/autodl-tmp/mit_bih_dataset 2>/dev/null | head -5 || echo NO_MIT
echo ===
ls -la /root/autodl-tmp/svdb 2>/dev/null | head -5 || echo NO_SVDB
echo ===
ls -la /root/autodl-tmp/FT-CNN/cache_ctx5_svdb 2>/dev/null || echo NO_CACHE
echo ===
which python; python -c 'import tensorflow as tf; print("tf", tf.__version__)' 2>&1 | tail -5
nvidia-smi -L 2>&1 | head -3
""",
    timeout=60,
)
print(o.read().decode("utf-8", "replace"))
err = e.read().decode("utf-8", "replace")
if err.strip():
    print("ERR", err[:800])
c.close()
