#!/usr/bin/env python3
import os
import paramiko

PASSWORD = os.environ["AUTODL_PASSWORD"]
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("connect.nmb2.seetacloud.com", port=28950, username="root", password=PASSWORD, timeout=30)
cmd = r"""bash -lc '
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base
cd /root/autodl-tmp/FT-CNN
echo === config ===
python -c "from src.utils.config import load_config; print(load_config(\"config/default.yaml\")[\"paths\"][\"data_root\"])"
echo === processes ===
ps aux | grep -E "build_cache|src.train|run_pipeline|python" | grep -v grep | head -20
echo === logs ===
ls -la logs 2>/dev/null || echo no_logs
tail -n 40 logs/pipeline.log 2>/dev/null || echo no_pipeline
tail -n 40 logs/nohup.out 2>/dev/null || echo no_nohup
tail -n 20 logs/build_cache.log 2>/dev/null || echo no_cache_log
'
"""
_, stdout, stderr = c.exec_command(cmd, get_pty=True, timeout=60)
print(stdout.read().decode("utf-8", "replace").encode("ascii", "replace").decode())
print(stderr.read().decode("utf-8", "replace").encode("ascii", "replace").decode())
c.close()
