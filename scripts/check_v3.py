#!/usr/bin/env python3
import os
import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(
    "connect.nmb2.seetacloud.com",
    port=14616,
    username="root",
    password=os.environ["AUTODL_PASSWORD"],
    timeout=60,
)
cmd = r"""
ls -la /root/autodl-tmp/FT-CNN/logs/06_cnn_lstm_v3_rr_branch/
echo ===
cat /root/autodl-tmp/FT-CNN/logs/06_cnn_lstm_v3_rr_branch/cnn_lstm_v3.pid 2>/dev/null || echo NO_PID
echo ===
tail -n 100 /root/autodl-tmp/FT-CNN/logs/06_cnn_lstm_v3_rr_branch/cnn_lstm_v3_nohup.out 2>/dev/null || echo NO_NOHUP
echo ===
ps aux | grep -E 'src.train|run_cnn_lstm_v3' | grep -v grep || echo NO_PROC
"""
_, o, e = c.exec_command(cmd, timeout=30)
print(o.read().decode("utf-8", "replace"))
err = e.read().decode("utf-8", "replace")
if err.strip():
    print("ERR", err[:500])
c.close()
