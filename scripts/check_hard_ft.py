#!/usr/bin/env python3
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(
    "connect.nmb2.seetacloud.com",
    port=int(os.environ.get("AUTODL_PORT", "37448")),
    username="root",
    password=os.environ["AUTODL_PASSWORD"],
    timeout=60,
)
remote_py = r"""
from pathlib import Path
p = Path('/root/autodl-tmp/FT-CNN/logs/10_cnn_lstm_v3_2_hard/nohup.out')
print('exists', p.exists())
if p.exists():
    text = p.read_text(encoding='utf-8', errors='replace')
    keys = ['HARD_MINE', 'Loaded init', 'after_oversample', 'TRAIN_CFG', 'Epoch', 'Accuracy', 'ALL_DONE', 'Error', 'Traceback']
    for line in text.splitlines():
        if any(k in line for k in keys) and 'ETA:' not in line:
            print(line[:240])
    print('---TAIL---')
    for line in text.splitlines()[-15:]:
        if 'ETA:' not in line:
            print(line[:240])
"""
# upload tiny helper then run
sftp = c.open_sftp()
with sftp.file("/tmp/peek_hard_log.py", "w") as f:
    f.write(remote_py)
sftp.close()
cmd = (
    "source /root/miniconda3/etc/profile.d/conda.sh && conda activate base && "
    "python /tmp/peek_hard_log.py; "
    "pgrep -af 'src.train.*hard' || true"
)
_, o, e = c.exec_command(cmd, timeout=60)
sys.stdout.write(o.read().decode("utf-8", "replace"))
err = e.read().decode("utf-8", "replace")
if err.strip():
    sys.stdout.write("STDERR:\n" + err[:500])
c.close()
