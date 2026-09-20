#!/usr/bin/env python3
"""Upload Phase1 code and launch A/B/C experiments on AutoDL."""

from __future__ import annotations

import os
import time
from pathlib import Path

import paramiko

HOST = "connect.nmb2.seetacloud.com"
PORT = int(os.environ.get("AUTODL_PORT", "41049"))
PASSWORD = os.environ["AUTODL_PASSWORD"]
REMOTE = "/root/autodl-tmp/FT-CNN"
LOCAL = Path(r"C:\Users\安\Desktop\ECG\FT-CNN")

PIPELINE = f"""#!/bin/bash
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
cd {REMOTE}
export TF_CPP_MIN_LOG_LEVEL=2
export TF_FORCE_GPU_ALLOW_GROWTH=true
mkdir -p logs artifacts

echo "===== EXP A: CE gamma=0, balanced cache =====" | tee logs/phase1.log
python -m src.train --focal-gamma 0 --use-class-weight false \\
  --artifact-dir artifacts/expA_ce \\
  --cache-dir cache 2>&1 | tee logs/expA_train.log
python -m src.eval --checkpoint artifacts/expA_ce/checkpoints/best.h5 \\
  --cache-dir cache --artifact-dir artifacts/expA_ce 2>&1 | tee logs/expA_eval.log

echo "===== EXP B: Focal gamma=2, balanced cache =====" | tee -a logs/phase1.log
python -m src.train --focal-gamma 2 --use-class-weight false \\
  --artifact-dir artifacts/expB_focal \\
  --cache-dir cache 2>&1 | tee logs/expB_train.log
python -m src.eval --checkpoint artifacts/expB_focal/checkpoints/best.h5 \\
  --cache-dir cache --artifact-dir artifacts/expB_focal 2>&1 | tee logs/expB_eval.log

echo "===== BUILD cache_mild: aug S/V/F/Q only, target 5000 =====" | tee -a logs/phase1.log
python scripts/build_cache.py --cache-dir cache_mild --target-per-class 5000 --aug-classes 1,2,3,4 \\
  2>&1 | tee logs/cache_mild.log

echo "===== EXP C: CE + mild aug =====" | tee -a logs/phase1.log
python -m src.train --focal-gamma 0 --use-class-weight false \\
  --artifact-dir artifacts/expC_ce_mild \\
  --cache-dir cache_mild 2>&1 | tee logs/expC_train.log
# Always evaluate on FULL original DS2 test from main cache
python -m src.eval --checkpoint artifacts/expC_ce_mild/checkpoints/best.h5 \\
  --cache-dir cache --artifact-dir artifacts/expC_ce_mild 2>&1 | tee logs/expC_eval.log

python - <<'PY'
import json
from pathlib import Path
root = Path("{REMOTE}")
rows = []
for name in ["expA_ce", "expB_focal", "expC_ce_mild"]:
    p = root / "artifacts" / name / "metrics.json"
    m = json.loads(p.read_text())
    r = m["classification_report"]
    rows.append({{
        "exp": name,
        "acc": m["accuracy"],
        "macro_f1": r["macro avg"]["f1-score"],
        "N_f1": r["N"]["f1-score"],
        "S_f1": r["S"]["f1-score"],
        "V_f1": r["V"]["f1-score"],
        "F_f1": r["F"]["f1-score"],
        "Q_f1": r["Q"]["f1-score"],
    }})
summary = {{"paper_acc": 0.9851, "runs": rows}}
out = root / "artifacts" / "phase1_summary.json"
out.write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
print("WROTE", out)
PY

echo ALL_DONE $(date) | tee -a logs/phase1.log
"""


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
        "scripts/build_cache.py",
    ]:
        sftp.put(str(LOCAL / rel), f"{REMOTE}/{rel}")
        print("put", rel)
    with sftp.file(f"{REMOTE}/scripts/run_phase1.sh", "w") as f:
        f.write(PIPELINE)
    sftp.chmod(f"{REMOTE}/scripts/run_phase1.sh", 0o755)
    sftp.close()

    # kill old jobs
    _, o, _ = c.exec_command(
        "pkill -f 'src.train' || true; pkill -f run_phase1.sh || true; pkill -f build_cache.py || true; sleep 1",
        timeout=30,
    )
    o.channel.recv_exit_status()

    transport = c.get_transport()
    chan = transport.open_session()
    chan.settimeout(20)
    chan.exec_command(
        f"setsid /bin/bash {REMOTE}/scripts/run_phase1.sh "
        f"> {REMOTE}/logs/phase1_nohup.out 2>&1 < /dev/null & echo $!"
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

    time.sleep(15)
    _, o, _ = c.exec_command(
        "ps -ef | grep -E 'src.train|run_phase1|build_cache' | grep -v grep || echo NO_PROC; "
        f"tail -n 30 {REMOTE}/logs/phase1_nohup.out",
        timeout=30,
    )
    print(o.read().decode("utf-8", "replace")[-2500:].encode("ascii", "replace").decode())
    c.close()


if __name__ == "__main__":
    main()
