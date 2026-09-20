#!/usr/bin/env python3
"""Organize FT-CNN/logs into per-run folders on AutoDL."""

from __future__ import annotations

import os
import textwrap

import paramiko

HOST = "connect.nmb2.seetacloud.com"
PORT = int(os.environ.get("AUTODL_PORT", "14616"))
PASSWORD = os.environ["AUTODL_PASSWORD"]
LOGS = "/root/autodl-tmp/FT-CNN/logs"

SHELL = textwrap.dedent(
    r"""
    set -euo pipefail
    cd /root/autodl-tmp/FT-CNN/logs
    mkdir -p _misc \
      01_initial_pipeline \
      02_retrain_no_class_weight \
      03_phase1_abc \
      04_cnn_lstm_ctx5 \
      05_cnn_lstm_v2_svdb \
      06_cnn_lstm_v3_rr_branch

    move_one() {
      local f="$1" dest="$2"
      if [ -e "$f" ]; then
        mv -n "$f" "$dest/"
        echo "moved $f -> $dest/"
      elif [ -e "$dest/$f" ]; then
        echo "already $dest/$f"
      else
        echo "skip_missing $f"
      fi
    }

    # 01 initial
    move_one pipeline.log 01_initial_pipeline
    move_one build_cache.log 01_initial_pipeline
    move_one train.log 01_initial_pipeline
    move_one nohup.out 01_initial_pipeline
    move_one eval.log 01_initial_pipeline

    # 02 retrain
    move_one retrain_pipeline.log 02_retrain_no_class_weight
    move_one retrain.log 02_retrain_no_class_weight
    move_one retrain_nohup.out 02_retrain_no_class_weight
    move_one retrain_eval.log 02_retrain_no_class_weight

    # 03 phase1
    move_one phase1.log 03_phase1_abc
    move_one phase1_nohup.out 03_phase1_abc
    move_one cache_mild.log 03_phase1_abc
    move_one expA_train.log 03_phase1_abc
    move_one expA_eval.log 03_phase1_abc
    move_one expB_train.log 03_phase1_abc
    move_one expB_eval.log 03_phase1_abc
    move_one expC_train.log 03_phase1_abc
    move_one expC_eval.log 03_phase1_abc

    # 04 cnn-lstm v1
    move_one cnn_lstm.log 04_cnn_lstm_ctx5
    move_one cnn_lstm.pid 04_cnn_lstm_ctx5
    move_one cnn_lstm_train.log 04_cnn_lstm_ctx5
    move_one cnn_lstm_eval.log 04_cnn_lstm_ctx5
    move_one cnn_lstm_nohup.out 04_cnn_lstm_ctx5

    # 05 cnn-lstm v2 (may be active)
    move_one cnn_lstm_v2.log 05_cnn_lstm_v2_svdb
    move_one cnn_lstm_v2.pid 05_cnn_lstm_v2_svdb
    move_one cnn_lstm_v2_train.log 05_cnn_lstm_v2_svdb
    move_one cnn_lstm_v2_eval.log 05_cnn_lstm_v2_svdb
    move_one cnn_lstm_v2_nohup.out 05_cnn_lstm_v2_svdb
    move_one cache_ctx5_svdb.log 05_cnn_lstm_v2_svdb

    # 06 cnn-lstm v3 RR branch
    move_one cnn_lstm_v3.log 06_cnn_lstm_v3_rr_branch
    move_one cnn_lstm_v3.pid 06_cnn_lstm_v3_rr_branch
    move_one cnn_lstm_v3_train.log 06_cnn_lstm_v3_rr_branch
    move_one cnn_lstm_v3_eval.log 06_cnn_lstm_v3_rr_branch
    move_one cnn_lstm_v3_nohup.out 06_cnn_lstm_v3_rr_branch

    # leftovers
    for f in *; do
      if [ -f "$f" ] && [ "$f" != "README.md" ] && [ "$f" != "organize.sh" ]; then
        mv -n "$f" _misc/ && echo "leftover $f -> _misc/"
      fi
    done
    if [ -d .ipynb_checkpoints ]; then
      mv -n .ipynb_checkpoints _misc/ && echo "moved .ipynb_checkpoints"
    fi

    echo "---TREE---"
    find . -maxdepth 2 | sort
    """
).strip()

README = """# FT-CNN training logs (organized)

| Folder | Run | Rough date |
|--------|-----|------------|
| `01_initial_pipeline` | First cache build + FT-CNN train/eval | 2026-09-17 am |
| `02_retrain_no_class_weight` | Retrain without class weights | 2026-09-17 late morning |
| `03_phase1_abc` | Phase1 Exp A / B / C | 2026-09-17 afternoon |
| `04_cnn_lstm_ctx5` | CNN+LSTM 5-beat (morph aug S/V/F/Q→5k) | 2026-09-18 |
| `05_cnn_lstm_v2_svdb` | Attention + center skip + SVDB fill S/V | 2026-09-18 |
| `06_cnn_lstm_v3_rr_branch` | Dedicated center RR-MLP branch (design B) | 2026-09-18 |

`_misc/` holds leftovers / notebook checkpoints.
"""


def main() -> None:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username="root", password=PASSWORD, timeout=30)
    sftp = c.open_sftp()
    with sftp.file(f"{LOGS}/organize.sh", "w") as f:
        f.write(SHELL + "\n")
    with sftp.file(f"{LOGS}/README.md", "w") as f:
        f.write(README)
    sftp.chmod(f"{LOGS}/organize.sh", 0o755)
    sftp.close()

    _, o, e = c.exec_command(f"bash {LOGS}/organize.sh", timeout=60)
    print(o.read().decode("utf-8", "replace"))
    err = e.read().decode("utf-8", "replace")
    if err.strip():
        print("ERR", err[-2000:])

    _, o, _ = c.exec_command(
        f"echo '=== TOP ==='; ls -la {LOGS}; "
        f"for d in {LOGS}/0* {LOGS}/_misc; do "
        f"  echo; echo \"=== $(basename $d) ===\"; ls -lah \"$d\" 2>/dev/null; "
        f"done"
    )
    print(o.read().decode("utf-8", "replace"))
    c.close()
    print("LOGS_ORGANIZED")


if __name__ == "__main__":
    main()
