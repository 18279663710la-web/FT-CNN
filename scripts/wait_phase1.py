#!/usr/bin/env python3
import json
import os
import time
from pathlib import Path

import paramiko

HOST = "connect.nmb2.seetacloud.com"
PORT = int(os.environ.get("AUTODL_PORT", "41049"))
PASSWORD = os.environ["AUTODL_PASSWORD"]
REMOTE = "/root/autodl-tmp/FT-CNN"
LOCAL = Path(r"C:\Users\安\Desktop\ECG\FT-CNN")


def main():
    for i in range(90):  # up to ~90 min
        c = paramiko.SSHClient()
        c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        c.connect(HOST, port=PORT, username="root", password=PASSWORD, timeout=30)
        _, o, _ = c.exec_command(
            "ps -ef | grep -E 'src.train|run_phase1|build_cache' | grep -v grep || echo NO_PROC; "
            f"tail -n 5 {REMOTE}/logs/phase1.log 2>/dev/null; "
            f"grep -E 'ALL_DONE|EXP |Accuracy:' {REMOTE}/logs/phase1.log {REMOTE}/logs/phase1_nohup.out 2>/dev/null | tail -n 30",
            timeout=30,
        )
        text = o.read().decode("utf-8", "replace")
        print(f"=== poll {i} ===")
        print(text[-1200:].encode("ascii", "replace").decode())
        if "ALL_DONE" in text:
            _, m, _ = c.exec_command(f"cat {REMOTE}/artifacts/phase1_summary.json", timeout=30)
            raw = m.read().decode("utf-8", "replace")
            (LOCAL / "artifacts" / "phase1_summary.json").write_text(raw, encoding="utf-8")
            for name in ["expA_ce", "expB_focal", "expC_ce_mild"]:
                _, mm, _ = c.exec_command(
                    f"cat {REMOTE}/artifacts/{name}/metrics.json", timeout=30
                )
                content = mm.read().decode("utf-8", "replace")
                d = LOCAL / "artifacts" / name
                d.mkdir(parents=True, exist_ok=True)
                (d / "metrics.json").write_text(content, encoding="utf-8")
            print("SUMMARY")
            print(raw)
            c.close()
            return
        c.close()
        time.sleep(60)
    print("TIMEOUT")


if __name__ == "__main__":
    main()
