#!/usr/bin/env python3
"""Phase 0: compare full DS2 test vs stratified ~10% subsample (paper Table3 scale)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import paramiko

HOST = "connect.nmb2.seetacloud.com"
PORT = int(os.environ.get("AUTODL_PORT", "41049"))
PASSWORD = os.environ["AUTODL_PASSWORD"]
REMOTE = "/root/autodl-tmp/FT-CNN"
LOCAL = Path(r"C:\Users\安\Desktop\ECG\FT-CNN")

DIAG_SCRIPT = r'''
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, accuracy_score, f1_score

tf.config.experimental.set_memory_growth(tf.config.list_physical_devices("GPU")[0], True)

ROOT = Path("/root/autodl-tmp/FT-CNN")
ckpt = ROOT / "artifacts/checkpoints/best.h5"
data = np.load(ROOT / "cache/test.npz")
beats = data["beats"][..., None].astype(np.float32)
labels = data["labels"].astype(np.int64)
rr = data["rr"].astype(np.float32)

model = tf.keras.models.load_model(ckpt, compile=False)
probs = model.predict([beats, rr], batch_size=64, verbose=0)
preds = probs.argmax(axis=1)

def summarize(y_true, y_pred, name):
    report = classification_report(
        y_true, y_pred, labels=list(range(5)),
        target_names=["N","S","V","F","Q"], output_dict=True, zero_division=0,
    )
    out = {
        "name": name,
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class": {k: {
            "precision": report[k]["precision"],
            "recall": report[k]["recall"],
            "f1": report[k]["f1-score"],
            "support": report[k]["support"],
        } for k in ["N","S","V","F","Q"]},
    }
    return out

full = summarize(labels, preds, "full_DS2")

# Stratified ~10% of beats to match paper Table3 scale (~11k)
rng = np.random.default_rng(42)
idx = []
for c in range(5):
    ids = np.where(labels == c)[0]
    n = max(1, int(round(0.1 * len(ids)))) if len(ids) else 0
    if n:
        idx.append(rng.choice(ids, size=min(n, len(ids)), replace=False))
idx = np.concatenate(idx) if idx else np.array([], dtype=int)
sub = summarize(labels[idx], preds[idx], "stratified_10pct_beats")

# Also match paper support counts approximately by class if possible
# Paper Table3: N9059 S804 V724 F278 Q80 = 10945
paper_support = {0: 9059, 1: 804, 2: 724, 3: 278, 4: 80}
idx2 = []
for c, n_want in paper_support.items():
    ids = np.where(labels == c)[0]
    n = min(n_want, len(ids))
    if n:
        idx2.append(rng.choice(ids, size=n, replace=False))
idx2 = np.concatenate(idx2)
paperish = summarize(labels[idx2], preds[idx2], "paper_table3_support_matched")

result = {
    "checkpoint": str(ckpt),
    "full_DS2": full,
    "stratified_10pct_beats": sub,
    "paper_table3_support_matched": paperish,
    "note": "Same model/predictions; only evaluation subset changes.",
}
out = ROOT / "artifacts/phase0_diag.json"
out.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
print("WROTE", out)
'''


def main():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username="root", password=PASSWORD, timeout=30)
    sftp = c.open_sftp()

    # sync critical files
    for rel in [
        "src/train.py",
        "src/eval.py",
        "src/models/ft_cnn.py",
        "src/models/losses.py",
        "src/utils/config.py",
        "config/default.yaml",
    ]:
        # keep remote data_root
        if rel.endswith("default.yaml"):
            import yaml
            local_cfg = yaml.safe_load((LOCAL / rel).read_text(encoding="utf-8"))
            with sftp.file(f"{REMOTE}/config/default.yaml", "r") as f:
                remote_cfg = yaml.safe_load(f.read().decode("utf-8"))
            remote_cfg["train"] = local_cfg["train"]
            remote_cfg["model"] = local_cfg["model"]
            remote_cfg["paths"]["data_root"] = "/root/autodl-tmp/mit_bih_dataset"
            with sftp.file(f"{REMOTE}/config/default.yaml", "w") as f:
                f.write(yaml.safe_dump(remote_cfg, sort_keys=False, allow_unicode=True))
            print("merged config")
            continue
        sftp.put(str(LOCAL / rel), f"{REMOTE}/{rel}")
        print("put", rel)

    with sftp.file(f"{REMOTE}/scripts/phase0_diag.py", "w") as f:
        f.write(DIAG_SCRIPT)
    sftp.close()

    cmd = (
        "bash -lc '"
        "source /root/miniconda3/etc/profile.d/conda.sh && conda activate base && "
        f"cd {REMOTE} && export TF_FORCE_GPU_ALLOW_GROWTH=true && "
        "export TF_CPP_MIN_LOG_LEVEL=2 && "
        "python scripts/phase0_diag.py"
        "'"
    )
    print(">>> running phase0")
    _, stdout, stderr = c.exec_command(cmd, timeout=600)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    print((out + err)[-8000:].encode("ascii", "replace").decode())

    _, m, _ = c.exec_command(f"cat {REMOTE}/artifacts/phase0_diag.json", timeout=30)
    raw = m.read().decode("utf-8")
    (LOCAL / "artifacts" / "phase0_diag.json").write_text(raw, encoding="utf-8")
    print("SAVED_LOCAL artifacts/phase0_diag.json")
    c.close()


if __name__ == "__main__":
    main()
