#!/usr/bin/env python3
"""Upload MIT-BIH WFDB files to AutoDL and point FT-CNN config at them."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import paramiko
import yaml

HOST = "connect.nmb2.seetacloud.com"
PORT = 28950
USER = "root"
PASSWORD = os.environ.get("AUTODL_PASSWORD", "")
LOCAL_DIR = Path(r"C:\Users\安\Desktop\project\项目\项目\autodl-tmp\数据集\mit_bih_dataset")
REMOTE_DIR = "/root/autodl-tmp/mit_bih_dataset"
# WFDB needs these; skip .xws and docs
KEEP_SUFFIXES = {".dat", ".hea", ".atr"}


def ensure_remote_dir(sftp: paramiko.SFTPClient, remote: str) -> None:
    parts = remote.strip("/").split("/")
    cur = ""
    for p in parts:
        cur += "/" + p
        try:
            sftp.stat(cur)
        except FileNotFoundError:
            sftp.mkdir(cur)


def main() -> int:
    if not PASSWORD:
        print("Set AUTODL_PASSWORD env var", file=sys.stderr)
        return 1
    if not LOCAL_DIR.is_dir():
        print(f"Missing local dataset: {LOCAL_DIR}", file=sys.stderr)
        return 1

    files = [
        f
        for f in LOCAL_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in KEEP_SUFFIXES
    ]
    total = sum(f.stat().st_size for f in files)
    print(f"Uploading {len(files)} files ({total/1024/1024:.1f} MB) -> {REMOTE_DIR}")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=30)
    sftp = client.open_sftp()
    ensure_remote_dir(sftp, REMOTE_DIR)

    for i, f in enumerate(sorted(files), 1):
        rpath = f"{REMOTE_DIR}/{f.name}"
        sftp.put(str(f), rpath)
        if i % 20 == 0 or i == len(files):
            print(f"  {i}/{len(files)} {f.name}")

    sftp.close()

    # Update remote FT-CNN config data_root
    cfg_path = "/root/autodl-tmp/FT-CNN/config/default.yaml"
    _, stdout, _ = client.exec_command(f"cat {cfg_path}")
    text = stdout.read().decode("utf-8")
    cfg = yaml.safe_load(text)
    cfg["paths"]["data_root"] = REMOTE_DIR
    new_text = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
    # write via sftp
    sftp = client.open_sftp()
    with sftp.file(cfg_path, "w") as rf:
        rf.write(new_text)
    sftp.close()

    # verify
    _, stdout, _ = client.exec_command(
        f"ls {REMOTE_DIR}/*.dat | wc -l; ls {REMOTE_DIR}/100.* ; "
        f"grep -A2 'data_root' {cfg_path}"
    )
    print(stdout.read().decode("utf-8", "replace"))
    client.close()
    print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
