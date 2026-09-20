#!/usr/bin/env python3
"""Upload FT-CNN to AutoDL and install deps."""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import paramiko

HOST = "connect.nmb2.seetacloud.com"
PORT = 28950
USER = "root"
PASSWORD = os.environ.get("AUTODL_PASSWORD")
if not PASSWORD:
    raise SystemExit("Set AUTODL_PASSWORD env var before running this script.")
REMOTE_DIR = "/root/autodl-tmp/FT-CNN"
LOCAL_DIR = Path(__file__).resolve().parents[1]

SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    "cache",
    "artifacts",
    ".superpowers",
}
SKIP_SUFFIXES = {".pyc", ".pyo"}


def should_skip(path: Path, root: Path) -> bool:
    rel = path.relative_to(root)
    parts = set(rel.parts)
    if parts & SKIP_DIRS:
        return True
    if path.suffix in SKIP_SUFFIXES:
        return True
    return False


def ensure_remote_dir(sftp: paramiko.SFTPClient, remote: str) -> None:
    parts = remote.strip("/").split("/")
    cur = ""
    for p in parts:
        cur += "/" + p
        try:
            sftp.stat(cur)
        except FileNotFoundError:
            sftp.mkdir(cur)


def upload_tree(sftp: paramiko.SFTPClient, local: Path, remote: str) -> int:
    ensure_remote_dir(sftp, remote)
    count = 0
    for path in local.rglob("*"):
        if should_skip(path, local):
            continue
        rel = path.relative_to(local).as_posix()
        rpath = f"{remote}/{rel}"
        if path.is_dir():
            ensure_remote_dir(sftp, rpath)
        else:
            ensure_remote_dir(sftp, str(Path(rpath).parent).replace("\\", "/"))
            sftp.put(str(path), rpath)
            count += 1
            if count % 20 == 0:
                print(f"  uploaded {count} files...")
    return count


def run(client: paramiko.SSHClient, cmd: str, timeout: int = 600) -> tuple[int, str, str]:
    print(f"$ {cmd}")
    stdin, stdout, stderr = client.exec_command(cmd, get_pty=True, timeout=timeout)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    code = stdout.channel.recv_exit_status()
    def _safe(s: str) -> str:
        return s.encode("utf-8", "replace").decode("ascii", "replace")
    if out:
        print(_safe(out))
    if err.strip():
        print(_safe(err))
    return code, out, err


def main() -> int:
    print(f"Connecting {HOST}:{PORT} ...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=30)
    print("Connected.")

    # Probe env
    run(
        client,
        "bash -lc 'source /root/miniconda3/etc/profile.d/conda.sh; "
        "conda activate base; which python; python -V; "
        "python -c \"import tensorflow as tf; print(tf.__version__)\"; "
        "ls -la /root/autodl-tmp'",
    )

    print(f"Uploading {LOCAL_DIR} -> {REMOTE_DIR}")
    sftp = client.open_sftp()
    n = upload_tree(sftp, LOCAL_DIR, REMOTE_DIR)
    sftp.close()
    print(f"Uploaded {n} files.")

    # Install deps without forcing tensorflow reinstall if already present
    install_cmd = (
        "bash -lc '"
        "source /root/miniconda3/etc/profile.d/conda.sh && "
        "conda activate base && "
        f"cd {REMOTE_DIR} && "
        "pip install -U pip && "
        "pip install \"wfdb>=4.1.0\" \"scipy>=1.7,<1.12\" \"numpy>=1.21,<1.25\" "
        "\"scikit-learn>=1.0,<1.4\" \"PyYAML>=6.0\" \"matplotlib>=3.5\" \"tqdm>=4.66\" && "
        "python -c \"import tensorflow as tf, wfdb, scipy, yaml; print(\\\"TF\\\", tf.__version__, \\\"OK\\\")\""
        "'"
    )
    code, _, _ = run(client, install_cmd, timeout=1200)
    client.close()
    if code != 0:
        print("Install failed with code", code)
        return code
    print("DONE: upload + deps OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
