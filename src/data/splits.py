from __future__ import annotations

from typing import Any


def get_splits(cfg: dict[str, Any]) -> dict[str, list[int]]:
    """Return patient-wise train_fit / val / test record IDs."""
    train = [int(x) for x in cfg["splits"]["train"]]
    test = [int(x) for x in cfg["splits"]["test"]]
    val = [int(x) for x in cfg["splits"]["val"]]
    val_set = set(val)
    missing = val_set - set(train)
    if missing:
        raise ValueError(f"val records not in train list: {sorted(missing)}")
    train_fit = [r for r in train if r not in val_set]
    return {"train_fit": train_fit, "val": val, "test": test}
