from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CFG = ROOT / "config" / "default.yaml"


def test_load_config_split_lengths():
    from src.utils.config import load_config

    cfg = load_config(DEFAULT_CFG)
    assert len(cfg["splits"]["train"]) == 22
    assert len(cfg["splits"]["test"]) == 22
    assert cfg["splits"]["val"] == [223, 230]
    assert set(cfg["splits"]["val"]).issubset(set(cfg["splits"]["train"]))


def test_load_config_missing_file_raises():
    from src.utils.config import load_config

    with pytest.raises(FileNotFoundError):
        load_config(ROOT / "config" / "does_not_exist.yaml")
