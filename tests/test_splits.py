from pathlib import Path

from src.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_splits_disjoint_and_complete():
    from src.data.splits import get_splits

    cfg = load_config(ROOT / "config" / "default.yaml")
    splits = get_splits(cfg)

    train_fit = set(splits["train_fit"])
    val = set(splits["val"])
    test = set(splits["test"])
    paper_train = set(cfg["splits"]["train"])
    paper_test = set(cfg["splits"]["test"])

    assert len(paper_train) == 22
    assert len(paper_test) == 22
    assert train_fit.isdisjoint(val)
    assert train_fit.isdisjoint(test)
    assert val.isdisjoint(test)
    assert train_fit | val == paper_train
    assert test == paper_test
    assert val == {223, 230}
