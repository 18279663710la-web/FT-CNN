from src.data.aami_map import CLASS_NAMES, symbol_to_aami


def test_class_names_order():
    assert CLASS_NAMES == ["N", "S", "V", "F", "Q"]


def test_normal_family_maps_to_n():
    for sym in ["N", "L", "R", "e"]:
        assert symbol_to_aami(sym) == 0


def test_supraventricular_maps_to_s():
    for sym in ["A", "a", "J", "S", "j", "x"]:
        assert symbol_to_aami(sym) == 1


def test_ventricular_maps_to_v():
    for sym in ["V", "E", "!"]:
        assert symbol_to_aami(sym) == 2


def test_fusion_and_unknown():
    assert symbol_to_aami("F") == 3
    assert symbol_to_aami("/") == 4
    assert symbol_to_aami("f") == 4
    assert symbol_to_aami("Q") == 4


def test_unknown_symbol_returns_none():
    assert symbol_to_aami("~") is None
    assert symbol_to_aami("|") is None
