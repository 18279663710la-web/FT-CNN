"""MIT-BIH annotation symbols → AAMI EC57 five classes (paper Table 1)."""

from __future__ import annotations

from typing import Optional

CLASS_NAMES = ["N", "S", "V", "F", "Q"]

# WFDB beat symbols aligned with paper Table 1 / common AAMI practice.
SYMBOL_TO_AAMI: dict[str, int] = {
    # N
    "N": 0,
    "L": 0,
    "R": 0,
    "e": 0,  # atrial escape (AAMI normal family)
    # S
    "A": 1,
    "a": 1,
    "J": 1,
    "S": 1,
    "j": 1,  # nodal/junctional escape → S (paper)
    "x": 1,  # non-conducted P / blocked APB → S (paper)
    # V
    "V": 2,
    "E": 2,
    "!": 2,
    # F
    "F": 3,
    # Q
    "/": 4,
    "f": 4,
    "Q": 4,
}


def symbol_to_aami(symbol: str) -> Optional[int]:
    """Return AAMI class index, or None if the annotation is not a beat class."""
    if symbol is None:
        return None
    return SYMBOL_TO_AAMI.get(str(symbol))
