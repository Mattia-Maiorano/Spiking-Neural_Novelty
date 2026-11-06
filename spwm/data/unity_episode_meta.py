"""Deterministic port of Unity ProceduralObjectSpawner.GenerateDeterministicSpec (plane + mass only).

Uses the .NET System.Random (Knuth subtractive generator) so that the target height plane of an exported
trajectory can be recovered from its seed, since the trajectory JSON stores only the payload mass.
"""
from __future__ import annotations

import ctypes
from typing import List, Tuple

_MBIG = 2147483647
_MSEED = 161803398
PLANES = ("low", "mid", "high")
MASS_MIN, MASS_MAX = 1.20, 4.50
#: Weight classes split the [1.20, 4.50] kg range in three equal tiers (matches mass_tier_breakdown tiers).
LIGHT_MAX, NORMAL_MAX = 2.30, 3.40


class DotNetRandom:
    def __init__(self, seed: int) -> None:
        seed = ctypes.c_int32(seed).value
        sa: List[int] = [0] * 56
        subtraction = _MBIG if seed == -2147483648 else abs(seed)
        mj = _MSEED - subtraction
        sa[55] = mj
        mk = 1
        for i in range(1, 55):
            ii = (21 * i) % 55
            sa[ii] = mk
            mk = mj - mk
            if mk < 0:
                mk += _MBIG
            mj = sa[ii]
        for _ in range(1, 5):
            for i in range(1, 56):
                sa[i] -= sa[1 + (i + 30) % 55]
                if sa[i] < 0:
                    sa[i] += _MBIG
        self._sa, self._inext, self._inextp = sa, 0, 21

    def _internal(self) -> int:
        self._inext = 1 if self._inext + 1 >= 56 else self._inext + 1
        self._inextp = 1 if self._inextp + 1 >= 56 else self._inextp + 1
        r = self._sa[self._inext] - self._sa[self._inextp]
        if r == _MBIG:
            r -= 1
        if r < 0:
            r += _MBIG
        self._sa[self._inext] = r
        return r

    def next_double(self) -> float:
        return self._internal() * (1.0 / _MBIG)

    def next_int(self, n: int) -> int:
        return int(self.next_double() * n)


def episode_meta(seed: int) -> Tuple[str, float]:
    """Returns (height_plane, mass_kg) Unity derives for `seed` (shape, mass, plane draw order)."""
    rng = DotNetRandom(seed * 7919 + 104729)
    rng.next_int(3)  # shape
    mass = MASS_MIN + (MASS_MAX - MASS_MIN) * rng.next_double()
    return PLANES[rng.next_int(3)], mass


def weight_class(mass: float) -> str:
    return "light" if mass < LIGHT_MAX else ("normal" if mass < NORMAL_MAX else "heavy")
