from __future__ import annotations
import abc
from dataclasses import dataclass
from enum import Enum
from typing import List, Protocol, Self

class ChordType(Enum):
    MAJOR = 0
    MINOR = 1
    DIMINISHED = 2
    AUGMENTED = 3
    MAJ_MAJ7 = 4
    MAJ_MIN7 = 5
    MIN_MAJ7 = 6
    MIN_MIN7 = 7
    DIM7 = 8
    HALF_DIM7 = 9
    AUG_MIN7 = 10
    AUG_MAJ7 = 11

    def __lt__(self, other: Numeral) -> bool:
        return self.value < other.value

    def __le__(self, other: Numeral) -> bool:
        return self.value <= other.value

class Numeral(Enum):
    I = 0
    bI = 1
    II = 2
    bII = 3
    III = 4
    sIII = 5
    bIII = 6
    IV = 7
    bIV = 8
    V = 9
    sV = 10
    bV = 11
    VI = 12
    bVI = 13
    VII = 14
    sVII = 15
    bVII = 16
    i = 17
    si = 18
    bi = 19
    ii = 20
    sii = 21
    bii = 22
    iii = 23
    siii = 24
    biii = 25
    iv = 26
    biv = 27
    siv = 28
    v = 29
    sv = 30
    bv = 31
    vi = 32
    svi = 33
    vii = 34
    svii = 35
    bvii = 36

    def __lt__(self, other: Numeral) -> bool:
        return self.value < other.value

    def __le__(self, other: Numeral) -> bool:
        return self.value <= other.value


class Inversion(Enum):
    root = 0
    first = 1
    second = 2
    third = 3

    def __lt__(self, other: Numeral) -> bool:
        return self.value < other.value

    def __le__(self, other: Numeral) -> bool:
        return self.value <= other.value



class Eq(Protocol):
    def eq_to(self, other: Self) -> bool:
        ...

@dataclass(order=True)
class MyToken(Eq):
    numeral: Numeral
    inversion: Inversion
    chord_type: ChordType

    def __hash__(self) -> int:
        return hash(repr(self))

    def __repr__(self):
        return f'MyToken({self.numeral.name} {self.inversion} {self.chord_type.name})'


    def eq_to(self, other: MyToken) -> bool:
        return self.numeral == other.numeral and self.inversion == other.inversion and self.chord_type == other.chord_type

    def get_all_numerals(self, ts: List[Eq]) -> List[Numeral]:
        return [t.numeral for t in ts]

    def get_all_inversions(self, ts: List[Eq]) -> List[Inversion]:
        return [t.inversion for t in ts]

