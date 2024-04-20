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
    II = 1
    III = 2
    IV = 3
    V = 4
    VI = 5
    i = 6
    ii = 7
    iii = 8
    iv = 9
    v = 10
    vi = 11
    vii = 12
    VII = 13
    bVII = 14
    svii = 15
    biii = 16

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

