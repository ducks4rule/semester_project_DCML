from typing import List
from data_classes import Eq, Numeral, Inversion, MyToken, ChordType
import numpy as np



def duplicated_head(xs: List[Eq]) -> bool:
    return xs[0].eq_to(xs[1])

def markov_transition_matrix(xs: List[Eq]) -> np.ndarray: 
    if duplicated_head(xs):
        xs.pop(0)
    new_xs = [x.numeral for x in xs]
    unique_tokens = list(set(new_xs))
    unique_tokens.sort()
    n = len(unique_tokens)
    matrix = np.zeros((n, n))
    for i in range(len(new_xs) - 1):
        cur = unique_tokens.index(new_xs[i])
        nxt = unique_tokens.index(new_xs[i + 1])
        matrix[cur][nxt] += 1
    for i in range(n):
        matrix[i] /= sum(matrix[i])
    return matrix

def get_chord_type_simple(x: MyToken) -> ChordType:
    if x.value < 6 or x.value in [14, 16]:
        return ChordType.MAJOR
    elif x.value < 12:
        return ChordType.MINOR
    elif x.value in [12, 13, 15]:
        return ChordType.DIMINISHED
    else:
        raise ValueError("Invalid numeral value")

def translate_figbass(x: str) -> Inversion:
    if x == '0' or x == '7':
        return Inversion.root
    elif x == '6' or x == '65':
        return Inversion.first
    elif x == '64' or x == '43':
        return Inversion.second
    elif x == '2':
        return Inversion.third
    else:
        raise ValueError("Invalid inversion value")

def translate_numeral(x: str) -> Numeral:
    if x == 'I':
        return Numeral.I
    elif x == 'II':
        return Numeral.II
    elif x == 'III':
        return Numeral.III
    elif x == 'IV':
        return Numeral.IV
    elif x == 'V':
        return Numeral.V
    elif x == 'VI':
        return Numeral.VI
    elif x == 'i':
        return Numeral.i
    elif x == 'ii':
        return Numeral.ii
    elif x == 'iii':
        return Numeral.iii
    elif x == 'biii':
        return Numeral.biii
    elif x == 'iv':
        return Numeral.iv
    elif x == 'v':
        return Numeral.v
    elif x == 'vi':
        return Numeral.vi
    elif x == 'vii':
        return Numeral.vii
    elif x == 'VII':
        return Numeral.VII
    elif x == 'bVII':
        return Numeral.bVII
    elif x == '#vii':
        return Numeral.svii
    else:
        raise ValueError("Invalid numeral value")
        

def translate_chord_type(x : Numeral, fb: str) -> ChordType:
    chord_type = get_chord_type_simple(x)
    # if fb in ['2', '6', '65', '43']:
    #     if chord_type == ChordType.MAJOR:
    #         return ChordType.MAJ_MIN7
    # if x.value in [12,15] and fb in ['2', '6', '65', '43']:
    #     return ChordType.DIM7
        
    ## TODO: finish this function
    return chord_type
            
