from typing import List
from data_classes import Eq, Numeral, Inversion, MyToken, ChordType
import numpy as np
import pickle



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
    if x == 'bI':
        return Numeral.bI
    elif x == 'II':
        return Numeral.II
    elif x == 'bII':
        return Numeral.bII
    elif x == 'III':
        return Numeral.III
    elif x == '#III':
        return Numeral.sIII
    elif x == 'bIII':
        return Numeral.bIII
    elif x == 'IV':
        return Numeral.IV
    elif x == 'bIV':
        return Numeral.bIV
    elif x == 'V':
        return Numeral.V
    elif x == '#V':
        return Numeral.sV
    elif x == 'bV':
        return Numeral.bV
    elif x == 'VI':
        return Numeral.VI
    elif x == 'bVI':
        return Numeral.bVI
    elif x == 'VII':
        return Numeral.VII
    elif x == 'bVII':
        return Numeral.bVII
    elif x == '#VII':
        return Numeral.sVII
    elif x == 'i':
        return Numeral.i
    elif x == '#i':
        return Numeral.si
    elif x == 'bi':
        return Numeral.bi
    elif x == 'ii':
        return Numeral.ii
    elif x == '#ii':
        return Numeral.sii
    elif x == 'bii':
        return Numeral.bii
    elif x == 'iii':
        return Numeral.iii
    elif x == '#iii':
        return Numeral.siii
    elif x == 'biii':
        return Numeral.biii
    elif x == 'iv':
        return Numeral.iv
    elif x == '#iv':
        return Numeral.siv
    elif x == 'biv':
        return Numeral.biv
    elif x == 'v':
        return Numeral.v
    elif x == '#v':
        return Numeral.sv
    elif x == 'bv':
        return Numeral.bv
    elif x == 'vi':
        return Numeral.vi
    elif x == '#vi':
        return Numeral.svi
    elif x == 'vii':
        return Numeral.vii
    elif x == '#vii':
        return Numeral.svii
    elif x == 'bvii':
        return Numeral.bvii
    else:
        raise ValueError("Invalid numeral value")
        

def translate_chord_type(ct: str) -> ChordType:
    # TODO: is the +7 correct?
    if ct == 'M':
        chord_type = ChordType.MAJOR
    elif ct == 'm':
        chord_type = ChordType.MINOR
    elif ct == 'o':
        chord_type = ChordType.DIMINISHED
    elif ct == '%7':
        chord_type = ChordType.HALF_DIM7
    elif ct == 'o7':
        chord_type = ChordType.DIM7
    elif ct == 'Mm7':
        chord_type = ChordType.MAJ_MIN7
    elif ct in ['MM7', '+7', 'Ger', 'It']:
        chord_type = ChordType.MAJ_MAJ7
    elif ct == 'mm7':
        chord_type = ChordType.MIN_MIN7
    elif ct == 'Fr':
        chord_type = ChordType.AUG_MAJ7
    elif ct == '+':
        chord_type = ChordType.AUGMENTED
    else:
        raise ValueError("Invalid chord type")
        
    return chord_type

    # MAJOR = 0
    # MINOR = 1
    # DIMINISHED = 2
    # AUGMENTED = 3
    # MAJ_MAJ7 = 4
    # MAJ_MIN7 = 5
    # MIN_MAJ7 = 6
    # MIN_MIN7 = 7
    # DIM7 = 8
    # HALF_DIM7 = 9
    # AUG_MIN7 = 10
    # AUG_MAJ7 = 11


            

def save_predictions_parameters_n_gram(pred, ground_truth, context, model, file_name_ext: str, verbose=False):
    dir = 'predictions/'
    name = '_' + file_name_ext
    with open(dir + 'n_gram_prediction' + name + '.pkl', 'wb') as f:
        pickle.dump(pred, f)
    with open(dir + 'n_gram_ground_truth' + name + '.pkl', 'wb') as f:
        pickle.dump(ground_truth, f)
    with open(dir + 'n_gram_context' + name + '.pkl', 'wb') as f:
        pickle.dump(context, f)
    with open(dir + 'n_gram_matrix' + name + '.pkl', 'wb') as f:
        pickle.dump(model.matrix, f)
    with open(dir + 'n_gram_unique_tokens_out' + name + '.pkl', 'wb') as f:
        pickle.dump(model.unique_tokens_out, f)
    with open(dir + 'n_gram_unique_tokens_in' + name + '.pkl', 'wb') as f:
        pickle.dump(model.unique_tokens_in, f)

    if verbose:
        print('='*60)
        print('model info of ' + file_name_ext + ' saved in directory predictions/')
        print('='*60)
