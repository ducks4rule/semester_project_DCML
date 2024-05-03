import numpy as np
import pandas as pd
import os

from n_gram_model import NGram
from data_classes import MyToken, Numeral, Inversion, ChordType
import utils
import pickle

 

def construct_data_per_file(file, path):
    df_all = pd.read_csv(path + file, sep='\t')
    df_all['figbass'] = df_all['figbass'].fillna(0)
    df_all['figbass'] = df_all['figbass'].astype(int).astype(str)
    df_all['figbass'].apply(lambda x: utils.translate_figbass(x))

# define new dataframe df with only the 'numeral' column
# df = df_all[['numeral', 'figbass']].values
    df = []
    for m, fb, ct in zip(df_all['numeral'], df_all['figbass'], df_all['chord_type']):
        if m == '@none' or fb == '@none' or ct == '@none':
            continue
        else:
            tok_nu = utils.translate_numeral(m)
            tok_in = utils.translate_figbass(fb)
            tok_ch = utils.translate_chord_type(ct)
            df.append(MyToken(tok_nu, tok_in, tok_ch))
    return df




def construct_data(save_data=False, verbose=False):
    path = "~/Documents/Mathematik/24 FS/Semester_Paper_DCML/data/ABC/harmonies/"
    path = os.path.expanduser(path)
    ext = '.tsv'
    # iterating over all files
    big_data_list = []
    for files in os.listdir(path):
        if files.endswith(ext):
            if verbose:
                print(files)  # printing file name of desired extension
            df = construct_data_per_file(files, path)
            big_data_list.append(df)
        else:
            continue

    if save_data:
        # save the big_data_list to a file with pickle
        with open('big_data_list.pkl', 'wb') as f:
            pickle.dump(big_data_list, f)
    print('Data constructed successfully!')
    return big_data_list
