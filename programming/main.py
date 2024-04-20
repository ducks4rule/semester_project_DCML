import numpy as np
import pandas as pd

from n_gram_model import NGram
from data_classes import MyToken, Numeral, Inversion, ChordType
import utils
import pickle

path = '~/Documents/Mathematik/24 FS/Semester_Paper_DCML/data/ABC/harmonies/'
file = 'n10op74_01.harmonies.tsv'
df_all = pd.read_csv(path + file, sep='\t')
df_all['figbass'].fillna(0, inplace=True)
df_all['figbass'] = df_all['figbass'].astype(int).astype(str)
df_all['figbass'].apply(lambda x: utils.translate_figbass(x))

# define new dataframe df with only the 'numeral' column
# df = df_all[['numeral', 'figbass']].values
df = []
for m, fb in zip(df_all['numeral'], df_all['figbass']):
    tok_nu = utils.translate_numeral(m)
    tok_in = utils.translate_figbass(fb)
    tok_ch = utils.translate_chord_type(tok_nu, fb)
    df.append(MyToken(tok_nu, tok_in, tok_ch))

len_df = len(df)



j = 0
# n = 3
inversions = [False, True]
verbose = False

for m in range(2, 10):
    for inversion in inversions:
        for j in range(5):
            print(m, inversion, j)
            n_gram_model = NGram(df, n=m, inversions=inversion)
            n_gram_model.fit()
            unique_tokens_in = n_gram_model.unique_tokens_in
            start = unique_tokens_in[0]
            prediction = n_gram_model.predict(start, 400, verbose=verbose)
            print(type(prediction))
            length = min(len_df, len(prediction))
            print('length ', length)

            # save preditcion to a file with pickle
            # also save df[:len(prediction)] to a file with pickle
            name = str(m) + '_inv_' + str(inversion) + '_num_' + str(j)
            with open('n_gram_prediction' + name + '.pkl', 'wb') as f:
                pickle.dump(prediction[:length], f)
            with open('n_gram_ground_truth' + name + '.pkl', 'wb') as f:
                pickle.dump(df[:length], f)
            with open('n_gram_matrix' + name + '.pkl', 'wb') as f:
                pickle.dump(n_gram_model.matrix, f)
            with open('n_gram_unique_tokens' + name + '.pkl', 'wb') as f:
                pickle.dump(n_gram_model.unique_tokens, f)
            with open('n_gram_unique_tokens_in' + name + '.pkl', 'wb') as f:
                pickle.dump(n_gram_model.unique_tokens_in, f)


