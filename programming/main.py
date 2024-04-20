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
for n, fb in zip(df_all['numeral'], df_all['figbass']):
    tok_nu = utils.translate_numeral(n)
    tok_in = utils.translate_figbass(fb)
    tok_ch = utils.translate_chord_type(tok_nu, fb)
    df.append(MyToken(tok_nu, tok_in, tok_ch))

len_df = len(df)




n_gram_model = NGram(df, n=3, inversions=False)
n_gram_model.fit()
unique_tokens_in = n_gram_model.unique_tokens_in
start = unique_tokens_in[0]
prediction = n_gram_model.predict(start, 400, verbose=True)

# save preditcion to a file with pickle
# also save df[:len(prediction)] to a file with pickle
with open('prediction.pkl', 'wb') as f:
    pickle.dump(prediction, f)
with open('ground_truth.pkl', 'wb') as f:
    pickle.dump(df[:len(prediction)], f)

with open('prediction.pkl', 'rb') as f:
    prediction = pickle.load(f)
with open('ground_truth.pkl', 'rb') as f:
    ground_truth = pickle.load(f)

