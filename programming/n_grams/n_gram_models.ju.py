# %% [markdown]
"""
# Definition of a n-gram model for all n
"""
# %%
# packages
import pandas as pd
import numpy as np

import os

# %% [markdown]
"""
loading the data
"""

# %%
path = '/home/lorenz/Documents/Mathematik/24 FS/Semester_Paper_DCML/data/ABC/harmonies/'
file = 'n10op74_01.harmonies.tsv'
df_all = pd.read_csv(path + file, sep='\t')

# define new dataframe df with only the 'numeral' column
df = df_all['numeral']
# df = df_all['chord']
df_ = pd.factorize(df)
print(len(df_[1]))


# %% [markdown]
"""
N-gram model
"""
# %%

class NGramModel:
    def __init__(self, df, n_gram_=2):
        self.df = df
        self.n_gram = n_gram_

        self.clean_data()

        self.data = self.generate_n_grams(self.df, self.n_gram)
        self.data_ = pd.factorize(self.data)
    
    def clean_data(self):
        # if df_ contains a None value, remove it and the corresponding entry in df
        df_ = pd.factorize(self.df)
        if None in df_[1]:
            ind = np.argwhere(df_[1] is None)
            self.df = np.delete(self.df, ind)

    # generate n-grams
    def generate_n_grams(self, df, n_gram):
        chords = df.values
        temp = zip(*[chords[i:] for i in range(0, n_gram)])
        n_grams = []
        for t in temp:
            n_grams.append(' '.join(t))
        return np.array(n_grams)

    # calculate the transition matrix
    def transition_probs(self):
        trans_mat = np.zeros((len(self.data_[1]), len(self.data_[1])))

        # count of transitions
        for ch1 in self.data_[1]:
            inds = np.argwhere(self.df == ch1).flatten()
            # if first entry = 0, remove it 
            if inds[0] == 0:
                inds = np.delete(inds, 0)
            for ch2 in self.data_[1]:
                before = sum([chd == ch2 for chd in self.df[inds-1]])
                trans_mat[self.data_[1] == ch1, self.data_[1] == ch2] = before/len(self.df[inds])
        self.trans_mat = trans_mat
        return trans_mat

    def transform_chords_to_vectors(self, ch):
        assert(ch in self.data_[1])
        ind =  np.argwhere(self.data_[1] == ch)[0]
        vec = np.zeros(len(self.data_[1]))
        vec[ind] = 1
        return vec

    def fit(self):
        self.clean_data()
        self.trans_mat = self.markov_transition_matrix()
        return self

    def step(self, ch):
        # print('step form ', ch)
        vec = self.transform_chords_to_vectors(ch)
        out = np.dot(self.trans_mat.T, vec)
        return np.random.choice(self.data_[1], 1, p=out)[0]

    def predict(self, ch, n=3, verbose=False, start_at_current=False):
        seq = []
        if start_at_current:
            seq.append(ch)

        for i in range(n):
            ch = self.step(ch)
            seq.append(ch)
            if verbose:
                print(ch)
        return seq


# %%
n_gram = NGramModel(df, n_gram_=3)
print(n_gram.data_)

# %% [markdown]
"""
fit the model
"""
# %%
markov = MarkovChain(df).fit()
# markov.predict('I', n=36, verbose=True)
    
# %% [markdown]
"""
test the metrics
"""

# %%
import chord_eval as ce

start = 'I'
length = 32

pred_chords = markov.predict(start, n=length, start_at_current=True)

org_chords = df.values[:length]

for y, y_hat in zip(pred_chords, org_chords):
    print(y, y_hat)
    tone_by_tone = ce.get_distance(y, y_hat)
    
