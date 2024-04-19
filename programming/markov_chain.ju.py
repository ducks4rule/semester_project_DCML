# %% [markdown]
"""
# Definition of a simple 1-gram Model
"""
# %%
# packages
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.utils.data
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np

import os

# %% [markdown]
"""
loading the data
"""

# %%
path = '/home/lorenz/Documents/Mathematik/24 FS/Semester_Paper_DCML/data/ABC/harmonies/n10op74_01.harmonies.tsv'
df_all = pd.read_csv(path, sep='\t')
print(df_all.columns)

# define new dataframe df with only the 'numeral' column
df = df_all['numeral']
# df = df_all['chord']
df_ = pd.factorize(df)
print(len(df_[1]))


# %% [markdown]
"""
Markov chain model
"""
# %%

class MarkovChain:
    def __init__(self, df):
        self.df = df
        self.df_ = pd.factorize(df)
        self.trans_mat = np.zeros((len(df_[1]), len(df_[1])))
    
    def clean_data(self):
        # if df_ contains a None value, remove it and the corresponding entry in df
        if None in self.df_[1]:
            ind = np.argwhere(self.df_[1] is None)
            self.df = np.delete(self.df, ind)
            self.df_ = pd.factorize(self.df)

    # calculate the transition matrix
    def markov_transition_matrix(self):
        trans_mat = np.zeros((len(self.df_[1]), len(self.df_[1])))

        # count of transitions
        for ch1 in self.df_[1]:
            inds = np.argwhere(self.df == ch1).flatten()
            # if first entry = 0, remove it 
            if inds[0] == 0:
                inds = np.delete(inds, 0)
            for ch2 in self.df_[1]:
                before = sum([chd == ch2 for chd in self.df[inds-1]])
                trans_mat[self.df_[1] == ch1, self.df_[1] == ch2] = before/len(self.df[inds])
        self.trans_mat = trans_mat
        return trans_mat

    def transform_chords_to_vectors(self, ch):
        assert(ch in self.df_[1])
        ind =  np.argwhere(self.df_[1] == ch)[0]
        vec = np.zeros(len(self.df_[1]))
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
        return np.random.choice(self.df_[1], 1, p=out)[0]

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


# %% [markdown]
"""
fit the model
"""
# %%
length_of_pred = 36
markov = MarkovChain(df).fit()
pred = markov.predict('I', n=length_of_pred, verbose=True)
    
# %% [markdown]
"""
save transition matrix and predictions
"""

# %%
# save predictions and ground truth in two files
np.savetxt('markov_predictions.csv', pred, delimiter=',', fmt='%s')
np.savetxt('markov_ground_truth.csv', df[:length_of_pred], delimiter=',', fmt='%s')

# %%
# save transition matrix in latx compatible format
matrix = markov.markov_transition_matrix()
np.savetxt('markov_transition_matrix.csv', matrix, delimiter=' & ', fmt='%1.2f', newline=' \\\\\n')
