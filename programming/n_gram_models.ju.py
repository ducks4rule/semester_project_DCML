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
path = '~/Documents/Mathematik/24 FS/Semester_Paper_DCML/data/ABC/harmonies/'
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
    def __init__(self, df, n_gram_: int = 2):
        self.df = df
        self.df_ = pd.factorize(df)
        assert(n_gram_ > 1)
        self.n_gram = n_gram_

        self.clean_data()

        self.input_df = self.generate_n_grams(self.df, self.n_gram-1)
        self.input_df_ = pd.factorize(self.input_df)
        self.data = self.generate_n_grams(self.df, self.n_gram)
        self.data_ = pd.factorize(self.data)
    
    def clean_data(self):
        # if df_ contains a None value, remove it and the corresponding entry in df
        if None in self.df_[1]:
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
        trans_mat = np.zeros((len(self.input_df_[1]), len(self.df_[1])))

        # count of transitions
        # data_split = np.array([tok.split()[:self.n_gram-1] for tok in self.data_[1]])
        for ch_lab, chords in enumerate(self.input_df_[1]):
            inds_chords = np.argwhere(self.input_df == chords).flatten()
            data_short = []
            labels_data = []
            for i, dat in enumerate(self.data_[1]):
                if dat.startswith(chords):
                    data_short.append(dat)
                    labels_data.append(i)
                
            before_count = len(inds_chords)
            followers = [tok.split()[-1] for tok in data_short]
            ind_followers = [np.argwhere(self.df_[1] == f)[0][0] for f in followers]
            followers_count = [sum(data_s == self.data) for data_s in data_short]
            followers_prob = [f_c/before_count for f_c in followers_count]
            for j, ind in enumerate(ind_followers):
                trans_mat[ch_lab, ind] = followers_prob[j]
        return trans_mat

    def transform_chords_to_vectors(self, ch):
        assert(ch in self.input_df_[1])
        ind =  np.argwhere(self.input_df_[1] == ch)[0]
        vec = np.zeros(len(self.input_df_[1]))
        vec[ind] = 1
        return vec

    def fit(self):
        self.clean_data()
        self.trans_mat = self.transition_probs()
        return self

    def step(self, ch):
        if len(ch.split()) == self.n_gram-1:
            print('simple step')
            return self.simple_step(ch)
        elif len(ch.split()) == 1:
            # TODO markov chain
            print('TODO this is where the markov chain should be')
        else:
            temp_mod = NGramModel(self.df, n_gram_=len(ch.split()) + 1)
            print('difficult step')
            return temp_mod.fit().step(ch)

    def simple_step(self, chs):
        vec = self.transform_chords_to_vectors(chs)
        out = np.dot(self.trans_mat.T, vec)
        return np.random.choice(self.df_[1], 1, p=out)[0]

    def predict(self, ch, n=5, verbose=False):
        seq = [ch]
        for i in range(n):
            print('step ', i)
            ch_temp = self.step(ch)
            seq.append(ch_temp)
            from_i = min(i, self.n_gram-1)
            ch = ' '.join(seq[-from_i:])
            print('ch ', ch)
            if verbose:
                print(ch)
        return seq


# %%
n_gram = NGramModel(df, n_gram_=4)
n_gram.fit()
pred = n_gram.predict('I V',n=5, verbose=True)
print('seq ', pred)

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
    
