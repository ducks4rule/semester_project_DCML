# %% [markdown]
"""
# Definition of a n-gram model for all n
"""
# %%
# packages
import pandas as pd
import numpy as np

# %% [markdown]
"""
loading the data
"""
# %%

class MyToken:
    def __init__(self,
                 numeral_: str,
                 inversion_: str):
        self.numeral = numeral_
        self.inversion = inversion_

    def __repr__(self):
        return f'{self.numeral} {self.inversion}'

    def __eq__(self, other):
        return self.numeral == other.numeral and self.inversion == other.inversion
    def __hash__(self):
        return hash(repr(self))

    def get_all_numerals(self, ts):
        return [t.numeral for t in ts]

    def get_all_inversions(self, ts):
        return [t.inversion for t in ts]


# %%
path = '~/Documents/Mathematik/24 FS/Semester_Paper_DCML/data/ABC/harmonies/'
file = 'n10op74_01.harmonies.tsv'
df_all = pd.read_csv(path + file, sep='\t')
df_all['figbass'].fillna('0', inplace=True)
df_all['figbass'] = df_all['figbass'].astype(int).astype(str)

# define new dataframe df with only the 'numeral' column
# df = df_all[['numeral', 'figbass']].values
df = []
for t in zip(df_all['numeral'], df_all['figbass']):
    df.append(MyToken(t[0], t[1]))
    
df_ = pd.factorize(df)

# %% [markdown]
"""
N-gram model
"""
# %%

class NGramModel:
    def __init__(self, df: list, n_gram_: int = 2):
        self.df = df
        self.df_ = pd.factorize(df)
        assert(n_gram_ > 1)
        self.n_gram = n_gram_

        self.input_df = self.generate_n_grams(self.df, self.n_gram-1)
        print(np.shape(self.input_df))
        # self.input_df_ = pd.factorize(self.input_df)
        self.data = self.generate_n_grams(self.df, self.n_gram)
        print(np.shape(self.data))
        self.data_ = pd.factorize(self.data)
    
    # generate n-grams
    def generate_n_grams(self, df, n_gram):
        chords = df
        temp = zip(*[chords[i:] for i in range(0, n_gram)])
        n_grams = []
        for t in temp:
            n_grams.append(t)
        return n_grams

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
        # self.clean_data()
        self.trans_mat = self.transition_probs()
        return self

    def step(self, ch):
        if len(ch.split()) == self.n_gram-1:
            print('simple step')
            return self.simple_step(ch)
        elif len(ch.split()) == 1:
            temp_mod = NGramModel(self.df, n_gram_=2)
            print('step w/ 2')
            return temp_mod.fit().step(ch)
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
n_gram = NGramModel(df, n_gram_=3)
n_gram.fit()
print(n_gram.generate_n_grams(df, 4))

n_gram.simple_step('I V')

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
    
