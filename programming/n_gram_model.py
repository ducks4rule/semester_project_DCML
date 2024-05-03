from typing import List
from data_classes import Eq, Numeral, Inversion, ChordType, MyToken
import numpy as np

class NGram:
    def __init__(self,
                 data: List[Eq],
                 n: int = 3,
                 inversions: bool = False):
        self.n = n
        self.inversion_switch = inversions
        self.data = self.inversions_yes_or_no(data)
        self.unique_tokens_out = self.get_output_tokens()
        self.unique_tokens_in = self.get_input_tokens()
        self.matrix = np.zeros((len(self.unique_tokens_in), len(self.unique_tokens_out)))

    def inversions_yes_or_no(self, data):
        # set all inversion to 0 in case inversion_switch is False
        if not self.inversion_switch:
            for dat_list in data:
                for dat in dat_list:
                    dat.inversion = 0
        return data

    def get_output_tokens(self):
        out_list = []
        for dat in self.data:
            out_list += list(set(dat))
        out_list = list(set(out_list))
        out_list.sort()
        out = dict(zip(out_list, range(len(out_list))))
        return out

    def get_input_tokens(self):
        tok_list =[]
        for dat_list in self.data:
            tupls = self.generate_n_gram(dat_list, self.n - 1)
            tok_list += list(set(tupls))
        tok_list = list(set(tok_list))
        tok_list.sort()
        print('tok_list', len(tok_list))
        tok = dict(zip(tok_list, range(len(tok_list))))
        return tok

            

    def duplicated_head(self, xs: List[Eq]) -> bool:
        return xs[0].eq_to(xs[1])

    def generate_n_gram(self, xs: List[Eq], n: int) -> List[List[Eq]]:
        if self.duplicated_head(xs):
            xs.pop(0)

        n_grams = []
        for i in range(len(xs) - n + 1):
            n_grams.append(xs[i:i+n])
        return [tuple(gram) for gram in n_grams]
            
    def update_transision_matrix(self, xs: List[Eq], n: int = 3) -> np.ndarray:
        n_grams = self.generate_n_gram(xs, n)

        n_out = len(self.unique_tokens_out)
        n_in = len(self.unique_tokens_in)
        matrix = np.zeros((n_in, n_out))
        for i in range(len(n_grams)):
            cur = self.unique_tokens_in[n_grams[i][:n - 1]]
            nxt = self.unique_tokens_out[n_grams[i][-1]]
            matrix[cur][nxt] += 1
        return matrix

    def n_gram_transition_matrix(self) -> np.ndarray:
        for dat_list in self.data:
            self.matrix += self.update_transision_matrix(dat_list, self.n)

        n_in = self.matrix.shape[0]
        n_out = self.matrix.shape[1]
        for i in range(n_in):
            sum = np.sum(self.matrix[i])
            if sum != 0:
                self.matrix[i] /= sum
            else:
                self.matrix[i] = np.zeros(n_out)
        return self.matrix

    def fit(self):
        self.matrix = self.n_gram_transition_matrix()
        return self

    def step(self, x: List[Eq]) -> Eq:
        index_vec = np.zeros(len(self.unique_tokens_in))
        index_vec[self.unique_tokens_in[x]] = 1

        out_vec = np.dot(self.matrix.T, index_vec)
        return np.random.choice(list(self.unique_tokens_out.keys()), 1, p=out_vec)[0]

    def toggle_list_tuple(self, x):
        if isinstance(x, list):
            return tuple(x)
        elif isinstance(x, tuple):
            return list(x)

    def predict(self, x: List[Eq], n: int, verbose=False) -> List[Eq]:
        x = self.inversions_yes_or_no([x])[0]
            
        if len(x) > self.n - 1:
            x = x[-(self.n - 1):]
            print('list shortend to ', len(x))

        if isinstance(x, list):
            predictions = x
            x = tuple(x)
        else:
            predictions = list(x)
        for i in range(n):
            if verbose and i % 40 == 0:
                print('step ', i, ' of ', n)
            x = self.step(x)
            predictions.append(x)
            x = predictions[-self.n + 1:]
            x = self.toggle_list_tuple(x)
        return predictions[-n:]

