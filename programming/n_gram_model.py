from typing import List
from data_classes import Eq, Numeral, Inversion, MyToken
import numpy as np

class NGram:
    def __init__(self,
                 data: List[Eq],
                 n: int = 3,
                 inversions: bool = False):
        self.n = n
        self.inversion_switch = inversions
        self.data = self.inversions_yes_or_no(data)
        self.unique_tokens = self.get_output_tokens()
        self.unique_tokens.sort()
        self.unique_tokens_in = self.generate_n_gram(self.data, n - 1)
        self.unique_tokens_in = list(set(self.unique_tokens_in))
        self.unique_tokens_in.sort()
        self.matrix = np.zeros((len(self.unique_tokens_in), len(self.unique_tokens)))


    def inversions_yes_or_no(self, data):
        # set all inversion to 0 if inversion_switch is False
        if not self.inversion_switch:
            for dat in data:
                dat.inversion = 0
        return data

    def get_output_tokens(self):
        return list(set(self.data))

    def duplicated_head(self, xs: List[Eq]) -> bool:
        return xs[0].eq_to(xs[1])

    def generate_n_gram(self, xs: List[Eq], n: int) -> List[List[Eq]]:
        if self.duplicated_head(xs):
            xs.pop(0)

        n_grams = []
        for i in range(len(xs) - n + 1):
            n_grams.append(xs[i:i+n])
        return [tuple(gram) for gram in n_grams]
            
    def n_gram_transition_matrix(self, xs: List[Eq], n: int = 3) -> np.ndarray:
        n_grams = self.generate_n_gram(xs, n)
        unique_tokens_out = self.unique_tokens
        unique_tokens_in = self.unique_tokens_in

        n_out = len(unique_tokens_out)
        n_in = len(unique_tokens_in)
        matrix = np.zeros((n_in, n_out))
        for i in range(len(n_grams)):
            cur = unique_tokens_in.index(n_grams[i][:n - 1])
            nxt = unique_tokens_out.index(n_grams[i][-1])
            matrix[cur][nxt] += 1
        for i in range(n_in):
            sum = np.sum(matrix[i])
            if sum != 0:
                matrix[i] /= sum
            else:
                matrix[i] = np.zeros(n_out)
        return matrix

    def fit(self):
        self.matrix = self.n_gram_transition_matrix(self.data, self.n)
        return self

    def step(self, x: List[Eq]) -> Eq:
        index_vec = np.zeros(len(self.unique_tokens_in))
        index_vec[self.unique_tokens_in.index(x)] = 1

        matrix = self.n_gram_transition_matrix(self.data, self.n)

        out_vec = np.dot(matrix.T, index_vec)
        return np.random.choice(self.unique_tokens, 1, p=out_vec)[0]

    def toggle_list_tuple(self, x):
        if isinstance(x, list):
            return tuple(x)
        elif isinstance(x, tuple):
            return list(x)

    def predict(self, x: List[Eq], n: int, verbose=False) -> List[Eq]:
        x = self.inversions_yes_or_no(x)
            
        if len(x) > self.n - 1:
            x = x[:self.n - 1]
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
        return predictions

