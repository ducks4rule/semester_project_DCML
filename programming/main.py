import matplotlib.pyplot as plt
import numpy as np
from n_gram_model import NGram
from data_classes import MyToken, Numeral, Inversion, ChordType
from load_data import construct_data
from metrics import calculate_metrics
import utils
import pickle
import subprocess




verbose = False

df_all = construct_data()


pred_n = 8
n_context = 16
n = [2, 3, 5, 8]
sqtt_number = [1, 4, 8, 16]
inversions = [False, True]

repeats = 3
scores_tbt = np.zeros(pred_n)
scores_bin = np.zeros(pred_n)
all_scores_tbt = {}
all_scores_bin = {}

for n_ in n:
    for sqtt_number_ in sqtt_number:
        for inversion in inversions:
            for j_ in range(repeats):
                df = df_all[sqtt_number_]
                context = df[:n_context]
                ground_truth = df[n_context:n_context + pred_n]
                model = NGram(df_all, n=n_, inversions=inversion)
                model.fit()
                file_name = 'test'
                pred = model.predict(context, pred_n, verbose=True)
                utils.save_predictions_parameters_n_gram(pred, ground_truth=ground_truth, model=model, context=context, file_name_ext=file_name, verbose=True)
                calculate_metrics(file_name=file_name, number=j_, sqtt_number=sqtt_number_, with_sps_metric=False, with_inversion=inversion)
                # print('n:', n_, 'sqtt_number:', sqtt_number_, 'inversion:', inversion, 'repeat:', j_)
                scores_tbt += np.load('scores/scores_tone_by_tone.npy')/repeats
                scores_bin += np.load('scores/scores_binary.npy')/repeats
            entry = f"n: {n_}, sqtt_number: {sqtt_number_}, inversion: {inversion}"
            all_scores_tbt[entry] = scores_tbt
            all_scores_bin[entry] = scores_bin

# save the dictionaries
with open('scores/all_scores_tbt.pkl', 'wb') as f:
    pickle.dump(all_scores_tbt, f)
with open('scores/all_scores_bin.pkl', 'wb') as f:
    pickle.dump(all_scores_bin, f)

# print the dictionaries
print(all_scores_tbt)
print(all_scores_bin)

# plot the dictionaries with plt 
plt.bar(all_scores_tbt.keys(), all_scores_tbt.values())
plt.show()

plt.bar(all_scores_bin.keys(), all_scores_bin.values())
plt.show()

