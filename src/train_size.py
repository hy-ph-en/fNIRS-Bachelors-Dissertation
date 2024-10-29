import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import torch

from scipy import stats

from benchnirs.load import load_dataset
from benchnirs.process import process_epochs
from benchnirs.learn import machine_learn, deep_learn


DATASETS = {'herff_2014_nb': ['1-back', '2-back', '3-back'],
            'shin_2018_nb': ['0-back', '2-back', '3-back'],
            'shin_2018_wg': ['baseline', 'word generation'],
            'shin_2016_ma': ['baseline', 'mental arithmetic'],
            'bak_2019_me': ['right', 'left', 'foot']}
TRAIN_SIZES = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
CONFIDENCE = 0.05  # stat confidence at 95 %


start_time = datetime.datetime.now()
date = start_time.strftime('%Y_%m_%d_%H%M')
out_folder = f'../results/train_size_{date}/'
os.mkdir(out_folder)
print(f'Output folder: {out_folder}')

print(f'Number of GPUs: {torch.cuda.device_count()}')

with open(f'{out_folder}results.csv', 'w') as w:
    w.write('dataset;train_size;model;fold;accuracy;hyperparameters\n')


for dataset in DATASETS.keys():
    print(f'=====\n{dataset}\n=====')
    out_path = f'{out_folder}{dataset}_'

    # Load and preprocess data
    epochs = load_dataset(dataset, bandpass=[0.01, 0.5], baseline=(-2, 0),
                          roi_sides=True, tddr=True)
    classes = DATASETS[dataset]
    epochs_lab = epochs[classes]

    nirs, labels, groups = process_epochs(epochs_lab, 9.9)
    dict_train_size = {'Train size': [], 'Chance': [], 'LDA': [], 'SVC': [],
                       'ANN': [], 'CNN': [], 'LSTM': []}
    for ts in TRAIN_SIZES:
        print(f'-----\nTrain size {ts}\n-----')
        # Run models
        lda, hps_lda, _ = machine_learn(
            nirs, labels, groups, 'lda', features=['mean', 'std', 'slope'],
            train_size=ts, out_path=f'{out_path}{ts}_lda_')
        svc, hps_svc, _ = machine_learn(
            nirs, labels, groups, 'svc', features=['mean', 'std', 'slope'],
            train_size=ts, out_path=f'{out_path}{ts}_svc_')
        ann, hps_ann, _ = deep_learn(
            nirs, labels, groups, len(classes), 'ann',
            features=['mean', 'std', 'slope'], train_size=ts,
            out_path=f'{out_path}{ts}_ann_')
        cnn, hps_cnn, _ = deep_learn(
            nirs, labels, groups, len(classes), 'cnn', features=None,
            train_size=ts, out_path=f'{out_path}{ts}_cnn_')
        lstm, hps_lstm, _ = deep_learn(
            nirs, labels, groups, len(classes), 'lstm', features=None,
            train_size=ts, out_path=f'{out_path}{ts}_lstm_')
        dict_train_size['Chance'] += [1/len(classes) for _ in lda]
        dict_train_size['LDA'] += lda
        dict_train_size['SVC'] += svc
        dict_train_size['ANN'] += ann
        dict_train_size['CNN'] += cnn
        dict_train_size['LSTM'] += lstm
        dict_train_size['Train size'] += [ts for _ in lda]

        # Write results
        results = {'LDA': [lda, hps_lda], 'SVC': [svc, hps_svc],
                   'ANN': [ann, hps_ann], 'CNN': [cnn, hps_cnn],
                   'LSTM': [lstm, hps_lstm]}
        with open(f'{out_folder}results.csv', 'a') as w:
            for model in results.keys():
                for fold, accuracy in enumerate(results[model][0]):
                    w.write(f'{dataset};{ts};{model};{fold+1};{accuracy};'
                            f'"{results[model][1][fold]}"\n')

    df_train_size = pd.DataFrame(dict_train_size)
    df_train_size = df_train_size.melt(
        id_vars=['Train size'], value_vars=list(results.keys())+['Chance'],
        var_name='Model', value_name='Accuracy')
    lp = sns.lineplot(data=df_train_size, y='Accuracy', x='Train size',
                      hue='Model', palette='colorblind')
    lp.set(ylim=(0.2, 0.7))
    plt.legend(bbox_to_anchor=(1.01, 0.5), loc=6)
    plt.savefig(f'{out_path}summary.png', bbox_inches='tight')
    plt.close()


# Stats
print('Stats...')
with open(f'{out_folder}stats.md', 'w') as w:
    df = pd.read_csv(f'{out_folder}results.csv', delimiter=';')
    w.write('## Correlation of model accuracies to train set size\n\n')
    w.write('|Dataset|Model|Shapiro p-value|Test|p-value|rho|\n')
    w.write('|:---:|:---:|:---:|:---:|:---:|:---:|\n')
    for dataset in DATASETS.keys():
        for model in results.keys():
            w.write(f'|{dataset}|{model}|')
            sub_df = df[(df['dataset'] == dataset) & (df['model'] == model)]
            accuracies = sub_df['accuracy'].to_numpy()
            train_sizes = sub_df['train_size'].to_numpy()
            # Check normality of the distribution
            _, p_shap = stats.shapiro(accuracies)
            w.write(f'{p_shap}|')
            if p_shap > CONFIDENCE:
                # Pearson's correlation test
                rho, p_pears = stats.pearsonr(accuracies, train_sizes)
                w.write(f'Pearson|{p_pears}|{rho}|\n')
            else:
                # Spearman's correlation test
                rho, p_spear = stats.spearmanr(accuracies, train_sizes)
                w.write(f'Spearman|{p_spear}|{rho}|\n')


end_time = datetime.datetime.now()
elapsed_time = end_time - start_time
print(f'===\nElapsed time: {elapsed_time}')
