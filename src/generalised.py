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
CONFIDENCE = 0.05  # stat confidence at 95 %


start_time = datetime.datetime.now()
date = start_time.strftime('%Y_%m_%d_%H%M')
out_folder = f'../results/generalised_{date}/'
os.mkdir(out_folder)
print(f'Output folder: {out_folder}')

print(f'Number of GPUs: {torch.cuda.device_count()}')

with open(f'{out_folder}summary.md', 'w') as w:
    w.write('# Accuracy table\n\n(Standard deviation on the cross-validation)')
    w.write('\n\n|Dataset|Chance level|')
    w.write('LDA (sd)|SVC (sd)|ANN (sd)|CNN (sd)|LSTM (sd)|\n')
    w.write('|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n')

with open(f'{out_folder}results.csv', 'w') as w:
    w.write('dataset;model;fold;accuracy;hyperparameters\n')


dict_accuracies = {}
for dataset in DATASETS.keys():
    print(f'=====\n{dataset}\n=====')
    out_path = f'{out_folder}{dataset}_'

    # Load and preprocess data
    epochs = load_dataset(dataset, bandpass=[0.01, 0.5], baseline=(-2, 0),
                          roi_sides=True, tddr=True)
    classes = DATASETS[dataset]
    epochs_lab = epochs[classes]

    # Run models
    nirs, labels, groups = process_epochs(epochs_lab, 9.9)
    lda, hps_lda, _ = machine_learn(
        nirs, labels, groups, 'lda', features=['mean', 'std', 'slope'],
        out_path=f'{out_path}lda_')
    svc, hps_svc, _ = machine_learn(
        nirs, labels, groups, 'svc', features=['mean', 'std', 'slope'],
        out_path=f'{out_path}svc_')
    ann, hps_ann, _ = deep_learn(
        nirs, labels, groups, len(classes), 'ann',
        features=['mean', 'std', 'slope'], out_path=f'{out_path}ann_')
    cnn, hps_cnn, _ = deep_learn(
        nirs, labels, groups, len(classes), 'cnn', features=None,
        out_path=f'{out_path}cnn_')
    lstm, hps_lstm, _ = deep_learn(
        nirs, labels, groups, len(classes), 'lstm', features=None,
        out_path=f'{out_path}lstm_')

    # Write results
    results = {'LDA': [lda, hps_lda], 'SVC': [svc, hps_svc],
               'ANN': [ann, hps_ann], 'CNN': [cnn, hps_cnn],
               'LSTM': [lstm, hps_lstm]}
    chance_level = np.around(1/len(classes), decimals=3)
    w_summary = open(f'{out_folder}summary.md', 'a')
    w_results = open(f'{out_folder}results.csv', 'a')
    w_summary.write(f'|{dataset}|{chance_level}|')
    for model in results.keys():
        w_summary.write(
            f'{np.around(np.mean(results[model][0]), decimals=3)} '
            f'({np.around(np.std(results[model][0]), decimals=3)})|')
        for fold, accuracy in enumerate(results[model][0]):
            hps = results[model][1][fold]
            w_results.write(f'{dataset};{model};{fold+1};{accuracy};"{hps}"\n')
    w_summary.write('\n')
    w_summary.close()
    w_results.close()
    dict_accuracies[dataset] = lda + svc + ann + cnn + lstm

dict_accuracies['Model'] = list(np.repeat(list(results.keys()), len(lda)))
df_accuracies = pd.DataFrame(dict_accuracies)
df_accuracies = df_accuracies.melt(
    id_vars=['Model'], value_vars=list(DATASETS.keys()),
    var_name='Dataset', value_name='Accuracy')
sns.pointplot(data=df_accuracies, y='Accuracy', x='Model', hue='Dataset',
              dodge=True, capsize=.1, palette='colorblind')
plt.legend(bbox_to_anchor=(1.01, 0.5), loc=6)
plt.savefig(f'{out_folder}summary.png', bbox_inches='tight')
plt.close()


# Stats
print('Stats...')
with open(f'{out_folder}stats.md', 'w') as w:
    df = pd.read_csv(f'{out_folder}results.csv', delimiter=';')
    w.write('## Comparison of model accuracies to chance level\n\n')
    w.write('|Dataset|Model|Shapiro p-value|Test|p-value|\n')
    w.write('|:---:|:---:|:---:|:---:|:---:|\n')
    anova_table = ''
    for dataset in DATASETS.keys():
        dataset_accuracies = []
        chance_level = 1 / len(DATASETS[dataset])
        normality = True
        for model in results.keys():
            w.write(f'|{dataset}|{model}|')
            sub_df = df[(df['dataset'] == dataset) & (df['model'] == model)]
            accuracies = sub_df['accuracy'].to_numpy()
            dataset_accuracies.append(accuracies)
            # Check normality of the distribution
            _, p_shap = stats.shapiro(accuracies)
            w.write(f'{p_shap}|')
            if p_shap > CONFIDENCE:
                # t-test
                _, p_tt = stats.ttest_1samp(accuracies, chance_level)
                w.write(f't-test|{p_tt}|\n')
            else:
                normality = False
                # Wilcoxon
                _, p_wilcox = stats.wilcoxon(accuracies-chance_level)
                w.write(f'Wilcoxon|{p_wilcox}|\n')
        _, p_bart = stats.bartlett(*dataset_accuracies)
        if normality and (p_bart > CONFIDENCE):
            _, p_anova = stats.f_oneway(*dataset_accuracies)
            anova_table += f'|{dataset}|{p_bart}|ANOVA|{p_anova}|\n'
        else:
            _, p_kru = stats.kruskal(*dataset_accuracies)
            anova_table += f'|{dataset}|{p_bart}|Kruskal|{p_kru}|\n'
    w.write('\n\n## Comparison of model accuracies to each other\n\n')
    w.write('|Dataset|Bartlett p-value|Test|p-value|\n')
    w.write(f'|:---:|:---:|:---:|:---:|\n{anova_table}')


end_time = datetime.datetime.now()
elapsed_time = end_time - start_time
print(f'===\nElapsed time: {elapsed_time}')
