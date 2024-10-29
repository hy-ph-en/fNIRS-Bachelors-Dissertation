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
out_folder = f'../results/personalised_{date}/'
os.mkdir(out_folder)
print(f'Output folder: {out_folder}')

print(f'Number of GPUs: {torch.cuda.device_count()}')

with open(f'{out_folder}summary.md', 'w') as w:
    w.write('# Accuracy table\n\n(Standard deviation on subjects)')
    w.write('\n\n|Dataset|Chance level|')
    w.write('LDA (sd)|SVC (sd)|ANN (sd)|CNN (sd)|LSTM (sd)|\n')
    w.write('|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n')

with open(f'{out_folder}results.csv', 'w') as w:
    w.write('dataset;subject;model;fold;accuracy;hyperparameters\n')


for dataset in DATASETS.keys():
    print(f'=====\n{dataset}\n=====')
    out_path = f'{out_folder}{dataset}_'

    # Load and preprocess data
    epochs = load_dataset(dataset, bandpass=[0.01, 0.5], baseline=(-2, 0),
                          roi_sides=True, tddr=True)
    classes = DATASETS[dataset]
    epochs_lab = epochs[classes]

    # Learn
    all_nirs, all_labels, all_groups = process_epochs(epochs_lab, 9.9)
    dict_accuracies = {'Model': [], 'Accuracy': []}
    all_results = {'LDA': [], 'SVC': [], 'ANN': [], 'CNN': [], 'LSTM': []}
    for subj in set(all_groups):
        print(f'-----\nSubject {subj+1}\n-----')
        indices = [i for i, value in enumerate(all_groups) if value == subj]
        nirs, labels = all_nirs[indices], all_labels[indices]

        # Run models
        lda, hps_lda, _ = machine_learn(
            nirs, labels, None, 'lda', features=['mean', 'std', 'slope'],
            out_path=f'{out_path}{subj+1}_lda_')
        svc, hps_svc, _ = machine_learn(
            nirs, labels, None, 'svc', features=['mean', 'std', 'slope'],
            out_path=f'{out_path}{subj+1}_svc_')
        ann, hps_ann, _ = deep_learn(
            nirs, labels, None, len(classes), 'ann',
            features=['mean', 'std', 'slope'],
            out_path=f'{out_path}{subj+1}_ann_')
        cnn, hps_cnn, _ = deep_learn(
            nirs, labels, None, len(classes), 'cnn', features=None,
            out_path=f'{out_path}{subj+1}_cnn_')
        lstm, hps_lstm, _ = deep_learn(
            nirs, labels, None, len(classes), 'lstm', features=None,
            out_path=f'{out_path}{subj+1}_lstm_')

        # Write results
        results = {'LDA': [lda, hps_lda], 'SVC': [svc, hps_svc],
                   'ANN': [ann, hps_ann], 'CNN': [cnn, hps_cnn],
                   'LSTM': [lstm, hps_lstm]}
        with open(f'{out_folder}results.csv', 'a') as w:
            for model in results.keys():
                dict_accuracies['Model'].append(model)
                dict_accuracies['Accuracy'].append(np.mean(results[model][0]))
                all_results[model].append(np.mean(results[model][0]))
                for fold, accuracy in enumerate(results[model][0]):
                    w.write(f'{dataset};{subj+1};{model};{fold+1};{accuracy};'
                            f'"{results[model][1][fold]}"\n')

    with open(f'{out_folder}summary.md', 'a') as w:
        chance_level = np.around(1/len(classes), decimals=3)
        w.write(f'|{dataset}|{chance_level}|')
        for model in all_results.keys():
            w.write(
                f'{np.around(np.mean(all_results[model]), decimals=3)} '
                f'({np.around(np.std(all_results[model]), decimals=3)})|')
        w.write('\n')

    df_accuracies = pd.DataFrame(dict_accuracies)
    sns.barplot(data=df_accuracies, y='Accuracy', x='Model', capsize=.1,
                palette='colorblind')
    plt.savefig(f'{out_path}summary.png')
    plt.close()


# Stats
print('Stats...')
with open(f'{out_folder}stats.md', 'w') as w:
    df = pd.read_csv(f'{out_folder}results.csv', delimiter=';')
    w.write('## Comparison of model accuracies to chance level\n\n')
    w.write('|Dataset|Subject|Model|Shapiro p-value|Test|p-value|\n')
    w.write('|:---:|:---:|:---:|:---:|:---:|:---:|\n')
    anova_table = ''
    for dataset in DATASETS.keys():
        df_dataset = df[df['dataset'] == dataset]
        subj_list = set(df_dataset['subject'].to_numpy())
        for subj in subj_list:
            dataset_accuracies = []
            chance_level = 1 / len(DATASETS[dataset])
            normality = True
            for model in results.keys():
                w.write(f'|{dataset}|{subj}|{model}|')
                sub_df = df_dataset[(df_dataset['subject'] == subj) &
                                    (df_dataset['model'] == model)]
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
                anova_table += f'|{dataset}|{subj}|{p_bart}|ANOVA|{p_anova}|\n'
            else:
                _, p_kru = stats.kruskal(*dataset_accuracies)
                anova_table += f'|{dataset}|{subj}|{p_bart}|Kruskal|{p_kru}|\n'
    w.write('\n\n## Comparison of model accuracies to each other\n\n')
    w.write('|Dataset|Subject|Bartlett p-value|Test|p-value|\n')
    w.write(f'|:---:|:---:|:---:|:---:|:---:|\n{anova_table}')


end_time = datetime.datetime.now()
elapsed_time = end_time - start_time
print(f'===\nElapsed time: {elapsed_time}')
