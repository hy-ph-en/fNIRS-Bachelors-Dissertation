Example
=======

Below is an example of how to use `BenchNIRS` with a custom convolutional neural network (CNN).

.. code-block:: python

    import datetime
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import pandas as pd
    import seaborn as sns
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from scipy import stats

    from benchnirs.load import load_dataset
    from benchnirs.process import process_epochs
    from benchnirs.learn import machine_learn, deep_learn


    ALL_DATA_PATH = '/folder/with/datasets/'  # path to the datasets
    DATASETS = {'herff_2014_nb': ['1-back', '2-back', '3-back'],
                'shin_2018_nb': ['0-back', '2-back', '3-back'],
                'shin_2018_wg': ['baseline', 'word generation'],
                'shin_2016_ma': ['baseline', 'mental arithmetic'],
                'bak_2019_me': ['right', 'left', 'foot']}
    CONFIDENCE = 0.05  # stat confidence at 95 %


    class CustomCNN(nn.Module):

        def __init__(self, n_classes):
            super(CustomCNN, self).__init__()
            self.conv1 = nn.Conv1d(4, 4, kernel_size=10, stride=2)  # tempo conv
            self.pool1 = nn.MaxPool1d(2)
            self.conv2 = nn.Conv1d(4, 4, kernel_size=5, stride=2)  # tempo conv
            self.pool2 = nn.MaxPool1d(2)
            self.fc1 = nn.Linear(20, 10)
            self.fc2 = nn.Linear(10, n_classes)

        def forward(self, x):
            batch_size = x.size(0)
            x = F.relu(self.conv1(x))
            x = self.pool1(x)
            x = F.relu(self.conv2(x))
            x = self.pool2(x)
            x = x.view(batch_size, -1)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x


    start_time = datetime.datetime.now()
    date = start_time.strftime('%Y_%m_%d_%H%M')
    out_folder = f'../results/custom_{date}/'
    os.mkdir(out_folder)
    print(f'Output folder: {out_folder}')

    print(f'Number of GPUs: {torch.cuda.device_count()}')

    with open(f'{out_folder}summary.md', 'w') as w:
        w.write('# Accuracy table\n\n(Standard deviation on the cross-validation)')
        w.write('\n\n|Dataset|Chance level|Average accuracy (sd)|\n')
        w.write('|:---:|:---:|:---:|\n')

    with open(f'{out_folder}results.csv', 'w') as w:
        w.write('dataset;fold;accuracy;hyperparameters\n')


    dict_accuracies = {'Accuracy': [], 'Dataset': []}
    for dataset in DATASETS.keys():
        print(f'=====\n{dataset}\n=====')
        data_path = f'{ALL_DATA_PATH}{dataset[:-3]}/'
        out_path = f'{out_folder}{dataset}_'

        # Load and preprocess data
        epochs = load_dataset(dataset, path=data_path, bandpass=[0.01, 0.5],
                            baseline=(-2, 0), roi_sides=True)
        classes = DATASETS[dataset]
        epochs_lab = epochs[classes]

        # Run models
        nirs, labels, groups = process_epochs(epochs_lab, 9.9)
        cnn, hps_cnn, metrics_cnn = deep_learn(
            nirs, labels, groups, len(classes), CustomCNN, features=None,
            out_path=f'{out_path}cnn_')

        # Write results
        results = {'CNN': [cnn, hps_cnn]}
        chance_level = np.around(1/len(classes), decimals=3)
        w_summary = open(f'{out_folder}summary.md', 'a')
        w_results = open(f'{out_folder}results.csv', 'a')
        w_summary.write(f'|{dataset}|{chance_level}|')
        w_summary.write(
            f'{np.around(np.mean(cnn), decimals=3)} '
            f'({np.around(np.std(cnn), decimals=3)})|')
        for fold, accuracy in enumerate(cnn):
            hps = hps_cnn[fold]
            w_results.write(f'{dataset};{fold+1};{accuracy};"{hps}"\n')
        w_summary.write('\n')
        w_summary.close()
        w_results.close()
        dict_accuracies['Accuracy'] += cnn
        dict_accuracies['Dataset'] += [dataset] * len(cnn)


    df_accuracies = pd.DataFrame(dict_accuracies)
    sns.barplot(data=df_accuracies, y='Accuracy', x='Dataset', capsize=.1,
                palette='colorblind')
    plt.savefig(f'{out_folder}summary.png')
    plt.close()


    # Stats
    print('Stats...')
    with open(f'{out_folder}stats.md', 'w') as w:
        df = pd.read_csv(f'{out_folder}results.csv', delimiter=';')
        w.write('## Comparison of the model accuracy to chance level\n\n')
        w.write('|Dataset|Shapiro p-value|Test|p-value|\n')
        w.write('|:---:|:---:|:---:|:---:|\n')
        for dataset in DATASETS.keys():
            dataset_accuracies = []
            chance_level = 1 / len(DATASETS[dataset])
            normality = True
            w.write(f'|{dataset}|')
            sub_df = df[df['dataset'] == dataset]
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


    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time
    print(f'===\nElapsed time: {elapsed_time}')
