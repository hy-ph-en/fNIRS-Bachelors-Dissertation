from benchnirs.load import load_dataset
from benchnirs.viz import epochs_viz


ALL_DATA_PATH = '/folder/with/datasets/'  # path to the datasets
DATASETS = {'herff_2014_nb': ['1-back', '2-back', '3-back'],
            'shin_2018_nb': ['0-back', '2-back', '3-back'],
            'shin_2018_wg': ['baseline', 'word generation'],
            'shin_2016_ma': ['baseline', 'mental arithmetic'],
            'bak_2019_me': ['right', 'left', 'foot']}


for dataset in DATASETS.keys():
    print(f'=====\n{dataset}\n=====')
    path = f'{ALL_DATA_PATH}{dataset[:-3]}/'

    # Load and preprocess data
    epochs = load_dataset(dataset, path=path, bandpass=[0.01, 0.5],
                          baseline=(-1.99, 0), roi_sides=True, tddr=True)
    classes = DATASETS[dataset]
    epochs_lab = epochs[classes]
    epochs_viz(epochs_lab)
