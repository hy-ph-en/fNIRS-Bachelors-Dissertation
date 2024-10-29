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
DATASETS = {'herff_2014_nb': ['1-back', '2-back', '3-back']}        #only using this dataset as it is the smallest and therefore most processable when first creating

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

for dataset in DATASETS.keys():
    # Load and preprocess data
    epochs = load_dataset(dataset, path=data_path, bandpass=[0.01, 0.5],
                          baseline=(-2, 0), roi_sides=True, tddr=True)
    classes = DATASETS[dataset]
    epochs_lab = epochs[classes]

    # Run models
    nirs, labels, groups = process_epochs(epochs_lab, 9.9)
    accuracy_cnn, hps_cnn, metrics_cnn = deep_learn(
        nirs, labels, groups, len(classes), CustomCNN, features=None,
        out_path=f'{out_path}cnn_')
    print(accuracy_cnn)


