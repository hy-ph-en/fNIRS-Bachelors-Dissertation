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
from benchnirs.learn import machine_learn, deep_learn, k_means


ALL_DATA_PATH = 'C:/Users/lukak/Downloads/herff_2014/'  # path to the datasets
DATASETS = {'herff_2014_nb': ['1-back', '2-back', '3-back']}        #only using this dataset as it is the smallest and therefore most processable when first creating

class ANN(nn.Module):

    def __init__(self, n_classes):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(4, 10)         #Layers 
        self.fc2 = nn.Linear(10, 10)     
        self.fc3 = nn.Linear(10, n_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)
        return x

for dataset in DATASETS.keys():
    # Load and preprocess data
    epochs = load_dataset(dataset, path=ALL_DATA_PATH, bandpass=[0.01, 0.5],
                          baseline=(-2, 0), roi_sides=True, tddr=True)
    classes = DATASETS[dataset]
    epochs_lab = epochs[classes]

    # Run models
    nirs, labels, groups = process_epochs(epochs_lab, 9.9)
    
    #Finds New Labels
    cluster_ids = k_means(nirs,labels,groups,'deep_learn',len(classes), ANN, features=['mean'])

    print("done")
    

#Hyper-Parameters - IN
input_size = 20
num_classes = len(classes)
