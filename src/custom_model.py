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


ALL_DATA_PATH = 'C:/Users/lukak/Downloads/herff_2014/'  # path to the datasets
DATASETS = {'herff_2014_nb': ['1-back', '2-back', '3-back']}        #only using this dataset as it is the smallest and therefore most processable when first creating

class ANN(nn.Module):

    def __init__(self, n_classes):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(4, 10)  
        self.fc2 = nn.Linear(10, 10)   #third layer    
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
    print(len(nirs))
    print(len(labels))
    print(len(groups))
    #Is this how you costimise it to work with ANN, or is this wrong? I can make a ANN model but this is how you do it within the framework?
    accuracy_ann, hps_ann, metrics_ann = deep_learn(nirs, labels, groups, len(classes), ANN, features=['mean'], out_path='../results/')
    
    print(accuracy_ann)
    

#Stuff I need to add?

#Hyper-Parameters - IN
input_size = 20
num_classes = len(classes)

#OUT - How to add?
learning_rate = 0.05    #How to specificy learning rate, or is that taken care of <- it is taken care of in the deeplearning function, the learning rate it output in the hps_ann
batch_size = 64



