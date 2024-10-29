from asyncio.windows_events import NULL
import itertools
from re import X
from turtle import xcor
import xdrlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tensorflow as tf
import math
import statistics
import pandas as pd

from pandas import DataFrame
from torch.utils.data import DataLoader, Dataset
from scipy.stats import linregress
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix)
from sklearn.model_selection import (StratifiedKFold, GroupKFold,
                                     train_test_split)
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle
from kmeans_pytorch import kmeans, kmeans_predict
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.spatial import distance
from numpy import zeros, newaxis
from statistics import mode



OUTER_K = 5
INNER_K = 5

# Standard machine learning parameters
MAX_ITER = 250000  # for support vector classifier
REG_LIST = [1e-3, 1e-2, 1e-1, 1e0]

# Deep learning parameters
MAX_EPOCH = 100
PATIENCE = 5  # for early stopping
BATCH_SIZE_LIST = [4, 8, 16, 32, 64]
LR_LIST = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]


def _extract_features(nirs, feature_list):
    """
    Perform feature extraction on NIRS data.

    Parameters
    ----------
    nirs : array of shape (n_epochs, n_channels, n_times)
        Processed NIRS data.

    feature_list : list of strings
        List of features to extract. The list can include ``'mean'`` for the
        mean along the time axis, ``'std'`` for standard deviation along the
        time axis and ``'slope'`` for the slope of the linear regression along
        the time axis.

    Returns
    -------
    nirs_features : array of shape (n_epochs, n_channels*n_features)
        Features extracted from NIRS data.
    """
    nirs_features = []
    for feature in feature_list:
        if feature == 'mean':
            feature = np.mean(nirs, axis=2)
        elif feature == 'std':
            feature = np.std(nirs, axis=2)
        elif feature == 'slope':
            x = range(nirs.shape[2])
            feature = []
            for epoch in nirs:
                ep_slopes = []
                for channel in epoch:
                    ep_slopes.append(linregress(x, channel).slope)
                feature.append(ep_slopes)
        nirs_features.append(feature)

    nirs_features = np.stack(nirs_features, axis=2)
    nirs_features = nirs_features.reshape(len(nirs), -1)  # flatten data

    return nirs_features


def machine_learn(nirs, labels, groups, model, features, normalize=False,
                  train_size=1., out_path='./'):
    """
    Perform nested k-fold cross-validation for standard machine learning models
    producing metrics and confusion matrices. The models include a linear
    discriminant analysis and a support vector classifier with grid search for
    the regularization parameter (inner k-fold cross-validation).

    Parameters
    ----------
    nirs : array of shape (n_epochs, n_channels, n_times)
        Processed NIRS data.

    labels : array of integer
        List of labels.

    groups : array of integer | None
        List of subject ID matching the epochs to perfrom a group k-fold
        cross-validation. If ``None``, performs a stratified k-fold
        cross-validation instead.

    model : string
        Standard machine learning to use. Either ``'lda'`` for a linear
        discriminant analysis or ``'svc'`` for a linear support vector
        classifier.

    features : list of string
        List of features to extract. The list can include ``'mean'`` for the
        mean along the time axis, ``'std'`` for standard deviation along the
        time axis and ``'slope'`` for the slope of the linear regression along
        the time axis.

    normalize : boolean
        Whether to normalize data before feeding to the model with min-max
        scaling based on the train set for each iteration of the outer
        cross-validation. Defaults to ``False`` for no normalization.

    train_size : float
        Percentage of the train set to keep. Used to study the impact of train
        set size. Defaults to ``1.`` to keep the whole train set.

    out_path : string
        Path to the directory into which the figures will be saved. Defaults to
        the file's directory.

    Returns
    -------
    accuracies : list of floats
        List of accuracies on the test sets (one for each iteration of the
        outer cross-validation).

    all_hps : list of floats | list of None
        List of regularization parameters for the SVC or a list of None for the
        LDA (one for each iteration of the outer cross-validation).

    additional_metrics : list of tuples
        List of tuples of metrics composed of (precision, recall, F1 score,
        support) on the outer cross-validation (one tuple for each iteration of
        the outer cross-validation). This uses the
        ``precision_recall_fscore_support`` function from scikit-learn with
        ``average='micro'``, ``y_true`` and ``y_pred`` being the true and the
        predictions on the specific iteration of the outer cross-validation.
    """
    print(f'Machine learning: {model}')

    # Feature extraction
    nirs = _extract_features(nirs, features)

    # K-fold cross-validator
    if groups is None:
        out_kf = StratifiedKFold(n_splits=OUTER_K)
        in_kf = StratifiedKFold(n_splits=INNER_K)
    else:
        out_kf = GroupKFold(n_splits=OUTER_K)
        in_kf = GroupKFold(n_splits=INNER_K)
    all_y_true = []
    all_y_pred = []
    accuracies = []
    additional_metrics = []
    all_hps = []
    out_split = out_kf.split(nirs, labels, groups)
    for k, out_idx in enumerate(out_split):
        print(f'\tFOLD #{k+1}')
        nirs_train, nirs_test = nirs[out_idx[0]], nirs[out_idx[1]]
        labels_train, labels_test = labels[out_idx[0]], labels[out_idx[1]]

        if groups is None:
            groups_train = None
            if train_size == 1.:
                nirs_train, labels_train = shuffle(nirs_train, labels_train)
            else:
                split = train_test_split(
                    nirs_train, labels_train, shuffle=True,
                    train_size=train_size, stratify=labels_train)
                nirs_train, labels_train = split[0], split[2]
        else:
            groups_train = groups[out_idx[0]]
            if train_size == 1.:
                nirs_train, labels_train, groups_train = shuffle(
                    nirs_train, labels_train, groups_train)
            else:
                split = train_test_split(
                    nirs_train, labels_train, groups_train, shuffle=True,
                    train_size=train_size, stratify=labels_train)
                nirs_train, labels_train, groups_train = (
                    split[0], split[2], split[4])

        all_y_true += labels_test.tolist()

        # Min-max scaling
        if normalize:
            maxs = nirs_train.max(axis=0)[np.newaxis, :]
            mins = nirs_train.min(axis=0)[np.newaxis, :]
            nirs_train = (nirs_train - mins) / (maxs - mins)
            nirs_test = (nirs_test - mins) / (maxs - mins)

        # LDA
        if model == 'lda':
            lda = LinearDiscriminantAnalysis()
            lda.fit(nirs_train, labels_train)
            y_pred = lda.predict(nirs_test).tolist()
            all_hps.append(None)

        # SVC
        elif model == 'svc':
            in_accuracies = [[] for _ in REG_LIST]
            for i, reg in enumerate(REG_LIST):
                in_split = in_kf.split(nirs_train, labels_train, groups_train)
                for in_idx in in_split:
                    nirs_in = nirs_train[in_idx[0]]
                    nirs_val = nirs_train[in_idx[1]]
                    labels_in = labels_train[in_idx[0]]
                    labels_val = labels_train[in_idx[1]]
                    svc = LinearSVC(C=reg, max_iter=MAX_ITER)
                    svc.fit(nirs_in, labels_in)
                    y_pred = svc.predict(nirs_val).tolist()
                    accuracy = accuracy_score(labels_val, y_pred)
                    in_accuracies[i].append(accuracy)

            # Get best SVC
            in_average_accuracies = np.mean(in_accuracies, axis=1)
            index_max = np.argmax(in_average_accuracies)
            best_reg = REG_LIST[index_max]
            all_hps.append(best_reg)

            # Fit best SVC on the whole train set
            svc = LinearSVC(C=best_reg, max_iter=MAX_ITER)
            svc.fit(nirs_train, labels_train)
            y_pred = svc.predict(nirs_test).tolist()

        # Metrics
        accuracies.append(accuracy_score(labels_test, y_pred))
        prfs = precision_recall_fscore_support(
            labels_test, y_pred, average='micro')
        additional_metrics.append(prfs)
        all_y_pred += y_pred

    # Figures
    cm = confusion_matrix(all_y_true, all_y_pred, normalize='true')
    sns.heatmap(cm, annot=True, cmap='crest')
    plt.xlabel('Predicted')
    plt.ylabel('Ground truth')
    plt.savefig(f'{out_path}confusion_matrix.png')
    plt.close()

    return accuracies, all_hps, additional_metrics


class _DatasetFromNumPy(Dataset):
    """
    Dataset loader from NumPy arrays for PyTorch.
    """

    def __init__(self, x, y):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


class _ANNClassifier(nn.Module):

    def __init__(self, n_classes):
        super(_ANNClassifier, self).__init__()
        self.fc1 = nn.Linear(12, 8)
        self.fc2 = nn.Linear(8, 4)
        self.fc3 = nn.Linear(4, n_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class _CNNClassifier(nn.Module):

    def __init__(self, n_classes):
        super(_CNNClassifier, self).__init__()
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


class _LSTMClassifier(nn.Module):

    def __init__(self, n_classes):
        super(_LSTMClassifier, self).__init__()
        self.unit_size = 20  # number of timepoints for each elt of the seq
        self.hidden_size = 36
        input_size = self.unit_size * 4  # number of timepoints x 4 channels
        self.lstm = nn.LSTM(input_size, self.hidden_size, batch_first=True)
        self.fc1 = nn.Linear(self.hidden_size, 16)
        self.fc2 = nn.Linear(16, n_classes)

    def forward(self, x):
        # Reshape
        r = x.size(-1) % self.unit_size
        if r > 0:
            x = x[:, :, :-r]  # crop to fit unit size
        x = x.reshape(x.size(0), 4, -1, self.unit_size)  # (b, ch, seq, tpts)
        x = x.permute(0, 2, 1, 3)  # (b, seq, ch, tpts)
        x = x.reshape(x.size(0), x.size(1), -1).double()

        # Initialise hidden and cell states
        h0 = torch.randn(1, x.size(0), self.hidden_size).double().to(x.device)
        c0 = torch.randn(1, x.size(0), self.hidden_size).double().to(x.device)

        # Feed to model
        x, _ = self.lstm(x, (h0, c0))
        x = x[:, -1, :]  # last output of the sequence
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def _train_dl(model_class, n_classes, batch_size, lr, nirs_train, labels_train,
              early_stop=False):
    """
    Create and train a deep learning model with PyTorch.
    """
    # Set device
    device_count = torch.cuda.device_count()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load data
    if early_stop:
        split = train_test_split(nirs_train, labels_train, shuffle=True,
                                 train_size=0.80,
                                 stratify=labels_train)
        nirs_train, labels_train = split[0], split[2]
        nirs_val, labels_val = split[1], split[3]
        dataset_val = _DatasetFromNumPy(nirs_val, labels_val)
        val_loader = DataLoader(dataset=dataset_val, batch_size=1,
                                shuffle=False)
    dataset_train = _DatasetFromNumPy(nirs_train, labels_train)
    train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size,
                              shuffle=True)

    # Instantiate model and hyperparameters
    clf = model_class(n_classes).double()
    if device_count > 1:
        clf = nn.DataParallel(clf)  # use multiple GPUs
    clf.to(device)
    criterion = nn.CrossEntropyLoss()  # LogSoftmax() & NLLLoss()
    optimizer = optim.Adam(clf.parameters(), lr=lr)

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    for epoch in range(MAX_EPOCH):  # loop over the data multiple times
        # Train
        clf.train()
        running_loss = 0.0
        correct = 0.0
        total = 0.0
        for i, data in enumerate(train_loader):
            # Get the inputs
            x, y = data[0].to(device), data[1].to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward
            outputs = clf(x)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, y)

            # Backward & optimize
            loss.backward()
            optimizer.step()

            # Get statistics
            running_loss += loss.item()
            total += y.size(0)
            correct += (predicted == y).sum()
            correct = int(correct)
        train_losses.append(running_loss / (i+1))
        train_accuracies.append(correct / total)

        if early_stop:
            # Validate
            clf.eval()
            with torch.no_grad():
                running_loss = 0.0
                correct = 0.0
                total = 0.0
                for i, data in enumerate(val_loader):
                    x, y = data[0].to(device), data[1].to(device)
                    outputs = clf(x)
                    _, predicted = torch.max(outputs.data, 1)
                    loss = criterion(outputs, y)
                    running_loss += loss.item()
                    total += y.size(0)
                    correct += (predicted == y).sum()
                    correct = int(correct)
                val_losses.append(running_loss / (i+1))
                val_accuracies.append(correct / total)
                last_sorted = sorted(val_losses[-PATIENCE:])
                if epoch >= PATIENCE and val_losses[-PATIENCE:] == last_sorted:
                    print(f'\t\t>Early stopping after {epoch+1} epochs')
                    break

    results = {'train_losses': train_losses,
               'train_accuracies': train_accuracies,
               'val_losses': val_losses,
               'val_accuracies': val_accuracies}
    return clf, results


def _test_dl(clf, nirs_test, labels_test):
    """
    Test a deep learning model with PyTorch.
    """
    # Set device
    device_count = torch.cuda.device_count()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load data sets
    dataset_test = _DatasetFromNumPy(nirs_test, labels_test)
    test_loader = DataLoader(dataset=dataset_test, batch_size=1,
                             shuffle=False)

    if device_count > 1:
        clf = nn.DataParallel(clf)  # use multiple GPUs
    clf.to(device)

    # Test
    clf.eval()
    with torch.no_grad():
        y_true = []
        y_pred = []
        correct = 0.0
        total = 0.0
        for data in test_loader:
            x, y = data[0].to(device), data[1].to(device)
            outputs = clf(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum()
            correct = int(correct)
            y_true.append(y.item())
            y_pred.append(predicted.item())
        results = {'test_accuracy': correct / total,
                   'y_true': y_true, 'y_pred': y_pred}
    return results


def deep_learn(nirs, labels, groups, n_classes, model, features=None,
               normalize=False, train_size=1., out_path='./'):
    """
    Perform nested k-fold cross-validation for a deep learning model. Produces
    loss graph, accuracy graph and confusion matrice. This is made with early
    stopping (with a validation set of 20 %) once the model has been fine tuned
    on the inner loop of the nested k-fold cross-validation.

    Parameters
    ----------
    nirs : array of shape (n_epochs, n_channels, n_times)
        Processed NIRS data.

    labels : array of integer
        List of labels.

    groups : array of integer | None
        List of subject ID matching the epochs to perfrom a group k-fold
        cross-validation. If ``None``, performs a stratified k-fold
        cross-validation instead.

    n_classes : integer
        Number of classes in the classification.

    model : string | PyTorch nn.Module class
        The PyTorch model class to use. If a string, can be either ``'ann'``,
        ``'cnn'`` or ``'lstm'``. If a PyTorch ``nn.Module`` class, the
        ``__init__()`` method must accept the number of classes as a parameter,
        and this needs to be the number of output neurons.

    features : list of string | None
        List of features to extract. The list can include ``'mean'`` for the
        mean along the time axis, ``'std'`` for standard deviation along the
        time axis and ``'slope'`` for the slope of the linear regression along
        the time axis. Defaults to ``None`` for no feature extration and using
        the raw data.

    normalize : boolean
        Whether to normalize data before feeding to the model with min-max
        scaling based on the train set for each iteration of the outer
        cross-validation. Defaults to ``False`` for no normalization.

    train_size : float
        Percentage of the train set to keep. Used to study the impact of train
        set size. Defaults to ``1.`` to keep the whole train set.

    out_path : string
        Path to the directory into which the figures will be saved. Defaults to
        the file's directory.

    Returns
    -------
    accuracies : list of floats
        List of accuracies on the test sets (one for each iteration of the
        outer cross-validation).

    all_hps : list of tuples
        List of hyperparameters (one tuple for each iteration of the outer
        cross-validation). Each tuple will be `(batch size, learning rate)`.

    additional_metrics : list of tuples
        List of tuples of metrics composed of (precision, recall, F1 score,
        support) on the outer cross-validation (one tuple for each iteration of
        the outer cross-validation). This uses the
        ``precision_recall_fscore_support`` function from scikit-learn with
        ``average='micro'``, ``y_true`` and ``y_pred`` being the true and the
        predictions on the specific iteration of the outer cross-validation.
    """
    print('Deep learning: ' + str(model))

    # Assign PyTorch model class
    if model == 'ann':
        model_class = _ANNClassifier
    elif model == 'cnn':
        model_class = _CNNClassifier
    elif model == 'lstm':
        model_class = _LSTMClassifier
    else:
        model_class = model

    # Feature extraction
    if features is not None:
        nirs = _extract_features(nirs, features)

    # K-fold cross-validation
    if groups is None:
        out_kf = StratifiedKFold(n_splits=OUTER_K)
        in_kf = StratifiedKFold(n_splits=INNER_K)
    else:
        out_kf = GroupKFold(n_splits=OUTER_K)
        in_kf = GroupKFold(n_splits=INNER_K)
    all_ks = []
    all_epochs = []
    all_train_losses = []
    all_train_accuracies = []
    all_val_losses = []
    all_val_accuracies = []
    all_y_true = []
    all_y_pred = []
    accuracies = []
    additional_metrics = []
    all_hps = []
    out_split = out_kf.split(nirs, labels, groups)
    for k, out_idx in enumerate(out_split):
        print(f'\tFOLD #{k+1}')

        nirs_train, nirs_test = nirs[out_idx[0]], nirs[out_idx[1]]
        labels_train, labels_test = labels[out_idx[0]], labels[out_idx[1]]

        if groups is None:
            groups_train = None
            if train_size == 1.:
                nirs_train, labels_train = shuffle(nirs_train, labels_train)
            else:
                split = train_test_split(
                    nirs_train, labels_train, shuffle=True,
                    train_size=train_size, stratify=labels_train)
                nirs_train, labels_train = split[0], split[2]
        else:
            groups_train = groups[out_idx[0]]
            if train_size == 1.:
                nirs_train, labels_train, groups_train = shuffle(
                    nirs_train, labels_train, groups_train)
            else:
                split = train_test_split(
                    nirs_train, labels_train, groups_train, shuffle=True,
                    train_size=train_size, stratify=labels_train)
                nirs_train, labels_train, groups_train = (
                    split[0], split[2], split[4])

        # Min-max scaling
        if normalize:
            if features is not None:
                maxs = nirs_train.max(axis=0)[np.newaxis, :]
                mins = nirs_train.min(axis=0)[np.newaxis, :]
            else:
                maxs = nirs_train.max(axis=(0, 2))[np.newaxis, :, np.newaxis]
                mins = nirs_train.min(axis=(0, 2))[np.newaxis, :, np.newaxis]
            nirs_train = (nirs_train - mins) / (maxs - mins)
            nirs_test = (nirs_test - mins) / (maxs - mins)

        hp_list = list(itertools.product(BATCH_SIZE_LIST, LR_LIST))
        in_accuracies = [[] for _ in hp_list]
        for i, hp in enumerate(hp_list):
            batch_size, lr = hp[0], hp[1]
            in_split = in_kf.split(nirs_train, labels_train, groups_train)
            for in_idx in in_split:
                nirs_in_train = nirs_train[in_idx[0]]
                labels_in_train = labels_train[in_idx[0]]
                nirs_val = nirs_train[in_idx[1]]
                labels_val = labels_train[in_idx[1]]

                clf, _ = _train_dl(model_class, n_classes, batch_size, lr,
                                   nirs_in_train, labels_in_train,
                                   early_stop=False)
                results = _test_dl(clf, nirs_val, labels_val)
                in_accuracies[i].append(results['test_accuracy'])

        # Get best hyperparameters
        in_average_accuracies = np.mean(in_accuracies, axis=1)
        index_max = np.argmax(in_average_accuracies)
        best_hps = hp_list[index_max]
        all_hps.append(best_hps)

        # Retrain with best hyperparameters
        clf, results = _train_dl(model_class, n_classes, best_hps[0],
                                 best_hps[1], nirs_train, labels_train,
                                 early_stop=True)

        # Append training metrics
        all_ks += [k for _ in results['train_losses']]
        all_epochs += [epoch for epoch in range(len(results['train_losses']))]
        all_train_losses += results['train_losses']
        all_val_losses += results['val_losses']
        all_train_accuracies += results['train_accuracies']
        all_val_accuracies += results['val_accuracies']

        results = _test_dl(clf, nirs_test, labels_test)
        all_y_true += results['y_true']
        all_y_pred += results['y_pred']
        accuracies.append(results['test_accuracy'])
        prfs = precision_recall_fscore_support(
            results['y_true'], results['y_pred'], average='micro')
        additional_metrics.append(prfs)

    # Plot loss and accuracy graphs
    _, axes = plt.subplots(ncols=2, figsize=(16, 6))
    dict_losses = {'k': all_ks, 'Epoch': all_epochs,
                   'Training': all_train_losses,
                   'Validation': all_val_losses}
    df_losses = DataFrame(dict_losses)
    df_losses = df_losses.melt(id_vars=['k', 'Epoch'],
                               value_vars=['Training', 'Validation'],
                               var_name='Condition', value_name='Loss')
    sns.lineplot(ax=axes[0], data=df_losses, y='Loss', x='Epoch',
                 hue='Condition', units='k', estimator=None)
    dict_accuracies = {'k': all_ks, 'Epoch': all_epochs,
                       'Training': all_train_accuracies,
                       'Validation': all_val_accuracies}
    df_accuracies = DataFrame(dict_accuracies)
    df_accuracies = df_accuracies.melt(id_vars=['k', 'Epoch'],
                                       value_vars=['Training', 'Validation'],
                                       var_name='Condition',
                                       value_name='Accuracy')
    sns.lineplot(ax=axes[1], data=df_accuracies, y='Accuracy', x='Epoch',
                 hue='Condition', units='k', estimator=None)
    plt.savefig(f'{out_path}graphs.png', bbox_inches='tight')
    plt.close()

    # Figures
    cm = confusion_matrix(all_y_true, all_y_pred, normalize='true')
    sns.heatmap(cm, annot=True, cmap='crest')
    plt.xlabel('Predicted')
    plt.ylabel('Ground truth')
    plt.savefig(f'{out_path}confusion_matrix.png')
    plt.close()

    return accuracies, all_hps, additional_metrics

#Where the New Code Will Go#
    #probably need a new feature extraction code 
    #wont import labels 
    
    #groups, n_classes, model, features=None, normalize=False, train_size=1., out_path='./'
def k_means(nirs,labels,groups, choosen_learn, classes, model, features):   #imports labels for comparison 
    
    
    #settings
    k_max = 10                    #num_clusters has to be changed later to be self improving, most likely using
    wss_setting = False           #If it should use wss
    silhouette_setting = True     #If it should use silhouette - need to fix data it seems
    cut_off = 1.1                 #Point at which the drop in wss performance means that it should stop
    kmeans_cluster_grouping = False  
    Mode = True                   #If to use Mode
    Mean = False                  #If to use Mean
    showGraphs = False            #Outputs graph on on every loop run
    labels_present = False        #If labels are present, automatically determined also
    run_pca = False               #if to run PCA
    components = 3                #how many components to reduce PCA by
    standarize = False            #Wether to standarize the value
    torch.set_printoptions(threshold=10_000)                                    #Allows for more terminal output
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning) #unnecessary to know
    
    #variables
    sse = []
    sil = []
    k_best = []
    k_best_wss = 3
    k_best_silhouette = NULL
    
    #label comparison - checks to see if the labels are present and if so compatible 
    if nirs.shape[0] == len(labels):
        labels_present = True
    else:
        y_train = []
        y_train = np.array(y_train)
    
    #run pca on data
    if run_pca:
        transfer = pca(nirs, components, standarize, showGraphs)
        nirs = None                                   #getting around the datasize issue
        nirs = transfer
    
    # set device
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
        
    #removes break condition 
    if Mode == True and Mean == True:
        print("Mode set to Default: Mean set to False")
        Mean = False
    if silhouette_setting == False:
        wss_setting = True
        print("wss set to default true due to silhouette being False")
    
    #Test Splitting - just used during testing
    #x, nirs_test, y_train, y_test = train_test_split(nirs, labels, train_size=1, random_state=42)            #just to make the datasize smaller
    #x, nirs_test, groups, y_test = train_test_split(nirs, groups, train_size=1, random_state=42)             #getting the groups, same random state to get the same outcome
    
    nirs_test = nirs    #if nirs_test is not being used
    x = nirs
    y_train = labels
    
    #Saving Values for Later
    keep_x = x
    keep_y = y_train
    
    #Data Organisation
    swap = np.array([np.transpose(x[0,:,:])])      #Placing Inital Element
    i = 1
    while i < x.shape[0]:
        transfer = x[i,:,:]                        #Slice of 3D array
        transfer = np.transpose(transfer)          #Tranposing Slice
        swap = np.append(swap, [transfer], axis=0) #Rebuilding 3D array
        i = i + 1                                  #Incremenetation
    x = swap                                       #Transfering rebuilt array
    
    swap = np.array([np.transpose(nirs_test[0,:,:])])
    i = 1
    while i < nirs_test.shape[0]:
        transfer = nirs_test[i,:,:]
        transfer = np.transpose(transfer)
        swap = np.append(swap, [transfer], axis=0)
        i = i + 1
    y = swap
    
    old_x = torch.from_numpy(x)
    for i in range(x.shape[0]):        #Data shuffle - Gives accurate k value
        np.random.shuffle(x[i,:,:])
    x_shuffle = x
    
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    

    #Finding optimal k - WSS
    for example in range(x.shape[0]):               #Currently the number of examples are reduced
        for num_clusters in range(2, k_max+1):      #goes through all the k values 
            if(wss_setting == False and silhouette_setting == False):
                print("Early Stop")
                break
            #later make the distance type costimisable 
            
            wss = 0
            cluster_ids_x, cluster_centers = kmeans(x[example,:,:], num_clusters=num_clusters, distance='euclidean', device=device)

            #Finding optimal k - WSS
            if(wss_setting):
                for count in range(x.shape[1]):
                    current_id = cluster_ids_x[count]        #i is the cluster value, x is the point in that cluster
                    wss += distance.euclidean(cluster_centers[current_id], x[example,count,:])

                    #can make a graph for the dis later using wss
                sse.append(wss)
                if len(sse)>=3:     
                    wss_decrease_past = sse[len(sse)-3]/ sse[len(sse)-2]
                    wss_decrease_current = sse[len(sse)-2]/sse[len(sse)-1]
                    
                    if(wss_decrease_past>(wss_decrease_current*cut_off)):   
                        k_best_wss = len(sse)             
                        wss_setting = False                  #Stop condition 
                        
            #Finding optimal k - Silhouette value
            if(silhouette_setting):
                sil.append(silhouette_score(x[example,:,:],
                       cluster_ids_x,metric = 'euclidean'))
                if(len(sil)>=3):
                    if(sil[len(sil)-2] > sil[len(sil)-1]):
                        sil.pop(0)
                        silhouette_setting = False           #Stop condition
                        k_best_silhouette = sil.index(max(sil))+3

                        if showGraphs:
                            #grapho-matic 
                            x_graph = []
                            for k in range(3, len(sil)+3):
                                x_graph.append(k)
                            plt.plot(x_graph, sil)
                            plt.xlabel('Cluster Number')
                            plt.ylabel('Silhouette Score')
                            plt.title('Silhouette Output')
                            plt.show()
        if showGraphs and wss_setting:
            #grapho-matic 
            x_graph = []
            for k in range(2, len(sse)+2):
                x_graph.append(k)
            plt.plot(x_graph, sse)
            plt.xlabel('Cluster Number')
            plt.ylabel('WSS Score')
            plt.title('Eblow Method Output')
            plt.show()
                                
        #Process for what k's to use and what not too
        if(len(sse) >= 1 and len(sil)>=1):
            wss_setting = True
            silhouette_setting = True
            if(k_best_wss == k_best_silhouette):
                print("Same k - Results back each other")
                k_best.append(k_best_silhouette)
            else:
                print("K is not Same - Default to Silhoutte score")
                k_best.append(k_best_silhouette)
        else:
            if(len(sse) >= 1):
                wss_setting = True
                k_best.append(k_best_wss)
            else:
                silhouette_setting = True
                k_best.append(k_best_silhouette)
        #Variable reset 
        sse = []
        sil = []
        
    k_value = round(sum(k_best)/len(k_best))    #finds best average k 
    print("best k")
    print(k_value)
    
    #Now that k has been found, we use this best k value to find the lables of all the values
    new_x = old_x.reshape(x.shape[0]*x.shape[1],x.shape[2])
    cluster_ids, centers = kmeans(new_x, num_clusters=k_value, distance='euclidean', device=torch.device('cuda:0'))
    
    #Mean label calcualtion
    label_mean = []
    for y in range(x.shape[0]):
        label_value = 0
        for i in range(x.shape[1]):         #takes average lable of every 100 time points 
           label_value += cluster_ids[i+(y*x.shape[1])].item()
        label_mean.append(round(label_value/x.shape[1]))
    
    #Mode label calculation
    label_mode = []    
    cluster_ids = cluster_ids.numpy()
    for y in range(x.shape[0]):
        label_mode.append(mode(cluster_ids[(y*x.shape[1]):((y*x.shape[1])+x.shape[1])]))

    #Mode or Mean - Also Makes it the Right Data Type
    if Mode:
        for y in range(x.shape[0]):
            y_train[y] = label_mode[y]
    else:
        for y in range(x.shape[0]):
            y_train[y] = label_mean[y]

    #Redoing the kmeans after the mode and mean have been done, as the labeling can change
    cluster_ids, centers = kmeans(new_x, num_clusters=(max(y_train)+1), distance='euclidean', device=torch.device('cuda:0'))
    
    if kmeans_cluster_grouping:
        final = [None]*x.shape[0]
        final = np.array(final)
        for y in range(x.shape[0]):
            final[y] =  mode(new_x[(y*x.shape[1]):((y*x.shape[1])+x.shape[1])]).numpy()
        np.array(final)
        
        y_train = None
        y_train, centers = kmeans(final, num_clusters=(max(y_train)+1), distance='euclidean', device=torch.device('cuda:0'))
        
    if(labels_present):
        TestingAgainstLabels(x_shuffle, old_x, x, keep_y, y_train, centers, cluster_ids) #or instead of y_train created_labels for kmeans of cluster groups
        AccuracyAgainstLabels(keep_x, keep_y, y_train, choosen_learn, groups, classes, model, features) #or instead of y_train created_labels for kmeans of cluster groups
        
    return y_train #or created_labels for kmeans of cluster groups, make this a setting

def TestingAgainstLabels(x_shuffle,old_x, x, keep_y, y_train, centers, cluster_ids):
    
    #Distance calculation for kmeans labeling - mode of points
    total_distance = []
    for i in range(max(y_train)+1):
        gap = 0
        for y in range(len(y_train)):
            if(y_train[y] == i):
                for h in range(len(old_x[y])):
                    gap += distance.euclidean(old_x[y,h,:], centers[i])
        total_distance.append(gap)
    distance_given = sum(total_distance)/len(total_distance)
    print("distance for mode of points")
    print(distance_given)
    
    #Distance calculation for kmeans labeling - individual
    list_lables = []
    for i in range(max(cluster_ids)+1):
        holding = []
        for y in range(cluster_ids.shape[0]):
            if(cluster_ids[y] == i):
                holding.append(y)               #keeps track of placement
        list_lables.append(holding)  
    
    list_lables = np.array(list_lables)

    flat_x = old_x.reshape(x.shape[0]*x.shape[1],x.shape[2])

    total_distance = []
    for i in range(max(cluster_ids)+1):
        gap = 0
        for y in range(len(list_lables[i])):
            if(cluster_ids[y] == i):
                list = np.array(list_lables[i])
                gap += distance.euclidean(flat_x[list[y]], centers[i])
        total_distance.append(gap)
    distance_kmeans = sum(total_distance)/len(total_distance)
    print("distance for mode of points individual")
    print(distance_kmeans)
    
    #Distance calculation for normal labeling
    list_lables = []
    for i in range(max(keep_y)+1):
        holding = []
        for y in range(keep_y.shape[0]):
            if(keep_y[y] == i):
                holding.append(y)               #keeps track of placement
        list_lables.append(holding)     
    list_lables = np.array(list_lables)
    clusters = []

    for y in range(len(list_lables)):
        holder = []
        for i in range(len(list_lables[y])):
            list = np.array(list_lables[y])
            for h in range(len(old_x[list[i],:])):
                holder.append(x_shuffle[list[i],h,:])
        holder = np.array(holder)
        holder.flatten()
        cluster = np.average(holder, axis=0)
        clusters.append(cluster)

    print("centers for assigned")       #needed for spacing calculations
    print(centers)
    print("centers for given")
    print(clusters)

    total_distance = []
    for i in range(max(keep_y)+1):
        gap = 0
        for y in range(len(keep_y)):
            if(keep_y[y] == i):
                for h in range(len(old_x[y])):
                    gap += distance.euclidean(old_x[y,h,:], clusters[i])
        total_distance.append(gap)
    distance_given = sum(total_distance)/len(total_distance)
    print("Distance calculation for Normal Labeling")
    print(distance_given)
    
    #Distance calculation, kmeans by cluster
    clustered_id = []
    for y in range(x.shape[0]):
        clustered_id.append(np.average(x[y,:], axis=0))

    cluster_ids, centers = kmeans(torch.from_numpy(np.array(clustered_id)), num_clusters=(max(y_train)+1), distance='euclidean', device=torch.device('cuda:0'))
    
    total_distance = []
    for i in range(max(y_train)+1):
        gap = 0
        for y in range(len(y_train)):
            if(y_train[y] == i):
                for h in range(len(old_x[y])):
                    gap += distance.euclidean(old_x[y,h,:], centers[i])
        total_distance.append(gap)
    distance_given = sum(total_distance)/len(total_distance)
    print("distance for mode of points by kernel")
    print(distance_given)

    return 0

def AccuracyAgainstLabels(keep_x, keep_y, y_train, choosen_learn, groups, classes, model, features):
    #kmeans labeling accuracy calculation
    if choosen_learn == 'deep_learn':
        accuracy, hps, metrics = deep_learn(keep_x, y_train, groups, classes, model, features=['mean'], out_path='../results - kmeans/')
    else:
        accuracy, hps, metrics = machine_learn(keep_x, y_train, groups, classes, model, features, out_path='../results - kmeans/')
    avg_accuracy = sum(accuracy)/len(accuracy)
    print("Assigned labeling accuracy")
    print(avg_accuracy)
    
    #Assigned labeling accuracy calculation
    if choosen_learn == 'deep_learn':
        accuracy, hps, metrics = deep_learn(keep_x, keep_y, groups, classes, model, features, out_path='../results/')
    else:
        accuracy, hps, metrics = machine_learn(keep_x, keep_y, groups, classes, model, features, out_path='../results/')
    avg_accuracy = sum(accuracy)/len(accuracy)
    print("Given Labeling Accuracy")
    print(avg_accuracy)
    
    return 0 
    
def pca(x, components, standarize, showGraphs):
    swap = np.array([np.transpose(x[0,:,:])])
    
    #configuring the data
    i = 1
    while i < x.shape[0]:
        transfer = x[i,:,:]
        transfer = np.transpose(transfer)
        swap = np.append(swap, [transfer], axis=0)
        i = i + 1
    x = swap

    #applying pca
    x = x.reshape(x.shape[0]*x.shape[1],x.shape[2])
    if standarize:
        nirs = StandardScaler().fit_transform(nirs)
    pca = PCA(n_components=components)
    principalComponents = pca.fit_transform(x)
    print("PCA Infromation Spread")
    print(pca.explained_variance_ratio_)
    
    optimum_components = 0
    for y in range(components):
        optimum_components = optimum_components + pca.explained_variance_ratio_[y]
        if(optimum_components > 0.8):
            print("Optimum Number of Componets")
            print(y)
            break
    
    #Making the 2D array into the orignal configuration
    num_examples = swap.shape[0]
    num_datapoints = swap.shape[1]
    swap = None
    swap = []
    principalComponents = np.array(principalComponents)
    principalComponents = np.transpose(principalComponents)
    remember_pca = principalComponents
    
    for i in range(num_examples):
        transfer = []
        for h in range(principalComponents.shape[0]):
            transfer.append(principalComponents[h,(i*num_datapoints):((i*num_datapoints)+num_datapoints)])
        swap.append(transfer)
    swap = np.array(swap)
    
    
    #Graphs
    if showGraphs:
        print("Graph to show the main information of the data against time")    #pca was also implemented to help with data visualisation
        time_points = list(range(1,x.shape[0]+1))
        for i in range(components):
            plt.plot(time_points, remember_pca[i])
        plt.xlabel('Time Points - 0.1 Seconds')
        plt.ylabel('Explained Variance')
        plt.title('Principal Component TimeLine Anyalsis')
        plt.show()
        
        grouped_data = []
        time_points = None
        print(remember_pca)
        for i in range(x.shape[0]):
            grouped_data.append(np.average(remember_pca[0,(i*x.shape[1]):((i*x.shape[1])+x.shape[1])]))
        time_points = list(range(1,len(grouped_data)+1))
        plt.plot(time_points, grouped_data)
        
        
        grouped_data = []
        time_points = None


        for i in range(x.shape[0]):  #As they were placed through 10 trails of each task
            grouped_data.append(np.average(remember_pca[0,(i*x.shape[1]*100):((i*x.shape[1]*100)+x.shape[1]*100)]))
        time_points = list(range(1,len(grouped_data)+1))
        for i in range(len(time_points)):
            transfer = time_points[i]
            time_points[i] = transfer*100
        plt.plot(time_points, grouped_data)
        
        plt.xlabel('Time Points - Seconds')
        plt.ylabel('Explained Variance')
        plt.title('Metal Exertion OverTime')
        plt.show()

        grouped_data = []
        time_points = None
        
        
    return swap