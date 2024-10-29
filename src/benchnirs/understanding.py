import itertools
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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
    for k, out_idx in enumerate(out_split):         #splittin
        print(f'\tFOLD #{k+1}')
        
        #print(out_idx)
        #print("################")
        #print(out_idx[0])
        nirs_train, nirs_test = nirs[out_idx[0]], nirs[out_idx[1]]
        labels_train, labels_test = labels[out_idx[0]], labels[out_idx[1]]
        
        for(i++i<nirs.length*0.8)
            all_data_train.append.nirs[i]
            
        all_data_train 
            
        #print(labels[out_idx[0]])
        print(nirs_train)
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
        in_accuracies = [[] for _ in hp_list]           #hyperparameter training
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
