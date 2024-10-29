import mne
import numpy as np

from mne.viz import plot_epochs_image, plot_compare_evokeds
from scipy.stats import sem


def _standard_error(x):
    """
    Compute confidence interval bands based on standard error.

    Parameters
    ----------
    x : array of shape (n_observations, n_times)
        The signal observations.

    Returns
    -------
    ci : arrays of shape (2, n_times)
        The confidence interval bands.
    """
    lower_band = np.mean(x, axis=0) + sem(x)
    upper_band = np.mean(x, axis=0) - sem(x)
    ci = np.array([lower_band, upper_band])

    return ci


def epochs_viz(mne_epochs, reject_criteria=None):
    """
    Perform processing and visualization on epochs. Processing includes
    baseline cropping and bad epoch removal.

    Parameters
    ----------
    mne_epochs : MNE Epochs object
        MNE epochs of filtered data with associated labels. Subject IDs are
        contained in the ``metadata`` property.

    reject_criteria : list of floats | None
        List of the 2 peak-to-peak rejection thresholds for HbO and HbR
        channels respectively in uM. Defaults to ``None`` for no rejection.
    """
    # Process epochs
    epochs = mne_epochs.copy()
    if reject_criteria is not None:
        reject = {'hbo': reject_criteria[0]*1e-6,
                  'hbr': reject_criteria[1]*1e-6}
        original_verbose = epochs.verbose
        epochs.verbose = True
        epochs.drop_bad(reject=reject)
        epochs.verbose = original_verbose

    # Plot all epochs
    epochs.plot(events=epochs.events, scalings='auto', n_epochs=10, block=True)

    # Plot power spectral density
    epochs.plot_psd(fmin=0, fmax=5, average=True)

    # Visualise with average epochs superposed
    conditions = epochs.event_id.keys()
    for condition in conditions:
        plot_epochs_image(epochs[condition], combine='mean', title=condition)

    # Visualise condition comparison
    evoked_dict = {}
    styles_dict = {}
    linestyles = ['dotted', 'dashed', 'solid']
    for idx, condition in enumerate(conditions):
        evs_hbo = list(epochs[condition].pick_types(fnirs='hbo').iter_evoked())
        evs_hbr = list(epochs[condition].pick_types(fnirs='hbr').iter_evoked())
        for ev in evs_hbo:
            ev.rename_channels(lambda x: x[:-4])  # remove ' HbO' from ch_names
        for ev in evs_hbr:
            ev.rename_channels(lambda x: x[:-4])  # remove ' HbR' from ch_names
        evoked_dict[f'{condition}/HbO'] = evs_hbo
        evoked_dict[f'{condition}/HbR'] = evs_hbr
        styles_dict[condition] = {'linestyle': linestyles[idx]}
    color_dict = {'HbO': '#AA3377', 'HbR': 'b'}
    plot_compare_evokeds(evoked_dict, combine='mean', ci=0.95,
                         colors=color_dict, styles=styles_dict,
                         title='Conditions with 0.95 CI')
    plot_compare_evokeds(evoked_dict, combine='mean', ci=_standard_error,
                         colors=color_dict, styles=styles_dict,
                         title='Conditions with standard error')

    # Get only HbO and HbR and remove STIM channel
    nirs_epochs = epochs.pick_types(stim=False, fnirs=True)
    print(nirs_epochs)

    # Get fnirs data as a numpy array
    channels = nirs_epochs.info['ch_names']
    nirs = nirs_epochs.get_data()
    print(f'Channels: {channels}')
    print(f'NIRS data shape: {nirs.shape}')

    # Get label list
    labels = nirs_epochs.events[:, 2]
    print(f'Epoch labels shape: {labels.shape}')

    # Get standard deviations
    for label_class in np.unique(labels):
        indices_class = [i for i, x in enumerate(labels) if x == label_class]
        nirs_class = np.take(nirs, indices_class, axis=0)
        std_class = np.mean(np.std(nirs_class, axis=0))
        std_class = round(std_class * 1e6, 3)  # from M to uM
        print('Standard deviation across trials, averaged across channels and '
              f'time [condition {label_class}]: {std_class} uM')
