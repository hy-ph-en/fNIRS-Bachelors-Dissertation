def process_epochs(mne_epochs, tmax=None, tslide=None, reject_criteria=None):
    """
    Perform processing on epochs including baseline cropping, bad epoch
    removal, label extraction and  unit conversion.

    Parameters
    ----------
    mne_epochs : MNE Epochs object
        MNE epochs of filtered data with associated labels. Subject IDs are
        contained in the ``metadata`` property.

    tmax : float | None
        End time of selection in seconds. Defaults to ``None`` to keep the
        initial end time.

    tslide : float | None
        Size of the sliding window in seconds. Will crop the epochs if ``tmax``
        is not a multiple of ``tslide``. Defaults to ``None`` for no window
        sliding.

    reject_criteria : list of floats | None
        List of the 2 peak-to-peak rejection thresholds for HbO and HbR
        channels respectively in uM. Defaults to ``None`` for no rejection.

    Returns
    -------
    nirs : array of shape (n_epochs, n_channels, n_times)
        Processed NIRS data in uM.

    labels : array of integer
        List of labels.

    groups : array of integer
        List of subject ID matching the epochs.
    """
    # Process epochs
    epochs = mne_epochs.copy()
    epochs.baseline = None
    epochs.crop(tmin=0, tmax=tmax)
    if reject_criteria is not None:
        reject = {'hbo': reject_criteria[0]*1e-6,
                  'hbr': reject_criteria[1]*1e-6}
        original_verbose = epochs.verbose
        epochs.verbose = True
        epochs.drop_bad(reject=reject)
        epochs.verbose = original_verbose

    # Extract data
    labels = epochs.events[:, 2] - 1  # start at 0
    groups = epochs.metadata['Subject_ID'].to_numpy() - 1  # start at 0
    nirs_epochs = epochs.pick_types(stim=False, fnirs=True)
    nirs = nirs_epochs.get_data()
    nirs *= 1e6  # convert from M to uM

    # Create sliding window
    if tslide is not None:
        sliding_size = int(tslide * nirs_epochs.info['sfreq'])
        r = nirs.shape[-1] % sliding_size
        if r > 0:
            nirs = nirs[:, :, :-r]  # crop to fit window size
        nirs = nirs.reshape(nirs.shape[0], nirs.shape[1], -1, sliding_size)
        labels = labels.repeat(nirs.shape[2])
        groups = groups.repeat(nirs.shape[2])
        nirs = nirs.swapaxes(1, 2)
        nirs = nirs.reshape(-1, nirs.shape[2], nirs.shape[3])

    print(f'Dataset shape: {nirs.shape}')

    return nirs, labels, groups
