import mne
import nirsimple as ns
import numpy as np
import scipy.io

from mne.channels import combine_channels
from pandas import DataFrame


PATH_HERFF_2014 = '../../data/herff_2014/'
PATH_SHIN_2018 = '../../data/shin_2018/'
PATH_SHIN_2016 = '../../data/shin_2016/'
PATH_BAK_2019 = '../../data/bak_2019/'


def _temporal_derivative_distribution_repair(raw):
    """
    Function taken from MNE-Python and modified to work on non-standard
    channels

    # Authors: Robert Luke <mail@robertluke.net> and Frank Fishburn
    # License: BSD-3-Clause

    Apply temporal derivative distribution repair to data.

    Applies temporal derivative distribution repair (TDDR) to data
    (Fishburn et al. 2019). This approach removes baseline shift
    and spike artifacts without the need for any user-supplied parameters.

    Parameters
    ----------
    raw : instance of Raw
        The raw data.

    Returns
    -------
    raw : instance of Raw
         Data with TDDR applied.
    """
    from mne.io import BaseRaw
    from mne.utils import _validate_type
    from mne.io.pick import _picks_to_idx

    raw = raw.copy().load_data()
    _validate_type(raw, BaseRaw, 'raw')

    picks = _picks_to_idx(raw.info, 'fnirs', exclude=[])
    for pick in picks:
        raw._data[pick] = _TDDR(raw._data[pick], raw.info['sfreq'])

    return raw


def _TDDR(signal, sample_rate):
    """
    Function taken from MNE-Python

    # Authors: Robert Luke <mail@robertluke.net> and Frank Fishburn
    # License: BSD-3-Clause
    """
    # This function is the reference implementation for the TDDR algorithm for
    #   motion correction of fNIRS data, as described in:
    #
    #   Fishburn F.A., Ludlum R.S., Vaidya C.J., & Medvedev A.V. (2019).
    #   Temporal Derivative Distribution Repair (TDDR): A motion correction
    #   method for fNIRS. NeuroImage, 184, 171-179.
    #   https://doi.org/10.1016/j.neuroimage.2018.09.025
    #
    # Usage:
    #   signals_corrected = TDDR( signals , sample_rate );
    #
    # Inputs:
    #   signals: A [sample x channel] matrix of uncorrected optical density or
    #            hb data
    #   sample_rate: A scalar reflecting the rate of acquisition in Hz
    #
    # Outputs:
    #   signals_corrected: A [sample x channel] matrix of corrected optical
    #   density data
    from scipy.signal import butter, filtfilt
    signal = np.array(signal)
    if len(signal.shape) != 1:
        for ch in range(signal.shape[1]):
            signal[:, ch] = _TDDR(signal[:, ch], sample_rate)
        return signal

    # Preprocess: Separate high and low frequencies
    filter_cutoff = .5
    filter_order = 3
    Fc = filter_cutoff * 2 / sample_rate
    signal_mean = np.mean(signal)
    signal -= signal_mean
    if Fc < 1:
        fb, fa = butter(filter_order, Fc)
        signal_low = filtfilt(fb, fa, signal, padlen=0)
    else:
        signal_low = signal

    signal_high = signal - signal_low

    # Initialize
    tune = 4.685
    D = np.sqrt(np.finfo(signal.dtype).eps)
    mu = np.inf
    iter = 0

    # Step 1. Compute temporal derivative of the signal
    deriv = np.diff(signal_low)

    # Step 2. Initialize observation weights
    w = np.ones(deriv.shape)

    # Step 3. Iterative estimation of robust weights
    while iter < 50:

        iter = iter + 1
        mu0 = mu

        # Step 3a. Estimate weighted mean
        mu = np.sum(w * deriv) / np.sum(w)

        # Step 3b. Calculate absolute residuals of estimate
        dev = np.abs(deriv - mu)

        # Step 3c. Robust estimate of standard deviation of the residuals
        sigma = 1.4826 * np.median(dev)

        # Step 3d. Scale deviations by standard deviation and tuning parameter
        if sigma == 0:
            break
        r = dev / (sigma * tune)

        # Step 3e. Calculate new weights according to Tukey's biweight function
        w = ((1 - r**2) * (r < 1)) ** 2

        # Step 3f. Terminate if new estimate is within
        # machine-precision of old estimate
        if abs(mu - mu0) < D * max(abs(mu), abs(mu0)):
            break

    # Step 4. Apply robust weights to centered derivative
    new_deriv = w * (deriv - mu)

    # Step 5. Integrate corrected derivative
    signal_low_corrected = np.cumsum(np.insert(new_deriv, 0, 0.0))

    # Postprocess: Center the corrected signal
    signal_low_corrected = signal_low_corrected - np.mean(signal_low_corrected)

    # Postprocess: Merge back with uncorrected high frequency component
    signal_corrected = signal_low_corrected + signal_high + signal_mean

    return signal_corrected


class _DatasetHerff2014NB():
    """
    Loader for n-back data from Herff et al., 2014.
    """

    def __init__(self):
        self.subject_list = ['subject1', 'subject2', 'subject3', 'subject4',
                             'subject5', 'subject6', 'subject7', 'subject8',
                             'subject9', 'subject10']

    def load(self, subject, path=PATH_HERFF_2014):
        data = {}
        data['sfreq'] = 25  # Hz
        data['ch_names'] = ['CH1 HbO', 'CH1 HbR', 'CH2 HbO', 'CH2 HbR',
                            'CH3 HbO', 'CH3 HbR', 'CH4 HbO', 'CH4 HbR',
                            'CH5 HbO', 'CH5 HbR', 'CH6 HbO', 'CH6 HbR',
                            'CH7 HbO', 'CH7 HbR', 'CH8 HbO', 'CH8 HbR']
        data['ch_types'] = ['hbo', 'hbr', 'hbo', 'hbr',
                            'hbo', 'hbr', 'hbo', 'hbr',
                            'hbo', 'hbr', 'hbo', 'hbr',
                            'hbo', 'hbr', 'hbo', 'hbr']
        data['sides'] = {'Right PFC HbO': [0, 2, 4, 6],
                         'Right PFC HbR': [1, 3, 5, 7],
                         'Left PFC HbO': [8, 10, 12, 14],
                         'Left PFC HbR': [9, 11, 13, 15]}
        data['event_id'] = {'1-back': 1, '2-back': 2, '3-back': 3}
        data['tmin'] = -5
        data['tmax'] = 44

        # Load data matrix
        mat = scipy.io.loadmat(f'{path}{subject}.mat')

        # Get Hb data
        delta_c = mat['data']  # in uM
        delta_c = delta_c * 1e-6  # convert uM to M
        data['delta_c'] = delta_c

        # Get labels
        labels = mat['label']
        labels[labels == -1] = 6  # replace -1 by 6 to avoid Epochs issue
        data['labels'] = labels

        return data


class _DatasetShin2018NB():
    """
    Loader for n-back data from Shin et al., 2018.
    """

    def __init__(self):
        self.subject_list = ['VP001-NIRS', 'VP002-NIRS', 'VP003-NIRS',
                             'VP004-NIRS', 'VP005-NIRS', 'VP006-NIRS',
                             'VP007-NIRS', 'VP008-NIRS', 'VP009-NIRS',
                             'VP010-NIRS', 'VP011-NIRS', 'VP012-NIRS',
                             'VP013-NIRS', 'VP014-NIRS', 'VP015-NIRS',
                             'VP016-NIRS', 'VP017-NIRS', 'VP018-NIRS',
                             'VP019-NIRS', 'VP020-NIRS', 'VP021-NIRS',
                             'VP022-NIRS', 'VP023-NIRS', 'VP024-NIRS',
                             'VP025-NIRS', 'VP026-NIRS']

    def load(self, subject, path=PATH_SHIN_2018):
        data = {}
        data['sides'] = {'Right PFC HbO': [9, 10, 19, 20, 21, 22, 23],
                         'Right PFC HbR': [45, 46, 55, 56, 57, 58, 59],
                         'Left PFC HbO': [0, 1, 2, 3, 4, 5, 6],
                         'Left PFC HbR': [36, 37, 38, 39, 40, 41, 42]}
        data['event_id'] = {'0-back': 1, '2-back': 2, '3-back': 3}
        data['tmin'] = -2
        data['tmax'] = 40

        # Load data matrices
        cnt = scipy.io.loadmat(f'{path}{subject}/cnt_nback.mat')
        mrk = scipy.io.loadmat(f'{path}{subject}/mrk_nback.mat')

        # Get sampling frequency (in Hz)
        data['sfreq'] = cnt['cnt_nback']['oxy'][0, 0]['fs'][0, 0][0][0]

        # Get channel info
        ch_oxy = cnt['cnt_nback']['oxy'][0, 0]['clab'][0, 0][0]
        ch_deoxy = cnt['cnt_nback']['deoxy'][0, 0]['clab'][0, 0][0]
        ch_names_oxy = [f'{ch[0]} HbO' for ch in ch_oxy]
        ch_names_deoxy = [f'{ch[0]} HbR' for ch in ch_deoxy]
        ch_types_oxy = ['hbo' for ch in ch_names_oxy]
        ch_types_deoxy = ['hbr' for ch in ch_names_deoxy]
        data['ch_names'] = ch_names_oxy + ch_names_deoxy
        data['ch_types'] = ch_types_oxy + ch_types_deoxy

        # Get Hb data
        delta_oxy = cnt['cnt_nback']['oxy'][0, 0]['x'][0, 0]  # in mM
        delta_deoxy = cnt['cnt_nback']['deoxy'][0, 0]['x'][0, 0]  # in mM
        delta_c = np.hstack((delta_oxy, delta_deoxy))
        delta_c = delta_c * 1e-3  # convert mM to M
        delta_c = np.swapaxes(delta_c, 0, 1)
        data['delta_c'] = delta_c

        # Get labels
        markers_time = mrk['mrk_nback']['time'][0, 0]/1000*data['sfreq']
        markers_time = np.around(markers_time).astype(int)
        y_idx_0back = np.nonzero(mrk['mrk_nback']['y'][0, 0][0])
        y_idx_2back = np.nonzero(mrk['mrk_nback']['y'][0, 0][1])
        y_idx_3back = np.nonzero(mrk['mrk_nback']['y'][0, 0][2])
        time_0back = markers_time[0, y_idx_0back]
        time_2back = markers_time[0, y_idx_2back]
        time_3back = markers_time[0, y_idx_3back]
        labels = np.zeros((1, delta_c.shape[-1]))
        labels[0, time_0back] = 1
        labels[0, time_2back] = 2
        labels[0, time_3back] = 3
        data['labels'] = labels

        return data


class _DatasetShin2018WG():
    """
    Loader for word generation data from Shin et al., 2018.
    """

    def __init__(self):
        self.subject_list = ['VP001-NIRS', 'VP002-NIRS', 'VP003-NIRS',
                             'VP004-NIRS', 'VP005-NIRS', 'VP006-NIRS',
                             'VP007-NIRS', 'VP008-NIRS', 'VP009-NIRS',
                             'VP010-NIRS', 'VP011-NIRS', 'VP012-NIRS',
                             'VP013-NIRS', 'VP014-NIRS', 'VP015-NIRS',
                             'VP016-NIRS', 'VP017-NIRS', 'VP018-NIRS',
                             'VP019-NIRS', 'VP020-NIRS', 'VP021-NIRS',
                             'VP022-NIRS', 'VP023-NIRS', 'VP024-NIRS',
                             'VP025-NIRS', 'VP026-NIRS']

    def load(self, subject, path=PATH_SHIN_2018):
        data = {}
        data['sides'] = {'Right PFC HbO': [9, 10, 19, 20, 21, 22, 23],
                         'Right PFC HbR': [45, 46, 55, 56, 57, 58, 59],
                         'Left PFC HbO': [0, 1, 2, 3, 4, 5, 6],
                         'Left PFC HbR': [36, 37, 38, 39, 40, 41, 42]}
        data['event_id'] = {'baseline': 1, 'word generation': 2}
        data['tmin'] = -2
        data['tmax'] = 10

        # Load data matrices
        cnt = scipy.io.loadmat(f'{path}{subject}/cnt_wg.mat')
        mrk = scipy.io.loadmat(f'{path}{subject}/mrk_wg.mat')

        # Get sampling frequency (in Hz)
        data['sfreq'] = cnt['cnt_wg']['oxy'][0, 0]['fs'][0, 0][0][0]

        # Get channel info
        ch_oxy = cnt['cnt_wg']['oxy'][0, 0]['clab'][0, 0][0]
        ch_deoxy = cnt['cnt_wg']['deoxy'][0, 0]['clab'][0, 0][0]
        ch_names_oxy = [f'{ch[0]} HbO' for ch in ch_oxy]
        ch_names_deoxy = [f'{ch[0]} HbR' for ch in ch_deoxy]
        ch_types_oxy = ['hbo' for ch in ch_names_oxy]
        ch_types_deoxy = ['hbr' for ch in ch_names_deoxy]
        data['ch_names'] = ch_names_oxy + ch_names_deoxy
        data['ch_types'] = ch_types_oxy + ch_types_deoxy

        # Get Hb data
        delta_oxy = cnt['cnt_wg']['oxy'][0, 0]['x'][0, 0]  # in mM
        delta_deoxy = cnt['cnt_wg']['deoxy'][0, 0]['x'][0, 0]  # in mM
        delta_c = np.hstack((delta_oxy, delta_deoxy))
        delta_c = delta_c * 1e-3  # convert mM to M
        delta_c = np.swapaxes(delta_c, 0, 1)
        data['delta_c'] = delta_c

        # Get labels
        markers_time = mrk['mrk_wg']['time'][0, 0]/1000*data['sfreq']
        markers_time = np.around(markers_time).astype(int)
        y_idx_wg = np.nonzero(mrk['mrk_wg']['y'][0, 0][0])
        y_idx_bl = np.nonzero(mrk['mrk_wg']['y'][0, 0][1])
        time_wg = markers_time[0, y_idx_wg]
        time_bl = markers_time[0, y_idx_bl]
        labels = np.zeros((1, delta_c.shape[-1]))
        labels[0, time_wg] = 2
        labels[0, time_bl] = 1
        data['labels'] = labels

        return data


class _DatasetShin2016MA():
    """
    Loader for mental arithmetic data from Shin et al., 2016.
    """

    def __init__(self):
        self.subject_list = ['subject 01', 'subject 02', 'subject 03',
                             'subject 04', 'subject 05', 'subject 06',
                             'subject 07', 'subject 08', 'subject 09',
                             'subject 10', 'subject 11', 'subject 12',
                             'subject 13', 'subject 14', 'subject 15',
                             'subject 16', 'subject 17', 'subject 18',
                             'subject 19', 'subject 20', 'subject 21',
                             'subject 22', 'subject 23', 'subject 24',
                             'subject 25', 'subject 26', 'subject 27',
                             'subject 28', 'subject 29']

    def load(self, subject, path=PATH_SHIN_2016):
        data = {}
        data['sides'] = {'Right PFC HbO': [4, 6, 10, 64],
                         'Right PFC HbR': [5, 7, 11, 65],
                         'Left PFC HbO': [0, 2, 8, 62],
                         'Left PFC HbR': [1, 3, 9, 63]}
        data['event_id'] = {'baseline': 1, 'mental arithmetic': 2}
        data['tmin'] = -2
        data['tmax'] = 10

        # Load data matrices
        cnt = scipy.io.loadmat(f'{path}{subject}/cnt.mat')
        mrk = scipy.io.loadmat(f'{path}{subject}/mrk.mat')

        # Get sampling frequency (in Hz)
        data['sfreq'] = cnt['cnt'][0, 1]['fs'][0, 0][0][0]

        # Get original channel info
        wls = cnt['cnt'][0, 1]['wavelengths'][0, 0][0].tolist()
        dpfs = [6., 6.]
        chs = cnt['cnt'][0, 1]['clab'][0, 0][0]
        chs = [str(ch[0]) for ch in chs]
        ch_wls = []
        ch_dpfs = []
        for ch in chs:
            if 'lowWL' in ch:
                ch_wls.append(wls[0])
                ch_dpfs.append(dpfs[0])
            elif 'highWL' in ch:
                ch_wls.append(wls[1])
                ch_dpfs.append(dpfs[1])
        chs = [ch.replace('lowWL', '') for ch in chs]
        chs = [ch.replace('highWL', '') for ch in chs]
        ch_distances = [3 for ch in chs]

        # Load intensities and labels
        ma_idx = [1, 3, 5]
        x_blocks = []
        labels_blocks = []
        for idx in ma_idx:
            x = np.swapaxes(cnt['cnt'][0, idx]['x'][0, 0], 0, 1)
            markers_time = mrk['mrk'][0, idx]['time'][0, 0]/1000*data['sfreq']
            markers_time = np.around(markers_time).astype(int)
            y_idx_ma = np.nonzero(mrk['mrk'][0, idx]['y'][0, 0][0])
            y_idx_bl = np.nonzero(mrk['mrk'][0, idx]['y'][0, 0][1])
            time_ma = markers_time[0, y_idx_ma]
            time_bl = markers_time[0, y_idx_bl]
            labels = np.zeros((1, x.shape[-1]))
            labels[0, time_ma] = 2
            labels[0, time_bl] = 1
            x_blocks.append(x)
            labels_blocks.append(labels)

        # Get channel info and Hb data
        delta_c_blocks = []
        for x_block in x_blocks:
            dod = ns.intensities_to_od_changes(x_block)
            mbll_data = ns.mbll(dod, chs, ch_wls, ch_dpfs, ch_distances,
                                unit='cm', table='wray')
            dc, ch_names, ch_types = mbll_data
            delta_c_blocks.append(dc)
        mne_ch_names = []
        for idx, ch_name in enumerate(ch_names):
            if ch_types[idx] == 'hbo':
                mne_ch_names.append(ch_name + ' HbO')
            elif ch_types[idx] == 'hbr':
                mne_ch_names.append(ch_name + ' HbR')
        data['ch_names'] = mne_ch_names
        data['ch_types'] = ch_types
        delta_c = np.hstack(delta_c_blocks)
        data['delta_c'] = delta_c

        # Get labels
        labels = np.hstack(labels_blocks)
        data['labels'] = labels

        return data


class _DatasetBak2019ME():
    """
    Loader for motor execution data from Bak et al., 2019.
    """

    def __init__(self):
        self.subject_list = ['fNIRS 01', 'fNIRS 02', 'fNIRS 03',
                             'fNIRS 04', 'fNIRS 05', 'fNIRS 06',
                             'fNIRS 07', 'fNIRS 08', 'fNIRS 09',
                             'fNIRS 10', 'fNIRS 11', 'fNIRS 12',
                             'fNIRS 13', 'fNIRS 14', 'fNIRS 15',
                             'fNIRS 16', 'fNIRS 17', 'fNIRS 18',
                             'fNIRS 19', 'fNIRS 20', 'fNIRS 21',
                             'fNIRS 22', 'fNIRS 23', 'fNIRS 24',
                             'fNIRS 25', 'fNIRS 26', 'fNIRS 27',
                             'fNIRS 28', 'fNIRS 29', 'fNIRS 30']

    def load(self, subject, path=PATH_BAK_2019):
        data = {}
        data['sides'] = {'Right MC HbO': range(10, 20),
                         'Right MC HbR': range(30, 40),
                         'Left MC HbO': range(0, 10),
                         'Left MC HbR': range(20, 30)}
        data['event_id'] = {'right': 1, 'left': 2, 'foot': 3}
        data['tmin'] = -2
        data['tmax'] = 10

        # Load data matrices
        mat = scipy.io.loadmat(f'{path}{subject}.mat')

        # Get sampling frequency (in Hz)
        data['sfreq'] = mat['dat']['fs'][0, 0][0, 0]

        # Get channel info
        ch_names = [ch.tolist()[0] for ch in mat['dat']['clab'][0, 0][0]]
        ch_names = [ch.replace('deoxy', ' HbR') for ch in ch_names]
        ch_names = [ch.replace('oxy', ' HbO') for ch in ch_names]
        ch_types = ['hbo' if ch.endswith('O') else 'hbr' for ch in ch_names]
        data['ch_names'] = ch_names
        data['ch_types'] = ch_types

        # Get Hb data
        data_keys = [ch for ch in mat.keys() if ch.startswith('ch')]
        delta_c = np.array([mat[key] for key in data_keys]).squeeze()
        delta_c = delta_c * 1e-3  # convert mM to M
        data['delta_c'] = delta_c

        # Get labels
        markers_time = mat['mrk']['time'][0, 0]/1000*data['sfreq']
        markers_time = np.around(markers_time).astype(int)
        y_idx_right = np.nonzero(mat['mrk']['y'][0, 0][0])
        y_idx_left = np.nonzero(mat['mrk']['y'][0, 0][1])
        y_idx_foot = np.nonzero(mat['mrk']['y'][0, 0][2])
        time_right = markers_time[0, y_idx_right]
        time_left = markers_time[0, y_idx_left]
        time_foot = markers_time[0, y_idx_foot]
        labels = np.zeros((1, delta_c.shape[-1]))
        labels[0, time_right] = 1
        labels[0, time_left] = 2
        labels[0, time_foot] = 3
        data['labels'] = labels

        return data


def load_dataset(dataset, path=None, bandpass=None, order=4, tddr=False,
                 baseline=(None, 0), roi_sides=False):
    """
    Load and filter one of the open access dataset.

    Parameters
    ----------
    dataset : string
        Dataset to use. ``'herff_2014_nb'`` for n-back from Herff et al., 2014.
        ``'shin_2018_nb'`` for n-back from Shin et al., 2018.
        ``'shin_2018_wg'`` for word generation from Shin et al., 2018.
        ``'shin_2016_ma'`` for mental arithmetic from Shin et al., 2016.
        ``'bak_2019_me'`` for motor execution from Bak et al., 2019.

    path : string | None
        Path of the dataset selected with the ``dataset`` parameter. Defaults
        to ``None`` to use the default path.

    bandpass : list of floats | None
        Cutoff frequencies of the bandpass Butterworth filter. Defaults to
        ``None`` for no filtering.

    order : integer
        Order of the bandpass Butterworth filter.

    tddr : boolean
        Wether to apply temporal derivative distribution repair.

    baseline : None or tuple of length 2
        The time interval to apply baseline correction. If ``None`` do not
        apply it. If a tuple ``(a, b)`` the interval is between ``a`` and ``b``
        (in seconds). If ``a`` is ``None`` the beginning of the data is used
        and if ``b`` is ``None`` then ``b`` is set to the end of the interval.
        If ``(None, None)`` all the time interval is used. Correction is
        applied by computing mean of the baseline period and subtracting it
        from the data. The baseline ``(a, b)`` includes both endpoints, i.e.
        all timepoints ``t`` such that ``a <= t <= b``.

    roi_sides : boolean
        Wether to average channels by side.

    Returns
    -------
    epochs : MNE Epochs object
        MNE epochs of filtered data with associated labels. Subject IDs are
        contained in the ``metadata`` property.
    """
    print('Loading epochs...')
    raws = None
    groups = []

    if dataset == 'herff_2014_nb':
        loader = _DatasetHerff2014NB()
    elif dataset == 'shin_2018_nb':
        loader = _DatasetShin2018NB()
    elif dataset == 'shin_2018_wg':
        loader = _DatasetShin2018WG()
    elif dataset == 'shin_2016_ma':
        loader = _DatasetShin2016MA()
    elif dataset == 'bak_2019_me':
        loader = _DatasetBak2019ME()
    else:
        loader = None

    for subj_id, subj in enumerate(loader.subject_list):
        if path is None:
            data = loader.load(subj)
        else:
            data = loader.load(subj, path)

        # Create MNE raw object from delta_c
        info = mne.create_info(ch_names=data['ch_names'], sfreq=data['sfreq'],
                               ch_types=data['ch_types'])
        raw = mne.io.RawArray(data['delta_c'], info, verbose=False)

        # TDDR
        if tddr is True:
            raw = _temporal_derivative_distribution_repair(raw)

        # Add labels as a stim channel to the MNE raw object
        info_stim = mne.create_info(['STI'], raw.info['sfreq'], ['stim'])
        stim_raw = mne.io.RawArray(data['labels'], info_stim, verbose=False)
        raw.add_channels([stim_raw], force_update_info=True)

        # Filter raw data
        if bandpass is not None:
            iir_params = dict(order=order, ftype='butter', output='sos')
            raw.filter(*bandpass, method='iir', iir_params=iir_params,
                       verbose=False)

        # Average channels by side
        if roi_sides is True:
            raw = combine_channels(raw, data['sides'], method='mean',
                                   keep_stim=True)

        # Get unlabelled possible segments
        events = mne.find_events(raw, consecutive=True, stim_channel='STI',
                                 shortest_event=1, verbose=False)
        include = list(data['event_id'].values())
        labelleds = mne.pick_events(events, include=include)
        unlabelleds = []
        srange = int((data['tmax'] - data['tmin']) * data['sfreq']) + 1
        s = events[0, 0] + abs(int(data['tmin']*data['sfreq']))
        while s <= events[-1, 0] - int(data['tmax']*data['sfreq']):
            smin = s + int(data['tmin']*data['sfreq'])
            smax = s + int(data['tmax']*data['sfreq'])
            lmins = labelleds + int(data['tmin']*data['sfreq'])
            lmaxs = labelleds + int(data['tmax']*data['sfreq'])
            lmin_overlap = (smin <= lmins) & (lmins < smax)
            lmax_overlap = (smin <= lmaxs) & (lmaxs < smax)
            l_overlap = lmin_overlap | lmax_overlap
            if np.any(l_overlap):
                s = labelleds[l_overlap][-1] + srange
            else:
                unlabelleds.append([s, 0, 9])
                s += srange
        unlabelleds = np.array(unlabelleds)

        # Recreate stim channel with labelled and unlabelled events
        new_events = np.vstack((labelleds, unlabelleds))
        new_events[:, 1] = 0  # make events instantaneous
        raw.add_events(new_events, stim_channel='STI', replace=True)

        # Add subject labels associated with the epochs
        events = mne.find_events(raw, stim_channel='STI', shortest_event=1,
                                 verbose=False)
        data['event_id']['unlabelled'] = 9
        include = list(data['event_id'].values())
        targets = mne.pick_events(events, include=include)
        for _ in targets:
            groups.append(subj_id + 1)

        # Add to global raw
        if raws is None:
            raws = raw.copy()
        else:
            raws.append(raw.copy())

    # Cut epochs
    all_events = mne.find_events(raws, stim_channel='STI', shortest_event=1,
                                 verbose=False)
    groups = DataFrame(groups, columns=['Subject_ID'])
    epochs = mne.Epochs(raws, all_events, event_id=data['event_id'],
                        metadata=groups, preload=True, tmin=data['tmin'],
                        tmax=data['tmax'], baseline=baseline, verbose=False)
    epochs.resample(10)  # resample to 10 Hz

    return epochs
