import os
import pickle
import itertools
import operator
import numpy as np
import h5py
from tqdm import tqdm
from scipy import io
from scipy.stats import binned_statistic
from scipy.interpolate import interp1d
from scipy.signal import resample,  convolve
from scipy.ndimage import convolve1d, gaussian_filter1d
from copy import deepcopy
import pdb

from pynwb import NWBHDF5IO

FILTER_DICT = {'gaussian':gaussian_filter1d, 'none': lambda x, **kwargs: x}

def moving_center(X, n, axis=0):
    if n % 2 == 0:
        n += 1
    w = -np.ones(n) / n
    w[n // 2] += 1
    X_ctd = convolve1d(X, w, axis=axis)
    return X_ctd

def sinc_filter(X, fc, axis=0):
        
    # Windowed sinc filter
    b = 0.08  # Transition band, as a fraction of the sampling rate (in (0, 0.5)).
    N = int(np.ceil((4 / b)))
    if not N % 2: N += 1  # Make sure that N is odd.
    n = np.arange(N)
    
    # Compute sinc filter.
    h = np.sinc(2 * fc * (n - (N - 1) / 2))

    # Compute Blackman window.
    w = 0.42 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) + \
        0.08 * np.cos(4 * np.pi * n / (N - 1))

    # Multiply sinc filter by window.
    h = h * w

    # Normalize to get unity gain.
    h = h / np.sum(h)
    return convolve(X, h)        

def window_spike_array(spike_times, tstart, tend):
    windowed_spike_times = np.zeros(spike_times.shape, dtype=np.object)

    for i in range(spike_times.shape[0]):
        for j in range(spike_times.shape[1]):
            wst, _ = window_spikes(spike_times[i, j], tstart[i], tend[i])
            windowed_spike_times[i, j] = wst

    return windowed_spike_times

def window_spikes(spike_times, tstart, tend, start_idx=0):

    spike_times = spike_times[start_idx:]
    spike_times[spike_times > tstart]

    if len(spike_times) > 0:
        start_idx = np.argmax(spike_times > tstart)
        end_idx = np.argmin(spike_times < tend)

        windowed_spike_times = spike_times[start_idx:end_idx]

        # Offset spike_times to start at 0
        if windowed_spike_times.size > 0:
                windowed_spike_times -= tstart

        return windowed_spike_times, end_idx - 1
    else:
        return np.array([]), start_idx

def align_behavior(x, T, bin_width):
    
    bins = np.linspace(0, T, int(T//bin_width))
    bin_centers = bins + (bins[1] - bins[0])/2
    bin_centers = bin_centers[:-1]
    xaligned = np.zeros((bin_centers.size, x.shape[-1]))
    
    for j in range(x.shape[-1]):
        interpolator = interp1d(np.linspace(0, T, x[:, j].size), x[:, j])
        xaligned[:, j] = interpolator(bin_centers)

    return xaligned

def align_peanut_behavior(t, x, bins):
    # Offset to 0
    t -= t[0]
    bin_centers = bins + (bins[1] - bins[0])/2
    bin_centers = bin_centers[:-1]
    interpolator = interp1d(t, x, axis=0)
    xaligned = interpolator(bin_centers)
    return xaligned, bin_centers

# spike_times: (n_trial, n_neurons)
#  trial threshold: If we require a spike threshold, trial threshold = 1 requires 
#  the spike threshold to hold for the neuron for all trials. 0 would mean no trials
def postprocess_spikes(spike_times, T, bin_width, boxcox, filter_fn, filter_kwargs,
                       spike_threshold=0, trial_threshold=1, high_pass=False):

    # Trials are of different duration
    if np.isscalar(T):
        ragged_trials = False
    else:
        ragged_trials = True

    # Discretize time over bins
    if ragged_trials:
        bins = []
        for i in range(len(T)):
            bins.append(np.linspace(0, T[i], int(T[i]//bin_width)))
        bins = np.array(bins, dtype=np.object)
        spike_rates = np.zeros((spike_times.shape[0], spike_times.shape[1]), dtype=np.object)
    else:
        bins = np.linspace(0, T, int(T//bin_width))
        spike_rates = np.zeros((spike_times.shape[0], spike_times.shape[1], bins.size - 1,))    

    # Did the trial/unit have enough spikes?
    insufficient_spikes = np.zeros(spike_times.shape)
    print('Processing spikes')
    for i in tqdm(range(spike_times.shape[0])):
        for j in range(spike_times.shape[1]):    

            # Ignore this trial/unit combo
            if np.any(np.isnan(spike_times[i, j])):
                insufficient_spikes[i, j] = 1          

            if ragged_trials:
                spike_counts = np.histogram(spike_times[i, j], bins=np.squeeze(bins[i]))[0]    
            else:
                spike_counts = np.histogram(spike_times[i, j], bins=bins)[0]

            if spike_threshold is not None:
                if np.sum(spike_counts) <= spike_threshold:
                    insufficient_spikes[i, j] = 1

            # Apply a boxcox transformation
            if boxcox is not None:
                spike_counts = np.array([(np.power(spike_count, boxcox) - 1)/boxcox 
                                         for spike_count in spike_counts])

            # Filter the resulting spike counts
            spike_rates_ = FILTER_DICT[filter_fn](spike_counts.astype(np.float), **filter_kwargs)

            # High pass to remove long term trends (needed for sabes data)
            if high_pass:
                spike_rates_ = moving_center(spike_rates_, 600)

            spike_rates[i, j] = spike_rates_

    # Filter out bad units
    sufficient_spikes = np.arange(spike_times.shape[1])[np.sum(insufficient_spikes, axis=0) < \
                                                        (1 - (trial_threshold -1e-3)) * spike_times.shape[0]]

    spike_rates = spike_rates[:, list(sufficient_spikes)]

    # Transpose so time is along the the second 'axis'
    if ragged_trials:
        spike_rates = [np.array([spike_rates[i, j] for j in range(spike_rates.shape[1])]).T for i in range(spike_rates.shape[0])]
    else:
        spike_rates = np.transpose(spike_rates, (0, 2, 1))

    return spike_rates

# Loader that operates on the files provided by the Shenoy lab
def load_shenoy(data_path, bin_width, boxcox, filter_fn, filter_kwargs, 
                spike_threshold=None, trial_threshold=0.5, tw=(-250, 550), 
                trialVersions='all', trialTypes='all', region='both'):

    # Code checks for list membership in specified trialtypes/trialversions
    if trialVersions != 'all' and type(trialVersions) != list:
        trialVersions = [trialVersions]
    if trialTypes != 'all' and type(trialTypes) != list:
        trialTypes = [trialTypes]

    dat = {}
    f = io.loadmat(data_path, squeeze_me=True, struct_as_record=False)
     
    # Filter out trials we should not use, period.
    trial_filters = {'success': 0, 'possibleRTproblem' : 1, 'unhittable' : 1, 'trialType': 0, 
                     'novelMaze': 1}
    
    bad_trials = []
    for i in range(f['R'].size):
        for key, value in trial_filters.items():
            if getattr(f['R'][i], key) == value:
                bad_trials.append(i)
    bad_trials = np.unique(bad_trials)
    print('%d Bad Trials being thrown away' % bad_trials.size)
    valid_trials = np.setdiff1d(np.arange(f['R'].size), bad_trials)

    # Filter out trialVersions and trialTypes not compliant
    trialVersion = np.array([f['R'][idx].trialVersion for idx in valid_trials])
    trialType = np.array([f['R'][idx].trialType for idx in valid_trials])

    if trialVersions != 'all':
        valid_trial_versions = set([idx for ii, idx in enumerate(valid_trials)
                                    if trialVersion[ii] in trialVersions])
    else:
        valid_trial_versions = set(valid_trials)
    if trialTypes != 'all':
        valid_trial_types = set([idx for ii, idx in enumerate(valid_trials)
                                 if trialType[ii] in trialTypes])
    else:
        valid_trial_types = set(valid_trials)

    valid_trials = np.array(list(set(valid_trials).intersection(valid_trial_versions).intersection(valid_trial_types)))    
    print('%d Trials selected' % valid_trials.size)

    # Timing information
    reveal_times = np.array([f['R'][idx].actualFlyAppears for idx in valid_trials])
    go_times = np.array([f['R'][idx].actualLandingTime for idx in valid_trials])
    reach_times = np.array([f['R'][idx].offlineMoveOnsetTime for idx in valid_trials])
    total_times = np.array([f['R'][idx].HAND.X.size for idx in valid_trials])

    # Neural data - filter by requested brain region
    n_units = f['R'][0].unit.size
    spike_times = []
    unit_lookup = ['PMD' if lookup == 1 else 'M1' for lookup in f['SU'].arrayLookup]
    for i in range(valid_trials.size):
        spike_times.append([])
        for j in range(len(unit_lookup)):
            if region == 'both' or unit_lookup[j] == region:
                if np.isscalar(f['R'][i].unit[j].spikeTimes):            
                    spike_times[i].append(np.array([f['R'][i].unit[j].spikeTimes]))
                else:
                    spike_times[i].append(np.array(f['R'][i].unit[j].spikeTimes))
    
    dat['spike_times'] = np.array(spike_times).astype(np.object)
    dat['reach_times'] = reach_times


    T  = tw[1] - tw[0]
    spike_rates = postprocess_spikes(window_spike_array(dat['spike_times'], dat['reach_times'] + tw[0], 
                                                        reach_times + tw[1]),
                                                        T, bin_width, boxcox, filter_fn, filter_kwargs, 
                                                        spike_threshold=spike_threshold, 
                                                        trial_threshold=trial_threshold)                      

    dat['spike_rates'] = spike_rates         


    #### Behavior ####
    handX = np.zeros(valid_trials.size).astype(np.object)
    handY = np.zeros(valid_trials.size).astype(np.object)

    for i in range(valid_trials.size):
        handX[i] = f['R'][i].HAND.X
        handY[i] = f['R'][i].HAND.Y


    dat['behavior'] = np.zeros((spike_rates.shape[0], 
                                spike_rates.shape[1], 2))

    for i in range(spike_rates.shape[0]):
        # Align behavioral variables to binned neural data
        hand = np.vstack([handX[i][dat['reach_times'][i] - int(T/2):dat['reach_times'][i] + int(T/2)],
                          handY[i][dat['reach_times'][i] - int(T/2):dat['reach_times'][i] + int(T/2)]]).T

        hand = align_behavior(hand, T, bin_width)
        hand -= hand.mean(axis=0, keepdims=True)
        hand /= hand.std(axis=0, keepdims=True)
        
        dat['behavior'][i, ...] = hand

    return dat

def load_sabes(filename, bin_width=50, boxcox=0.5, filter_fn='none', filter_kwargs={}, spike_threshold=100,
               std_behavior=False, region='M1', get_dof_only=False, **kwargs):

    # Convert bin width to s
    bin_width /= 1000

    # Load MATLAB file
    with h5py.File(filename, "r") as f:
        # Get channel names (e.g. M1 001 or S1 001)
        n_channels = f['chan_names'].shape[1]
        chan_names = []
        for i in range(n_channels):
            chan_names.append(f[f['chan_names'][0, i]][()].tobytes()[::2].decode())
        # Get M1 and S1 indices
        M1_indices = [i for i in range(n_channels) if chan_names[i].split(' ')[0] == 'M1']
        S1_indices = [i for i in range(n_channels) if chan_names[i].split(' ')[0] == 'S1']

        # Get time
        t = f['t'][0, :]
        # Individually process M1 and S1 indices
        dat = {}

        if region == 'M1':
            indices = M1_indices
        elif region == 'S1':
            indices = S1_indices
        elif region == 'both':
            indices = list(range(n_channels))

        # Perform binning
        n_channels = len(indices)
        n_sorted_units = f["spikes"].shape[0] - 1  # The FIRST one is the 'hash' -- ignore!
        n_units = n_channels * n_sorted_units
        max_t = t[-1]

        spike_times = np.zeros((n_sorted_units - 1, len(indices))).astype(np.object)


        for i, chan_idx in enumerate(indices):
            for unit_idx in range(1, n_sorted_units): # ignore hash
                spike_times_ = f[f["spikes"][unit_idx, chan_idx]][()]
                # Ignore this case (no data)
                if spike_times_.shape == (2,):
                    spike_times[unit_idx - 1, i] = np.nan
                else:
                    # offset spike times
                    spike_times[unit_idx - 1, i] = spike_times_[0, :] - t[0]

        # Reshape into format (ntrials, units)
        spike_times = spike_times.reshape((1, -1))
        # Total length of the time series
        T = t[-1] - t[0]
        spike_rates = postprocess_spikes(spike_times, T, bin_width, boxcox,
                                         filter_fn, filter_kwargs, spike_threshold, high_pass=True)

        dat['spike_rates'] = spike_rates

        # Get cursor position
        cursor_pos = f["cursor_pos"][:].T
        cursor_interp = align_behavior(cursor_pos, T, bin_width)
        if std_behavior:
            cursor_interp -= cursor_interp.mean(axis=0, keepdims=True)
            cursor_interp /= cursor_interp.std(axis=0, keepdims=True)

        dat["behavior"] = cursor_interp

        # Target position
        target_pos = f["target_pos"][:].T
        target_interp = align_behavior(target_pos, T, bin_width)
        # cursor_interp -= cursor_interp.mean(axis=0, keepdims=True)
        # cursor_interp /= cursor_interp.std(axis=0, keepdims=True)
        dat['target'] = target_interp

        dat['time'] = np.squeeze(align_behavior(t[:, np.newaxis], T, bin_width))

        return dat

def load_peanut_across_epochs(fpath, epochs, spike_threshold, **loader_kwargs):

    dat_allepochs = {}
    dat_per_epoch = []

    unit_ids = []

    for epoch in epochs:
        dat = load_peanut(fpath, epoch, spike_threshold, **loader_kwargs)
        unit_ids.append(set(dat['unit_ids']))
        dat_per_epoch.append(dat)

    unit_id_intersection = unit_ids[0]
    for i in range(1, len(epochs)):
        unit_id_intersection.intersection(unit_ids[i])

    for i, epoch in enumerate(epochs):
        dat = dat_per_epoch[i]
        unit_idxs = np.isin(dat['unit_ids'], np.array(list(unit_id_intersection)).astype(int)) 

def load_peanut(fpath, epoch, spike_threshold, bin_width=25, boxcox=0.5,
                filter_fn='none', speed_threshold=4, region='HPc', **filter_kwargs):
    '''
        Parameters:
            fpath: str
                 path to file
            epoch: list of ints
                which epochs (session) to load. The rat is sleeping during odd numbered epochs
            spike_threshold: int
                throw away neurons that spike less than the threshold during the epoch
            bin_width:  float 
                Bin width for binning spikes. Note the behavior is sampled at 25ms
            boxcox: float or None
                Apply boxcox transformation
            filter_fn: str
                Check filter_dict
            filter_kwargs
                keyword arguments for filter_fn
    '''

    data = pickle.load(open(fpath, 'rb'))
    dict_ = data['peanut_day14_epoch%d' % epoch]
    
    # Collect single units located in hippocampus

    HPc_probes = [key for key, value in dict_['identification']['nt_brain_region_dict'].items()
                  if value == 'HPc']

    OFC_probes = [key for key, value in dict_['identification']['nt_brain_region_dict'].items()
                  if value == 'OFC']

    if region == 'HPc':
        probes = HPc_probes
    elif region == 'OFC':
        probes = OFC_probes

    spike_times = []
    unit_ids = []
    for probe in dict_['spike_times'].keys():
        probe_id = probe.split('_')[-1]
        if probe_id in probes:
            for unit, times in dict_['spike_times'][probe].items():
                spike_times.append(list(times))
                unit_ids.append((probe_id, unit))
        else:
            continue


    # sort spike times
    spike_times = [list(np.sort(times)) for times in spike_times]

    # Apply spike threshold

    spike_threshold_filter = [idx for idx in range(len(spike_times))
                              if len(spike_times[idx]) > spike_threshold]
    spike_times = np.array(spike_times, dtype=object)
    spike_times = spike_times[spike_threshold_filter]
    unit_ids = np.array(unit_ids)[spike_threshold_filter]

    t = dict_['position_df']['time'].values
    T = t[-1] - t[0] 
    # Convert bin width to s
    bin_width = bin_width/1000
    
    # covnert smoothin bandwidth to indices
    if filter_fn == 'gaussian':
        filter_kwargs['sigma'] /= bin_width
        filter_kwargs['sigma'] = min(1, filter_kwargs['sigma'])
    
    bins = np.linspace(0, T, int(T//bin_width))

    spike_rates = np.zeros((bins.size - 1, len(spike_times)))
    for i in range(len(spike_times)):
        # translate to 0
        spike_times[i] -= t[0]
        
        spike_counts = np.histogram(spike_times[i], bins=bins)[0]
        if boxcox is not None:
            spike_counts = np.array([(np.power(spike_count, boxcox) - 1)/boxcox
                                     for spike_count in spike_counts])
        spike_rates_ = FILTER_DICT[filter_fn](spike_counts.astype(np.float), **filter_kwargs)
        
        spike_rates[:, i] = spike_rates_
    
    # Align behavior with the binned spike rates
    pos_linear = dict_['position_df']['position_linear'].values
    pos_xy = np.array([dict_['position_df']['x-loess'], dict_['position_df']['y-loess']]).T
    pos_linear, taligned = align_peanut_behavior(t, pos_linear, bins)
    pos_xy, _ = align_peanut_behavior(t, pos_xy, bins)
    
    dat = {}
    dat['unit_ids'] = unit_ids
    # Apply movement threshold
    if speed_threshold is not None:
        vel = np.divide(np.diff(pos_linear), np.diff(taligned))
        # trim off first index to match lengths
        spike_rates = spike_rates[1:, ...]
        pos_linear = pos_linear[1:, ...]
        pos_xy = pos_xy[1:, ...]

        spike_rates = spike_rates[np.abs(vel) > speed_threshold]

        pos_linear = pos_linear[np.abs(vel) > speed_threshold]
        pos_xy = pos_xy[np.abs(vel) > speed_threshold]

    dat['unit_ids'] = unit_ids
    dat['spike_rates'] = spike_rates
    dat['behavior'] = pos_xy
    dat['behavior_linear'] = pos_linear[:, np.newaxis]
    dat['time'] = taligned
    return dat

##### Peanut Segmentation #####
def segment_peanut(dat, loc_file, epoch, box_size=20, start_index=0, return_maze_points=False):

    with open(loc_file, 'rb') as f:
        ldict = pickle.load(f)
        
    edgenames = ldict['peanut_day14_epoch2']['track_graph']['edges_ordered_list']
    nodes = ldict['peanut_day14_epoch%d' % epoch]['track_graph']['nodes']
    for key, value in nodes.items():
        nodes[key] = (value['x'], value['y'])
    endpoints = []
    lengths = []
    for edgename in edgenames:
        endpoints.append(ldict['peanut_day14_epoch%d' % epoch]['track_graph']['edges'][edgename]['endpoints'])
        lengths.append(ldict['peanut_day14_epoch%d' % epoch]['track_graph']['edges'][edgename]['length'])
        
    # pos = np.array([ldict['peanut_day14_epoch%d' % epoch]['position_input']['position_x'],
    #             ldict['peanut_day14_epoch%d' % epoch]['position_input']['position_y']]).T

    pos = dat['behavior']
    if epoch in [2, 6, 10, 14]:
        transition1 = find_transitions(pos, nodes, 'handle_well', 'left_well', 
                                                   ignore=['center_maze', 'left_corner'], box_size=box_size, start_index=start_index)
        transition2 = find_transitions(pos, nodes, 'handle_well', 'right_well',
                                                   ignore=['center_maze', 'right_corner'], box_size=box_size, start_index=start_index)
    elif epoch in [4, 8, 12, 16]:
        transition1 = find_transitions(pos, nodes, 'center_well', 'left_well', 
                                                   ignore=['center_maze', 'left_corner'], box_size=box_size, start_index=start_index)
        transition2 = find_transitions(pos, nodes, 'center_well', 'right_well',
                                                   ignore=['center_maze', 'right_corner'], box_size=box_size, start_index=start_index)
    if return_maze_points:
        return transition1, transition2, nodes, endpoints
    else:
        return transition1, transition2

def in_box(pos, node, box_size):
    box_points = [np.array(node) + box_size/2 * np.array([1, 1]), # Top right
                  np.array(node) + box_size/2 * np.array([1, -1]), # Bottom right
                  np.array(node) + box_size/2 * np.array([-1, 1]), # Top left
                  np.array(node) + box_size/2 * np.array([-1, -1])] # Bottom left

    in_xlim = np.bitwise_and(pos[:, 0] > box_points[-1][0], 
                             pos[:, 0] < box_points[0][0])
    in_ylim = np.bitwise_and(pos[:, 1] > box_points[-1][1], 
                             pos[:, 1] < box_points[0][1])    
    return np.bitwise_and(in_xlim, in_ylim)
    
def find_transitions(pos, nodes, start_node, end_node, ignore=['center_maze'],
                     box_size=20, start_index=1000):
    pos = pos[start_index:]
    
    in_node_boxes = {}
    for key, value in nodes.items():
        in_node_boxes[key] = in_box(pos, value, box_size)
        
    in_node_boxes_windows = {}
    for k in in_node_boxes.keys():
        in_node_boxes_windows[k] = [[i for i,value in it] 
                                    for key,it in 
                                    itertools.groupby(enumerate(in_node_boxes[k]), key=operator.itemgetter(True)) 
                                    if key != 0]

    # For each window of time that the rat is in the start node box, find which box it goes to next. If this
    # box matches the end_node, then add the intervening indices to the list of transitions
    transitions = []
    for start_windows in in_node_boxes_windows[start_node]:
        next_box_times = {}
        
        # When does the rat leave the start_node
        t0 = start_windows[-1]
        for key, windows in in_node_boxes_windows.items():
            window_times = np.array([time for window in windows for time in window])
            # what is the first time after t0 that the rat enters this node/box
            valid_window_times = window_times[window_times > t0]
            if len(valid_window_times) > 0:
                next_box_times[key] = window_times[window_times > t0][0]
            else:
                next_box_times[key] = np.inf

        # Order the well names by next_box_times
        node_names = list(next_box_times.keys())
        node_times = list(next_box_times.values())
        
        
        node_order = np.argsort(node_times)
        idx = 0
        # Find the first node that is not the start_node and is not in the list of nodes to ignore
        while (node_names[node_order[idx]] in ignore) or (node_names[node_order[idx]] == start_node):
            idx += 1

        if node_names[node_order[idx]] == end_node:
            # Make sure to translate by the start index
            transitions.append(np.arange(t0, node_times[node_order[idx]]) + start_index)
            
    return transitions

# Segment the time series and then use the linearized positions to calculate the occupancy normalized firing rates, binned by position
def location_bin_peanut(fpath, loc_file, epoch, spike_threshold=100, sigma = 2):

    # No temporal binning
    dat = load_peanut(fpath, epoch, spike_threshold=spike_threshold, bin_width=1, boxcox=None,
                      speed_threshold=0)

    transition1, transition2 = segment_peanut(dat, loc_file, epoch)
    occupation_normed_rates = []
    transition_bins = []
    for transition_ in [transition1, transition2]:

        # Concatenate position and indices
        pos = []
        indices = []
        for trans in transition_:
            pos.extend(list(dat['behavior_linear'][trans, 0]))
            indices.extend(trans)
        indices = np.array(indices)

        bins = np.linspace(min(pos), max(pos), int((max(pos) - min(pos))/2))
        transition_bins.append(bins)
        # Histogram the linearized positions into the bins
        occupation_counts, _, idxs = binned_statistic(pos, pos, statistic='count', bins=bins)

        # Sum up spike counts
        binned_spike_counts = np.zeros((len(occupation_counts), dat['spike_rates'].shape[1]))
        for j in range(len(occupation_counts)):
            bin_idxs = indices[np.where(idxs==j + 1)[0]]
            for k in range(dat['spike_rates'].shape[1]):
                binned_spike_counts[j, k] = np.sum(dat['spike_rates'][bin_idxs, k])  

        # Smooth occupation_cnts and binned_spike_counts by a Gaussian filter of width 2 indices (4 cm)
        smooth_occupation_counts = gaussian_filter1d(occupation_counts, sigma=sigma)
        smooth_binned_rates = gaussian_filter1d(binned_spike_counts, sigma=sigma, axis=0)
        # Normalize binned_spike_counts by occupation_counts
        smooth_binned_rates = np.divide(smooth_binned_rates, smooth_occupation_counts[:, np.newaxis])

        # Set units to hertz
        dt = dat['time'][1] - dat['time'][0]
        smooth_binned_rates /= dt
        occupation_normed_rates.append(smooth_binned_rates)

    return occupation_normed_rates, transition_bins

def load_shenoy_large(path, bin_width=50, boxcox=0.5, trialize=False, filter_fn='none', filter_kwargs={}, spike_threshold=100,
                      trial_threshold=0.5, std_behavior=False, location='M1', interval='full'):

    # Convert bin width to s
    bin_width /= 1000

    io = NWBHDF5IO(path, 'r')
    nwbfile_in = io.read()

    # Get successful trial indices
    valid_trials = np.nonzero(nwbfile_in.trials.is_successful[:])[0]

    # Need to restrict to trials where there is a non-zero delay period prior to go cue
    if interval == 'before_go':
        valid_trials_ = []
        for trial in valid_trials:
            if nwbfile_in.trials.go_cue_time[trial] - nwbfile_in.trials.start_time[trial] > 2 * bin_width:
                valid_trials_.append(trial)
        valid_trials = np.array(valid_trials_)

    print('%d valid trials' % len(valid_trials))

    # Get index of electrodes located in the desired area
    loc_dict = {'M1':'M1 Motor Cortex', 'PMC': 'Pre-Motor Cortex, dorsal'}
    valid_units = []
    for i, loc in enumerate(nwbfile_in.electrodes.location):
        if loc == loc_dict[location]:
            valid_units.append(i)

    print('%d valid  units' % len(valid_units))

    raw_spike_times = np.array(nwbfile_in.units.spike_times_index)
    if trialize:

        # Trialize spike_times
        spike_times = np.zeros((len(valid_trials), len(valid_units)), dtype=np.object)
        T = np.zeros((valid_trials.size, 2))
        print('Trializing spike times')
        for j, unit in tqdm(enumerate(valid_units)):
            end_idx = 0
            for i, trial in enumerate(valid_trials):
                if interval == 'full':
                    T[i, 0] = nwbfile_in.trials.start_time[trial]
                    T[i, 1] = nwbfile_in.trials.stop_time[trial]
                elif interval == 'before_go':
                    T[i, 0] = nwbfile_in.trials.start_time[trial]
                    T[i, 1] = nwbfile_in.trials.go_cue_time[trial]
                elif interval == 'after_go':
                    T[i, 0] = nwbfile_in.trials.go_cue_time[trial]
                    T[i, 1] = nwbfile_in.trials.stop_time[trial]
                else:
                    raise ValueError('Invalid interval, please specify full, before_go, or after_go')
                windowed_spike_times, end_idx = window_spikes(raw_spike_times[unit], 
                                                            T[i][0], T[i][1], end_idx)
                spike_times[i, j] = windowed_spike_times

        T = np.squeeze(np.diff(T, axis=1))
    else:
        spike_times = np.zeros((1, len(valid_units)), dtype=np.object)
        for j, unit in enumerate(valid_units):
            spike_times[0, j] = raw_spike_times[unit]
        T = nwbfile_in.units.obs_intervals[0][1]

    # Filter spikes
    spike_rates = postprocess_spikes(spike_times, T, bin_width, boxcox, filter_fn, filter_kwargs,
                                     spike_threshold)

    dat = {}

    if trialize:
        dat['target_pos'] = np.array([nwbfile_in.trials.target_pos_index[trial] for trial in valid_trials])
    else:
        dat['target_pos'] = np.array([nwbfile_in.trials.target_pos_index])
    dat['spike_rates'] = spike_rates

    # Return go_cue_times relative to start_time
    dat['go_times'] = nwbfile_in.trials.go_cue_time[valid_trials] - nwbfile_in.trials.start_time[valid_trials]

    t = nwbfile_in.processing['behavior'].data_interfaces['Position'].spatial_series['Cursor'].timestamps

    if trialize:
        # Trialize behavior
        cursor = np.zeros(len(valid_trials), dtype=np.object) 
        hand = np.zeros(len(valid_trials), dtype=np.object) 

        t = nwbfile_in.processing['behavior'].data_interfaces['Position'].spatial_series['Cursor'].timestamps

        print('Trializing Behavior')
        for i, trial in tqdm(enumerate(valid_trials)):
            if interval == 'full':
                start_index = np.argmax(t > nwbfile_in.trials.start_time[trial])
                end_index = np.argmin(t < nwbfile_in.trials.stop_time[trial])
            elif interval == 'before_go':
                start_index = np.argmax(t > nwbfile_in.trials.start_time[trial])
                end_index = np.argmin(t < nwbfile_in.trials.go_cue_time[trial])
            elif interval == 'after_go':
                start_index = np.argmax(t > nwbfile_in.trials.go_cue_time[trial])
                end_index = np.argmin(t < nwbfile_in.trials.stop_time[trial])
            else:
                raise ValueError('Invalid interval, please specify full, before_go, or after_go')
            cursor[i] = nwbfile_in.processing['behavior'].data_interfaces['Position'].spatial_series['Cursor'].data[start_index:end_index]
            hand[i] = nwbfile_in.processing['behavior'].data_interfaces['Position'].spatial_series['Hand'].data[start_index:end_index]

        # Align behavior    
        cursor_interp = np.array([align_behavior(c, T[i], bin_width) for i, c in enumerate(cursor)], dtype=np.object)
        hand_interp = np.array([align_behavior(h, T[i], bin_width) for i, h in enumerate(hand)], dtype=np.object)

    else:
        cursor = nwbfile_in.processing['behavior'].data_interfaces['Position'].spatial_series['Cursor'].data
        hand = nwbfile_in.processing['behavior'].data_interfaces['Position'].spatial_series['Hand'].data

        cursor_interp = align_behavior(cursor, T, bin_width)
        hand_interp = align_behavior(cursor, T, bin_width)

    dat['behavior'] = cursor_interp
    dat['behavior_3D'] = hand_interp

    io.close()

    return dat
