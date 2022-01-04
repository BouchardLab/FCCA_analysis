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

from mpi4py import MPI
from mpi_utils.ndarray import Gatherv_rows

from loaders import window_spikes, align_behavior

FILTER_DICT = {'gaussian':gaussian_filter1d, 'none': lambda x, **kwargs: x}

# spike_times: (n_trial, n_neurons)
#  trial threshold: If we require a spike threshold, trial threshold = 1 requires 
#  the spike threshold to hold for the neuron for all trials. 0 would mean no trials
def mpi_postprocess_spikes(spike_times, T, bin_width, boxcox, filter_fn, filter_kwargs,
                           spike_threshold=0, trial_threshold=1, high_pass=False, comm=None):

    # Discretize time over bins
    bins = np.linspace(0, T, int(T//bin_width))

    # Did the trial/unit have enough spikes?

    print('Checking spike threshold')

    # Assess spike count sufficiency
    if comm.rank == 0:
        insufficient_spikes = np.zeros(spike_times.shape)
        if spike_threshold is not None:
            for i in tqdm(range(spike_times[0:2].shape[0])):
                for j in range(spike_times.shape[1]):    
                    # Ignore this trial/unit combo
                    if np.any(np.isnan(spike_times[i, j])):
                        insufficient_spikes[i, j] = 1          

                    spike_counts = np.histogram(spike_times[i, j], bins=bins)[0]

                    if spike_threshold is not None:
                        if np.sum(spike_counts) <= spike_threshold:
                            insufficient_spikes[i, j] = 1

    print('Filtering spikes')

    trial_indices = np.array_split(np.arange(spike_times.shape[0]), comm.size)
    spike_rates = np.zeros((len(trial_indices[comm.rank]), bins.size - 1, 
                           spike_times.shape[1]))

    for i, trial_index in tqdm(enumerate(trial_indices[comm.rank]), total=len(trial_indices[comm.rank])):
        for j in range(spike_times.shape[1]):    
            # Apply a boxcox transformation
            if boxcox is not None:
                spike_counts = np.histogram(spike_times[trial_index, j], bins=bins)[0]
                spike_counts = np.array([(np.power(spike_count, boxcox) - 1)/boxcox 
                                         for spike_count in spike_counts])

            # Filter the resulting spike counts
            spike_rates_ = FILTER_DICT[filter_fn](spike_counts.astype(np.float), **filter_kwargs)

            # High pass to remove long term trends (needed for sabes data)
            if high_pass:
                spike_rates_ = moving_center(spike_rates_, 600)

            spike_rates[i, :, j] = spike_rates_

    spike_rates = np.ascontiguousarray(spike_rates)

    # Gather spikes
    spike_rates = Gatherv_rows(spike_rates, comm)

    if comm.rank == 0:
        # Filter out bad units
        sufficient_spikes = np.arange(spike_times.shape[1])[np.sum(insufficient_spikes, axis=0) < \
                                                            (1 - (trial_threshold -1e-3)) * spike_times.shape[0]]
        spike_rates = spike_rates[..., list(sufficient_spikes)]
    else:
        spike_rates = None

    return spike_rates

# Load in parallel by 
def mpi_load_shenoy(comm, data_path, bin_width, boxcox, filter_fn, filter_kwargs, 
                    spike_threshold=None, trial_threshold=0.5, tw=(-250, 550), 
                    trialVersions='all', trialTypes='all', region='both'):

    # Code checks for list membership in specified trialtypes/trialversions
    if trialVersions != 'all' and type(trialVersions) != list:
        trialVersions = [trialVersions]
    if trialTypes != 'all' and type(trialTypes) != list:
        trialTypes = [trialTypes]

    dat = {}
    if comm.rank == 0:

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
        
        spike_times = np.array(spike_times).astype(np.object)
        dat['spike_times'] = spike_times
        dat['reach_times'] = reach_times

    else:
        spike_times = None
        reach_times = None

    spike_times = comm.bcast(spike_times)
    reach_times = comm.bcast(reach_times)

    T  = tw[1] - tw[0]
    spike_rates = mpi_postprocess_spikes(window_spikes(spike_times, reach_times + tw[0], 
                                                       reach_times + tw[1]),
                                         T, bin_width, boxcox, filter_fn, filter_kwargs, 
                                         spike_threshold=spike_threshold, 
                                         trial_threshold=trial_threshold, comm=comm)                      

    dat['spike_rates'] = spike_rates         
    if comm.rank == 0:



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
