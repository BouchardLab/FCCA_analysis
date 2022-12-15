import pdb
import argparse
import numpy as np
import matplotlib.pyplot as plt
import scipy 
import sys
import time
import glob
import pickle
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LinearRegression

import itertools
from sklearn.model_selection import KFold

sys.path.append('/home/akumar/nse/neural_control')
from utils import apply_df_filters, calc_loadings
from loaders import load_sabes
from segmentation import reach_segment_sabes
from decoders import lr_decode_windowed, lr_bv_windowed

from mpi4py import MPI

start_times = {'indy_20160426_01': 0,
               'indy_20160622_01':1700,
               'indy_20160624_03': 500,
               'indy_20160627_01': 0,
               'indy_20160630_01': 0,
               'indy_20160915_01': 0,
               'indy_20160921_01': 0,
               'indy_20160930_02': 0,
               'indy_20160930_05': 300,
               'indy_20161005_06': 0,
               'indy_20161006_02': 350,
               'indy_20161007_02': 950,
               'indy_20161011_03': 0,
               'indy_20161013_03': 0,
               'indy_20161014_04': 0,
               'indy_20161017_02': 0,
               'indy_20161024_03': 0,
               'indy_20161025_04': 0,
               'indy_20161026_03': 0,
               'indy_20161027_03': 500,
               'indy_20161206_02': 5500,
               'indy_20161207_02': 0,
               'indy_20161212_02': 0,
               'indy_20161220_02': 0,
               'indy_20170123_02': 0,
               'indy_20170124_01': 0,
               'indy_20170127_03': 0,
               'indy_20170131_02': 0,
               'loco_20170210_03':0, 
               'loco_20170213_02':0, 
               'loco_20170214_02':0, 
               'loco_20170215_02':0, 
               'loco_20170216_02': 0, 
               'loco_20170217_02': 0, 
               'loco_20170227_04': 0, 
               'loco_20170228_02': 0, 
               'loco_20170301_05':0, 
               'loco_20170302_02':0}


def gen_run(name, didxs=np.arange(35)):
    with open(name, 'w') as rsh:
        rsh.write('#!/bin/bash\n')
        for d in didxs:
            rsh.write('mpirun -n 8 python biasvariance_vst.py %d\n' % d)

def reach_segment(data_file)    :
    dat = load_sabes(data_file)
    start_time = start_times[data_file.split('/')[-1].split('.mat')[0]]

    target_locs = []
    time_on_target = []
    valid_transition_times = []

    target_diff = np.diff(dat['target'].T)
    # This will yield the last index before the transition
    transition_times = np.sort(np.unique(target_diff.nonzero()[1]))
    #transition_times = target_diff.nonzero()[1]

    # For each transition, make a record of the location, time on target, and transition_vector
    # Throw away those targets that only appear for 1 timestep
    for i, transition_time in enumerate(transition_times):

        # Only lingers at the target for one timestep
        if i < len(transition_times) - 1:
            if np.diff(transition_times)[i] == 1:
                continue

        target_locs.append(dat['target'][transition_time][:])
        valid_transition_times.append(transition_time)
        
    for i, transition_time in enumerate(valid_transition_times):
            
        if i == 0:
            time_on_target.append(transition_time + 1)
        else:
            time_on_target.append(transition_time - valid_transition_times[i - 1] + 1)
            
    target_locs = np.array(target_locs)
    time_on_target = np.array(time_on_target)
    valid_transition_times = np.array(valid_transition_times)

    # Filter out by when motion starts
    if start_time > valid_transition_times[0]:
        init_target_loc = target_locs[valid_transition_times < start_time][-1]
    else:
        init_target_loc = target_locs[0]

    target_locs = target_locs[valid_transition_times > start_time]
    time_on_target = time_on_target[valid_transition_times > start_time]
    valid_transition_times = valid_transition_times[valid_transition_times > start_time]

    target_pairs = []
    for i in range(1, len(target_locs)):
        target_pairs.append((i - 1, i))

    # Velocity profiles
    vel = np.diff(dat['behavior'], axis=0)

    # Pair of target corrdinates
    valid_target_pairs = []
    # Tuple of indices that describes start and end of reach
    transition_times = []
    transition_vectors = []

    valid_target_pairs = [(target_locs[target_pairs[i][0]], target_locs[target_pairs[i][1]]) for i in range(len(target_pairs))]
    transition_times = [(valid_transition_times[target_pairs[i][0]] + 1, valid_transition_times[target_pairs[i][1]]) for i in range(len(target_pairs))]
    transition_vectors = [target_locs[target_pairs[i][1]] - target_locs[target_pairs[i][0]] for i in range(len(target_pairs))]

    velocity = scipy.ndimage.gaussian_filter1d(vel, axis=0, sigma=1)
    velocity_seg = [np.linalg.norm(velocity[t[0]:t[1], :], axis=1) for t in transition_times]
    velocity_normseg = [v/np.max(v) for v in velocity_seg]
    npks = [len(scipy.signal.find_peaks(v, height=0.15)[0]) for v in velocity_normseg]

    straight_reach = np.argwhere(np.array(npks) == 1).squeeze()
    # exclude very multi peaked
    correction_reach = np.argwhere(np.bitwise_and(np.array(npks) > 1, np.array(npks) < 5)).squeeze()




    behavior = [dat['behavior'][t[0]:t[1]] for t in transition_times]
    centered_behavior = [b - b[0, :] for b in behavior]

    # Rotate by the transition vector
    theta = [np.arctan2((b[-1, :] - b[0, :])[1], (b[-1, :] - b[0, :])[0] ) for b in centered_behavior]

    R = lambda theta: np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    centrot_behavior = [b @ R(th) for (b, th) in zip(centered_behavior, theta)]

    return dat, transition_times, straight_reach, correction_reach, velocity_seg, centrot_behavior


def get_peak_assignments(vel, dtpkl):

    pkassign = []
    for j, v in enumerate(vel):
        if np.isnan(dtpkl[j]):
            pkassign.append(np.zeros(v.size))
        else:
            pka = np.zeros(v.size)
            pka[dtpkl[j] -1:] = 1
            pkassign.append(pka)

    return np.array(pkassign)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('didx', type=int)
    data_path = '/mnt/Secondary/data/sabes'

    args = parser.parse_args()    
    didx = args.didx
    comm = MPI.COMM_WORLD

    #dimvals = np.array([2, 6, 10, 15])
    # Fix dimension to 6
    dimval = 6
    measure_from_end=False

    lag = 2
    decoding_window = 5

    # Sliding windows
    window_width = 2
    #window_centers = np.linspace(0, 35, 25)[0:9]
    window_centers = np.arange(-5, 25)
    windows = [(int(wc - window_width//2), int(wc + window_width//2)) for wc in window_centers]

    if comm.rank == 0:
        # Load indy, sabes dataframes
        with open('/mnt/Secondary/data/postprocessed/indy_dimreduc_nocv.dat', 'rb') as f:
            indy_df = pickle.load(f)
        for f in indy_df:
            f['data_file'] = f['data_file'].split('/')[-1]

        indy_df = pd.DataFrame(indy_df)


        with open('/mnt/Secondary/data/postprocessed/loco_dimreduc_nocv_df.dat', 'rb') as f:
            loco_df = pickle.load(f)
        loco_df = pd.DataFrame(loco_df)
        good_loco_files = ['loco_20170210_03.mat',
        'loco_20170213_02.mat',
        'loco_20170215_02.mat',
        'loco_20170227_04.mat',
        'loco_20170228_02.mat',
        'loco_20170301_05.mat',
        'loco_20170302_02.mat']

        loco_df = apply_df_filters(loco_df, data_file=good_loco_files,   
                                   loader_args={'bin_width': 50, 'filter_fn': 'none', 'filter_kwargs': {}, 'boxcox': 0.5, 'spike_threshold': 100, 'region': 'M1'})

        sabes_df = pd.concat([indy_df, loco_df])

        data_files = np.unique(sabes_df['data_file'].values)
        data_file = data_files[didx]
        print(data_file)
        dffca = apply_df_filters(sabes_df, data_file=data_file, dim=dimval, dimreduc_method='LQGCA')
        dfpca = apply_df_filters(sabes_df, data_file=data_file, dim=dimval, dimreduc_method='PCA')

        try:
            assert(dffca.shape[0] == 1)
            assert(dfpca.shape[0] == 1)
        except:
            pdb.set_trace()        

        coefpca = dfpca.iloc[0]['coef'][:, 0:dimval]
        coeffcca = dffca.iloc[0]['coef'][:, 0:dimval]

        dat = load_sabes('%s/%s' % (data_path, data_file))
        # Note the lower error threshold
        dat = reach_segment_sabes(dat, data_file=data_file.split('.mat')[0], err_thresh=0.9)

        # Measure the distance from target
        # Calculate 
        # (1) The peaks in distance to target
        # (2) troughs in velocity
        # (3) Number of velocity peaks/velocity troughs
        dt = []
        dtpks = []
        dtpkl = []

        # Intersection
        transition_times = np.array(dat['transition_times'])
        for j, tt in enumerate(transition_times):        
            target_loc = dat['target_pairs'][j][1]

            dt_ = np.linalg.norm(dat['behavior'][tt[0]:tt[1]] - dat['target_pairs'][j][1], axis=1)

            dt.append(dt_)
            
            pks, _ = scipy.signal.find_peaks(dt_/np.max(dt_), height=0.1, prominence=0.1)

            # Require that the peak comes after the maximum value
            pks = pks[pks > np.argmax(dt_)]
            # Require that we have gotten at least halfway to the target, but not too close
            if len(pks) > 0:
                if np.any((dt_/np.max(dt_))[:pks[0]] < 0.5) and not np.any((dt_/np.max(dt_))[:pks[0]] < 0.1):
                    # Get the FWHM of the peak widths
                    w, _, l, r = scipy.signal.peak_widths(dt_/np.max(dt_), [pks[0]], rel_height=0.5)
                    dtpkl.append(int(np.floor(l[0])))
                else:
                    pks = []
                    dtpkl.append(np.nan)
            else:
                dtpkl.append(np.nan)
            
            dtpks.append(pks)

        Z = dat['behavior'].squeeze()
        X = dat['spike_rates'].squeeze()

        # Apply lag
        X = X[lag:, :]
        Z = Z[:-lag, :]
        velocity = np.diff(Z, axis=0)

        # Exclude any reaches that lie within +/- lag of the start/end of the session
        too_soon = [j for j in range(len(transition_times)) if transition_times[j][0] < lag]
        too_late = [j for j in range(len(transition_times)) if transition_times[j][1] > dat['behavior'].shape[0] - lag]

        # Straight/Direct vs. Corrective reaches
        straight_reach = [idx for idx in range(len(dt)) if len(dtpks[idx]) == 0]
        correction_reach = [idx for idx in range(len(dt)) if len(dtpks[idx]) > 0]

        for idx in too_soon:
            if idx in straight_reach:
                straight_reach.remove(idx)
            elif idx in correction_reach:
                correction_reach.remove(idx)
        for idx in too_late:
            if idx in straight_reach:
                straight_reach.remove(idx)
            elif idx in correction_reach:
                correction_reach.remove(idx)

        velocity_seg = [np.linalg.norm(velocity[t[0]:t[1], :], axis=1) for t in transition_times]

        # Segment the corrective reaches by pre/post corrective movement
        pkassign = get_peak_assignments(velocity_seg, dtpkl)

        # For corrective reaches, add an offset so windows are measured with respect to the first minimum in dt
        offsets = np.zeros(len(transition_times))

        for idx in correction_reach:
            dt_ = dt[idx]
            # Normalize by max
            dt_ /= np.max(dt_)

            dt_0 = dt_[:dtpks[idx][0]]
            
            # Steepest decline
            dt_00 = dt_0[np.argmin(np.diff(dt_0)):]
            zero = np.argmin(dt_00) + np.argmin(np.diff(dt_0))

            offsets[idx] = zero

    else:
        dat = None
        data_files = None
        coefpca = None
        coeffcca = None
        transition_times = None
        straight_reach = None
        correction_reach = None
        offsets = None
        pkassign = None
        X = None
        Z = None

    coefpca = comm.bcast(coefpca)
    coeffcca = comm.bcast(coeffcca)
    transition_times = comm.bcast(transition_times)
    straight_reach = comm.bcast(straight_reach)
    correction_reach = comm.bcast(correction_reach)
    pkassign = comm.bcast(pkassign)
    offsets = comm.bcast(offsets)

    X = comm.bcast(X)
    Z = comm.bcast(Z)

    # Distribute windows across ranks
    windows = np.array_split(windows, comm.size)[comm.rank]

    bias = np.zeros((len(windows), 8, 6))
    var = np.zeros((len(windows), 8, 6))
    mse = np.zeros((len(windows), 8, 6))
    wr2 = np.zeros((len(windows), 8, 6))

    print('%d straight, %d correction' % (len(straight_reach), len(correction_reach)))


    # Cross-validate the prediction
    for j, window in enumerate(windows):
        Xpca = X @ coefpca
        Xlqg = X @ coeffcca

        ################################## First, we train on single peak reaches, and test on the latter portion of multipeak reaches
        

        # Feed into lr_decoder. Use lag of 0 since we already applied, but feed in the decoding window
        mse_, bias_, var_ =  lr_bv_windowed(Xpca, Z, 0, window, transition_times, straight_reach, correction_reach, pkassign, decoding_window=decoding_window, n_boots=200, random_seed=500, offsets=offsets)
        bias[j, 0, :] = bias_
        var[j, 0, :] = var_
        mse[j, 0, :] = mse_

        mse_, bias_, var_ =  lr_bv_windowed(Xlqg, Z, 0, window, transition_times, straight_reach, correction_reach, pkassign, decoding_window=decoding_window, n_boots=200, random_seed=500, offsets=offsets)
        bias[j, 1, :] = bias_
        var[j, 1, :] = var_
        mse[j, 1, :] = mse_

        # Also keep track of the r2

        r2pos, r2vel, r2acc, r2post, r2velt, r2acct, msetr, msete,  _ = lr_decode_windowed(Xpca, Z, lag, window, transition_times, train_idxs=straight_reach,
                                                                                            test_idxs=correction_reach, decoding_window=decoding_window, measure_from_end=measure_from_end,
                                                                                            pkassign=pkassign, offsets=offsets) 
        wr2[j, 0, :] = (r2pos, r2vel, r2acc, r2post, r2velt, r2acct)
        r2pos, r2vel, r2acc, r2post, r2velt, r2acct, msetr, msete,  _ = lr_decode_windowed(Xlqg, Z, lag, window, transition_times, train_idxs=straight_reach,
                                                                                        test_idxs=correction_reach, decoding_window=decoding_window, measure_from_end=measure_from_end,
                                                                                        pkassign=pkassign, offsets=offsets)
        wr2[j, 1, :] = (r2pos, r2vel, r2acc, r2post, r2velt, r2acct)

        ############################################################# Second, we train on both single and multi peak reaches and test on the latter half of multi peak
        # Feed into lr_decoder. Use lag of 0 since we already applied, but feed in the decoding window
        mse_, bias_, var_ =  lr_bv_windowed(Xpca, Z, 0, window, transition_times, np.sort(np.concatenate([straight_reach, correction_reach])), 
                                            correction_reach, pkassign, decoding_window=decoding_window, n_boots=200, random_seed=500, apply_pk_to_train=True, offsets=offsets)
        bias[j, 2, :] = bias_
        var[j, 2, :] = var_
        mse[j, 2, :] = mse_

        mse_, bias_, var_ =  lr_bv_windowed(Xlqg, Z, 0, window, transition_times, np.sort(np.concatenate([straight_reach, correction_reach])), 
                                            correction_reach, pkassign, decoding_window=decoding_window, n_boots=200, random_seed=500, apply_pk_to_train=True, offsets=offsets)
        bias[j, 3, :] = bias_
        var[j, 3, :] = var_
        mse[j, 3, :] = mse_

        # Also keep track of the r2

        r2pos, r2vel, r2acc, r2post, r2velt, r2acct, msetr, msete,  _ = lr_decode_windowed(Xpca, Z, lag, window, transition_times, train_idxs=np.sort(np.concatenate([straight_reach, correction_reach])),
                                                                                            test_idxs=correction_reach, decoding_window=decoding_window, measure_from_end=measure_from_end,
                                                                                            pkassign=pkassign,  apply_pk_to_train=True, offsets=offsets) 
        wr2[j, 2, :] = (r2pos, r2vel, r2acc, r2post, r2velt, r2acct)
        
        r2pos, r2vel, r2acc, r2post, r2velt, r2acct, msetr, msete,  _ = lr_decode_windowed(Xlqg, Z, lag, window, transition_times, train_idxs=np.sort(np.concatenate([straight_reach, correction_reach])),
                                                                                        test_idxs=correction_reach, decoding_window=decoding_window, measure_from_end=measure_from_end,
                                                                                        pkassign=pkassign,  apply_pk_to_train=True, offsets=offsets)
        wr2[j, 3, :] = (r2pos, r2vel, r2acc, r2post, r2velt, r2acct)

        ############################################################## Third, we train on multi peak only during the first portion and test on the latter half
        # Feed into lr_decoder. Use lag of 0 since we already applied, but feed in the decoding window
        mse_, bias_, var_ =  lr_bv_windowed(Xpca, Z, 0, window, transition_times, correction_reach, 
                                            correction_reach, pkassign, decoding_window=decoding_window, n_boots=200, random_seed=500,  apply_pk_to_train=True, offsets=offsets)
        bias[j, 4, :] = bias_
        var[j, 4, :] = var_
        mse[j, 4, :] = mse_

        mse_, bias_, var_ =  lr_bv_windowed(Xlqg, Z, 0, window, transition_times, correction_reach, 
                                            correction_reach, pkassign, decoding_window=decoding_window, n_boots=200, random_seed=500,  apply_pk_to_train=True, offsets=offsets)
        bias[j, 5, :] = bias_
        var[j, 5, :] = var_
        mse[j, 5, :] = mse_

        # Also keep track of the r2

        r2pos, r2vel, r2acc, r2post, r2velt, r2acct, msetr, msete,  _ = lr_decode_windowed(Xpca, Z, lag, window, transition_times, train_idxs=correction_reach,
                                                                                            test_idxs=correction_reach, decoding_window=decoding_window, measure_from_end=measure_from_end,
                                                                                            pkassign=pkassign,  apply_pk_to_train=True, offsets=offsets) 
        wr2[j, 4, :] = (r2pos, r2vel, r2acc, r2post, r2velt, r2acct)
        r2pos, r2vel, r2acc, r2post, r2velt, r2acct, msetr, msete,  _ = lr_decode_windowed(Xlqg, Z, lag, window, transition_times, train_idxs=correction_reach,
                                                                                        test_idxs=correction_reach, decoding_window=decoding_window, measure_from_end=measure_from_end,
                                                                                        pkassign=pkassign,  apply_pk_to_train=True, offsets=offsets)
        wr2[j, 5, :] = (r2pos, r2vel, r2acc, r2post, r2velt, r2acct)

        ############################################################## Third, we train on multi peak only during the first portion and test on the latter half
        # Feed into lr_decoder. Use lag of 0 since we already applied, but feed in the decoding window
        # Lastly, train on t
        mse_, bias_, var_ =  lr_bv_windowed(Xpca, Z, 0, window, transition_times, straight_reach, 
                                            correction_reach, pkassign=None, decoding_window=decoding_window, n_boots=200, random_seed=500, offsets=offsets)
        bias[j, 6, :] = bias_
        var[j, 6, :] = var_
        mse[j, 6, :] = mse_

        mse_, bias_, var_ =  lr_bv_windowed(Xlqg, Z, 0, window, transition_times, straight_reach, 
                                            correction_reach, pkassign=None, decoding_window=decoding_window, n_boots=200, random_seed=500, offsets=offsets)
        bias[j, 7, :] = bias_
        var[j, 7, :] = var_
        mse[j, 7, :] = mse_

        # Also keep track of the r2

        r2pos, r2vel, r2acc, r2post, r2velt, r2acct, msetr, msete,  _ = lr_decode_windowed(Xpca, Z, lag, window, transition_times, train_idxs=straight_reach,
                                                                                            test_idxs=correction_reach, decoding_window=decoding_window, measure_from_end=measure_from_end,
                                                                                            pkassign=None, offsets=offsets) 
        wr2[j, 6, :] = (r2pos, r2vel, r2acc, r2post, r2velt, r2acct)
        r2pos, r2vel, r2acc, r2post, r2velt, r2acct, msetr, msete,  _ = lr_decode_windowed(Xlqg, Z, lag, window, transition_times, train_idxs=straight_reach,
                                                                                        test_idxs=correction_reach, decoding_window=decoding_window, measure_from_end=measure_from_end,
                                                                                        pkassign=None, offsets=offsets)
        wr2[j, 7, :] = (r2pos, r2vel, r2acc, r2post, r2velt, r2acct)

    windows = np.array(windows)
    dpath = '/home/akumar/nse/neural_control/data/biasvariance_vst4'
    #dpath = '/mnt/sdb1/nc_data/decodingvt'
    with open('%s/didx%d_rank%d.dat' % (dpath, didx, comm.rank), 'wb') as f:
        f.write(pickle.dumps(bias))
        f.write(pickle.dumps(var))
        f.write(pickle.dumps(mse))
        f.write(pickle.dumps(wr2))
        f.write(pickle.dumps(windows))
        f.write(pickle.dumps(offsets))
        f.write(pickle.dumps(dimval))