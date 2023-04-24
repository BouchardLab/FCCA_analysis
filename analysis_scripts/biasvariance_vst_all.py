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
from decoders import lr_decode_windowed, lr_bv_windowed, expand_state_space

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


def get_peak_assignments_vel(velocity_seg):

    # Get the width of each peaks, exactly partitioning the time series
    velnorm = [v/np.max(v) for v in velocity_seg]    
    peak_indices = [scipy.signal.find_peaks(v, height=0.4)[0] for v in velnorm]
    peak_widths = [scipy.signal.peak_widths(v, peaks=pkidxs, rel_height=1.0) for v, pkidxs in zip(velnorm, peak_indices)]

    # Assign points to the closest peak
    def closest_peak(pks, pnt):
        pk_dist = [np.abs(pk - pnt) for pk in pks]
        if len(pk_dist) > 0:
            return np.argmin(pk_dist)
        else:
            # Shouldn't be used as these fall outside of both single peak and multi peak reaches
            return np.nan

    pkassign = [np.array([closest_peak(peak_indices[j], t) for t in range(len(v))]) for j, v in enumerate(velnorm)]

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
    window_centers = np.arange(-5, 35)

    # Pool together n decoding windows prior for training 
    train_windows = [[(int(wc - window_width//2), int(wc + window_width//2))] for wc in window_centers]
    test_windows = [[(int(wc - window_width//2), int(wc + window_width//2))] for wc in window_centers]

    if comm.rank == 0:
        # with open('/home/akumar/nse/neural_control/data/indy_dimreduc_marginal_nocv.dat', 'rb') as f:
        #     sabes_df = pickle.load(f)
        # with open('/mnt/Secondary/data/postprocessed/indy_dimreduc_nocv.dat', 'rb') as f:
        #     sabes_df = pickle.load(f)
        with open('/mnt/Secondary/data/postprocessed/indy_decoding_df2.dat', 'rb') as f:
            indy_df = pickle.load(f)
        indy_df = pd.DataFrame(indy_df)

        with open('/mnt/Secondary/data/postprocessed/loco_decoding_df.dat', 'rb') as f:
            loco_df = pickle.load(f)
        loco_df = pd.DataFrame(loco_df)
        loco_df = apply_df_filters(loco_df,
                                loader_args={'bin_width': 50, 'filter_fn': 'none', 'filter_kwargs': {}, 'boxcox': 0.5, 'spike_threshold': 100, 'region': 'M1'},
                                decoder_args={'trainlag': 4, 'testlag': 4, 'decoding_window': 5})
        good_loco_files = ['loco_20170210_03.mat',
        'loco_20170213_02.mat',
        'loco_20170215_02.mat',
        'loco_20170227_04.mat',
        'loco_20170228_02.mat',
        'loco_20170301_05.mat',
        'loco_20170302_02.mat']

        loco_df = apply_df_filters(loco_df, data_file=good_loco_files)        
        
        # with open('/home/akumar/nse/neural_control/data/sabes_dimreduc_nocv.dat', 'rb') as f:
        #     sabes_df = pickle.load(f)

        sabes_df = pd.concat([indy_df, loco_df])

        data_files = np.unique(sabes_df['data_file'].values)
        data_file = data_files[didx]

        # df = apply_df_filters(sabes_df, data_file=data_file, dim=dimval)
        # assert(df.shape[0] == 1)
        # coefpca.append(df.iloc[0]['pcacoef'])
        # coeffcca.append(df.iloc[0]['lqgcoef'])            

        dffca = apply_df_filters(sabes_df, data_file=data_file, dim=dimval, dimreduc_method='LQGCA', dimreduc_args={'T':3, 'loss_type':'trace', 'n_init':10})
        dfpca = apply_df_filters(sabes_df, data_file=data_file, dim=dimval, dimreduc_method='PCA')

        assert(dffca.shape[0] == 5)
        assert(dfpca.shape[0] == 5)
        
        coefpca = [apply_df_filters(dfpca, fold_idx=k).iloc[0]['coef'][:, 0:dimval] for k in range(5)]
        coeffcca = [apply_df_filters(dffca, fold_idx=k).iloc[0]['coef'][:, 0:dimval] for k in range(5)]

        dat = load_sabes('/mnt/Secondary/data/sabes/%s' % data_file)
        # dat = load_sabes(data_file)
        data_file = data_file.split('/')[-1]
        # dat = load_sabes('/mnt/sdb1/nc_data/sabes/%s' % data_file)
        dat = reach_segment_sabes(dat, data_file=data_file.split('.mat')[0])
        X = np.squeeze(dat['spike_rates'])
        Z = dat['behavior']
        # transition_times = dat['transition_times']
        transition_times = np.array(dat['transition_times'])

        # We calculate velocity by using expand state space, and shift the transition times accordingly
        Z, _ = expand_state_space([dat['behavior']], [dat['spike_rates'].squeeze()], True, True)
        # Flatten list structure imposed by expand_state_space
        Z = Z[0]

        # Shift transition times by 2 due to expand state space
        transition_times = np.array([(t[0] - 2, t[1] - 2) for t in dat['transition_times']])

    else:
        dat = None
        data_files = None
        coefpca = None
        coeffcca = None
        X = None
        Z = None
        transition_times = None

    coefpca = comm.bcast(coefpca)
    coeffcca = comm.bcast(coeffcca)

    X = comm.bcast(X)
    Z = comm.bcast(Z)

    transition_times = comm.bcast(transition_times)

    # Distribute windows across ranks
    train_windows = np.array_split(train_windows, comm.size)[comm.rank]
    test_windows = np.array_split(test_windows, comm.size)[comm.rank]

    bias = np.zeros((len(test_windows), 2, 5, 6))
    var = np.zeros((len(test_windows), 2, 5, 6))
    mse = np.zeros((len(test_windows), 2, 5, 6))
    wr2 = np.zeros((len(test_windows), 2, 5, 6))
    ntr = np.zeros((len(test_windows), 2, 5, 2))

    for j, train_window in enumerate(train_windows):
         for fold, (train_idxs, test_idxs) in tqdm(enumerate(KFold(n_splits=5).split(Z))): 

            Xpca = X @ coefpca[fold]
            Xfcca = X @ coeffcca[fold]

            # Need to turn train/test_idxs returned by KFold into an indexing of the transition times
            tt_train_idxs = [idx for idx in range(len(transition_times)) if transition_times[idx][0] in train_idxs and transition_times[idx][1] in train_idxs]
            tt_test_idxs = [idx for idx in range(len(transition_times)) if transition_times[idx][0] in test_idxs and transition_times[idx][1] in test_idxs]

            # Train on all reaches, and test on all reaches. Here we set pkassign to 1 across the entire reach

            pkassign = None

            mse_, bias_, var_, ntr_ =  lr_bv_windowed(Xpca, Z, lag, train_window, test_windows[j], transition_times, tt_train_idxs, tt_test_idxs, 
                                                        pkassign, decoding_window=decoding_window, n_boots=200, random_seed=500, offsets=None, norm=False)

            bias[j, 0, fold, :] = bias_
            var[j, 0, fold, :] = var_
            mse[j, 0, fold, :] = mse_
            ntr[j, 0, fold, 0] = ntr_

            mse_, bias_, var_, ntr_ =  lr_bv_windowed(Xfcca, Z, lag, train_window, test_windows[j], transition_times, tt_train_idxs, tt_test_idxs, 
                                                pkassign, decoding_window=decoding_window, n_boots=200, random_seed=500, offsets=None, norm=False)
            bias[j, 1, fold, :] = bias_
            var[j, 1, fold, :] = var_
            mse[j, 1, fold, :] = mse_
            ntr[j, 1, fold, 0] = ntr_

            # Also keep track of the r2
            r2pos, r2vel, r2acc, r2post, r2velt, r2acct, _, ntr_, _, _, _, _ = lr_decode_windowed(Xpca, Z, lag, train_window, test_windows[j], transition_times, train_idxs=tt_train_idxs,
                                                                                                test_idxs=tt_test_idxs, decoding_window=decoding_window, pkassign=pkassign, offsets=None) 
            wr2[j, 0, fold, :] = (r2pos, r2vel, r2acc, r2post, r2velt, r2acct)
            ntr[j, 0, fold, 1] = ntr_

            r2pos, r2vel, r2acc, r2post, r2velt, r2acct, _, ntr_, _, _, _, _ = lr_decode_windowed(Xfcca, Z, lag, train_window, test_windows[j], transition_times, train_idxs=tt_train_idxs,
                                                                                           test_idxs=tt_test_idxs, decoding_window=decoding_window, 
                                                                                           pkassign=pkassign, offsets=None)
            wr2[j, 1, fold, :] = (r2pos, r2vel, r2acc, r2post, r2velt, r2acct)
            ntr[j, 1, fold, 1] = ntr_

                        
    dpath = '/home/akumar/nse/neural_control/data/biasvariance_vst_all'
    #dpath = '/mnt/sdb1/nc_data/decodingvt'
    with open('%s/didx%d_rank%d.dat' % (dpath, didx, comm.rank), 'wb') as f:
        f.write(pickle.dumps(bias))
        f.write(pickle.dumps(var))
        f.write(pickle.dumps(mse))
        f.write(pickle.dumps(wr2))
        f.write(pickle.dumps(train_windows))
        f.write(pickle.dumps(test_windows))
        f.write(pickle.dumps(dimval))
        f.write(pickle.dumps(ntr))