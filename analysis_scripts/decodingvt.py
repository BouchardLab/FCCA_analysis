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
from decoders import lr_decode_windowed

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
               }

def gen_run(name, atype, didxs, error_filt_params, reach_filt_params):
    combs = itertools.product(atype, didxs, error_filt_params, reach_filt_params)
    with open(name, 'w') as rsh:
        rsh.write('#!/bin/bash\n')
        for (at, di, efp, rfp) in combs:
            rsh.write('mpirun -n 8 python decodingvt.py %d %d %d --error_thresh=%.2f --error_op=%s --q=%.2f --filter_op=%s\n'
                    % (at, di, rfp[0], efp[0], efp[1], rfp[1], rfp[2]))

# Filter reaches by:
# 0: Nothing
# 1: Top/Bottom filter_percentile in reach straightness
# 2: Top/Bottom filter_percentile in reach duration
# 3: Reach length (discrete category)
# 4: Number of peaks in velocity (n, equal, ge, le)
# Can add error threshold on top
def filter_reach_type(dat, reach_filter, error_percentile=0., error_op='ge', q=1., op='ge'):

    error_thresh = np.quantile(dat['target_pair_error'], error_percentile)
    transition_times = np.array(dat['transition_times'], dtype=object)
    
    if error_op == 'ge':
        error_filter = np.squeeze(np.argwhere(dat['target_pair_error'] >= error_thresh)).astype(int)
    else:
        error_filter = np.squeeze(np.argwhere(dat['target_pair_error'] <= error_thresh)).astype(int)

    transition_times = transition_times[error_filter]

    if reach_filter == 0:
        reach_filter = np.arange(len(transition_times))
    elif reach_filter == 1:
        straight_dev = dat['straight_dev'][error_filter]
        straight_thresh = np.quantile(straight_dev, q)
        if op == 'ge':
            reach_filter = np.squeeze(np.argwhere(straight_dev >= straight_thresh))
        else:
            reach_filter = np.squeeze(np.argwhere(straight_dev <= straight_thresh))
    elif reach_filter == 2:
        # No need to apply error filter here
        reach_duration = np.array([t[1] - t[0] for t in transition_times])

        duration_thresh = np.quantile(reach_duration, q)
        if op == 'ge':
            reach_filter = np.squeeze(np.argwhere(reach_duration >= duration_thresh))
        else:
            reach_filter = np.squeeze(np.argwhere(reach_duration <= duration_thresh))
    elif reach_filter == 3:
        l = np.array([np.linalg.norm(np.array(dat['target_pairs'][i])[1, :] - np.array(dat['target_pairs'][i])[0, :]) 
              for i in range(len(dat['target_pairs']))])
        l = l[error_filter]
        l_thresh = np.quantile(l, q)
        if op == 'ge':
            reach_filter = np.squeeze(np.argwhere(l >= l_thresh))
        else:
            reach_filter = np.squeeze(np.argwhere(l <= l_thresh))

    elif reach_filter == 4:

        # Identify peaks in the velocity
        vel = np.diff(dat['behavior'], axis=0)

        npeaks = []
        pks = []
        pkdata = []

        for t0, t1 in transition_times:
            vel_ = np.linalg.norm(vel[t0:t1, :], axis=1)
            pks_, pkdata = scipy.signal.find_peaks(vel_, prominence=2)
            npeaks.append(len(pks_))
            pks.append(pks_)

        if op == 'eq':
            reach_filter = np.squeeze(np.argwhere(np.array(npeaks) == q))
        elif op == 'lt':
            reach_filter = np.squeeze(np.argwhere(np.array(npeaks) < q))
        elif op == 'gt':
            reach_filter = np.squeeze(np.argwhere(np.array(npeaks) > q))

    transition_times = transition_times[reach_filter]
    print('%d Reaches' % len(transition_times))
    return transition_times, error_filter, reach_filter

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('atype', type=int)
    parser.add_argument('didx', type=int)
    parser.add_argument('reach_filter', type=int, default=0)
    parser.add_argument('--error_thresh', type=float, default=1.)
    parser.add_argument('--error_op', default='le')
    parser.add_argument('--q', type=float, default=0.5)
    parser.add_argument('--filter_op', default='ge')

    args = parser.parse_args()    
    atype = args.atype
    didx = args.didx
    comm = MPI.COMM_WORLD

    # Restrict the analysis to just the first fold
    folds = np.array([0])
    #dimvals = np.unique(sabes_df1['dim'].values)[0:21]
    dimvals = np.array([6, 10, 15, 20])

    if comm.rank == 0:
        with open('/home/akumar/nse/neural_control/data/sabes_decoding_df.dat', 'rb') as f:
            sabes_df1 = pickle.load(f)
        with open('/home/akumar/nse/neural_control/data/sabes_decoding_sf.dat', 'rb') as f:
            sabes_df2 = pickle.load(f)

        sabes_df2 = pd.DataFrame(sabes_df2)
        data_files = np.unique(sabes_df1['data_file'].values)
        data_file = data_files[didx]

        # Extract only what is needed from the dataframes and then purge them from memory. 
        coefpca = [[] for f in range(folds.size)]
        coeffcca = [[] for f in range(folds.size)]
        trainidxs = [[] for f in range(folds.size)]
        testidxs = [[] for f in range(folds.size)]

        for f in folds:
            for d, dimval in enumerate(dimvals):
                df1 = apply_df_filters(sabes_df1, data_file=data_file, dimreduc_method='PCA', fold_idx=f, dim=dimval)
                df2 = apply_df_filters(sabes_df2, data_file=data_file, dimreduc_method='LQGCA', fold_idx=f, dim=dimval,
                                        dimreduc_args={'T': 3, 'loss_type': 'trace', 'n_init': 10})

                assert(df1.shape[0] == 1)
                assert(df2.shape[0] == 1)

                assert(np.all(df1.iloc[0]['train_idxs'] == df2.iloc[0]['train_idxs']))
                assert(np.all(df1.iloc[0]['test_idxs'] == df2.iloc[0]['test_idxs']))

                trainidxs[f] = df1.iloc[0]['train_idxs']
                testidxs[f] = df1.iloc[0]['test_idxs']
                
                coefpca[f].append(df1.iloc[0]['coef'][:, 0:dimval])
                coeffcca[f].append(df2.iloc[0]['coef'])


        dat = load_sabes('/mnt/Secondary/data/sabes/%s' % data_file)
        dat = reach_segment_sabes(dat, start_times[data_file.split('.mat')[0]])
        X = np.squeeze(dat['spike_rates'])
        Z = dat['behavior']
        # transition_times = dat['transition_times']

        transition_times, error_filter, reach_filter = filter_reach_type(dat, args.reach_filter, 
                                                                         args.error_thresh, args.error_op, 
                                                                         q=args.q, op=args.filter_op)
        # Encode the error_thresh, error_op, reach filter, q and op into a string
        filter_params = {'error_thresh':args.error_thresh, 'error_op':args.error_op,
                         'reach_filter':args.reach_filter, 'q':args.q, 'op':args.filter_op}

        filter_string = 'rf_%d_op_%s_q_%d_et_%d_eop_%s' % (int(args.reach_filter), args.filter_op, int(100*args.q),
                                                           int(100*args.error_thresh), args.error_op)

        del sabes_df1
        del sabes_df2

    else:
        dat = None
        data_files = None
        coefpca = None
        coeffcca = None
        trainidxs = None
        testidxs = None
        X = None
        Z = None
        transition_times = None
        error_filter = None
        reach_filter = None
        filter_params = None
        filter_string = None

    coefpca = comm.bcast(coefpca)
    coeffcca = comm.bcast(coeffcca)
    trainidxs = comm.bcast(trainidxs)
    testidxs = comm.bcast(testidxs)
    X = comm.bcast(X)
    Z = comm.bcast(Z)
    transition_times = comm.bcast(transition_times)
    error_filter = comm.bcast(error_filter)
    reach_filter = comm.bcast(reach_filter)
    filter_params = comm.bcast(filter_params)
    filter_string = comm.bcast(filter_string)

    lag = 4
    decoding_window = 5

    if atype == 0:
        # Sliding windows
        window_width = 10
        window_centers = np.linspace(0, 35, 15)

        wr2 = np.zeros((dimvals.size, len(windows), 2, 6))
        mse = np.zeros((dimvals.size, len(windows), 2, 2), dtype=object)

        for d, dimval in tqdm(enumerate(dimvals)):

            coef_pca = coefpca[0][d]
            coef_fcca = coeffcca[0][d]

            xpca = X @ coef_pca
            xfcca = X @ coef_fcca

            # Apply projection
            for j, window in enumerate(windows):
                r2pos, r2vel, r2acc, r2post, r2velt, r2acct, msetr, msete, _ = lr_decode_windowed(xpca, Z, lag, window, transition_times,
                                                                                    train_idxs, test_idxs, decoding_window=decoding_window) 
                wr2[d, j, 0, :] = (r2pos, r2vel, r2acc, r2post, r2velt, r2acct)
                mse[d, j, 0, :] = (msetr, msete)
                r2pos, r2vel, r2acc, r2post, r2velt, r2acct, msetr, msete, _ = lr_decode_windowed(xfcca, Z, lag, window, transition_times,
                                                                                    train_idxs, test_idxs, decoding_window=decoding_window)
                wr2[d, j, 1, :] = (r2pos, r2vel, r2acc, r2post, r2velt, r2acct)
                mse[d, j, 1, :] = (msetr, msete)
                
    elif atype == 1:
        vel = np.diff(Z, axis=0)

        npeaks = []
        pks = []
        pkdata = []

        for t0, t1 in transition_times:
            vel_ = np.linalg.norm(vel[t0:t1, :], axis=1)
            pks_, pkdata = scipy.signal.find_peaks(vel_, prominence=2)
            npeaks.append(len(pks_))
            pks.append(pks_)

        single_peak_reaches = np.where(np.array(npeaks) == 1)[0]
        pks = [pks[idx] for idx in single_peak_reaches]
        transition_times = np.array(transition_times)[single_peak_reaches]

        # Take a look at (1) Beginning to half max, (2) half max to half max (3) half max to end (4) peak to end
        pw = []
        for k, (t0, t1) in enumerate(transition_times):
            vel_ = np.linalg.norm(vel[t0:t1, :], axis=1)
            pstart = np.where(vel_ > np.max(vel_)/2)[0][0]
            pend = np.where(vel_ > np.max(vel_)/2)[0][-1]
            pw.append((pstart, np.argmax(vel_), pend))

        windows = []
        valid_transition_times = []
        for k1 in range(6):
            windows.append([])    
            valid_transition_times.append([])


        # Modify to ensure that the window size is at least as long as the decoding window
        for k2 in range(len(pw)):

            start = 0
            end = pw[k2][0]
            if end - start > decoding_window + 1: 
                windows[0].append((start, end))
                valid_transition_times[0].append(transition_times[k2])

            start = pw[k2][0]
            end = pw[k2][2]
            if end - start > decoding_window + 1: 
                windows[1].append((start, end))
                valid_transition_times[1].append(transition_times[k2])

            start = pw[k2][0]
            end = transition_times[k2][1] - transition_times[k2][0]
            if end - start > decoding_window + 1: 
                windows[2].append((start, end))
                valid_transition_times[2].append(transition_times[k2])

            start = pw[k2][1]
            end = transition_times[k2][1] - transition_times[k2][0]
            if end - start > decoding_window + 1: 
                windows[3].append((start, end))
                valid_transition_times[3].append(transition_times[k2])


            start = pw[k2][2]
            end = transition_times[k2][1] - transition_times[k2][0]
            if end - start > decoding_window + 1: 
                windows[4].append((start, end))
                valid_transition_times[4].append(transition_times[k2])

            start = -5
            end = pw[k2][0]
            if end - start > decoding_window + 1: 
                windows[5].append((start, end))
                valid_transition_times[5].append(transition_times[k2])

        wr2 = np.zeros((dimvals.size, len(windows), 2, 6))
        mse = np.zeros((dimvals.size, len(windows), 2, 2), dtype=object)

        # Folds distributed over ranks
        train_idxs = trainidxs[comm.rank]
        test_idxs = testidxs[comm.rank]

        for f, fold in enumerate(folds):
            for d, dimval in tqdm(enumerate(dimvals)):

                # Apply projection
                coef_pca = coefpca[comm.rank][d]
                coef_fcca = coeffcca[comm.rank][d]

                xpca = X @ coef_pca
                xfcca = X @ coef_fcca

                for j, window in enumerate(windows):
                    if len(window) > 10:
                        r2pos, r2vel, r2acc, r2post, r2velt, r2acct, msetr, msete, _ = lr_decode_windowed(xpca, Z, lag, window, valid_transition_times[j],
                                                                                            train_idxs, test_idxs, decoding_window=decoding_window) 
                        wr2[d, j, 0, :] = (r2pos, r2vel, r2acc, r2post, r2velt, r2acct)
                        mse[d, j, 0, :] = (msetr, msete)
                        r2pos, r2vel, r2acc, r2post, r2velt, r2acct, msetr, msete, _ = lr_decode_windowed(xfcca, Z, lag, window, valid_transition_times[j],
                                                                                            train_idxs, test_idxs, decoding_window=decoding_window)
                        wr2[d, j, 1, :] = (r2pos, r2vel, r2acc, r2post, r2velt, r2acct)
                        mse[d, j, 1, :] = (msetr, msete)
                    else:
                        wr2[f, d, j, 0, :] = (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
                        wr2[f, d, j, 1, :] = (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
                        mse[d, j, 0, :] = (np.nan, np.nan)
                        mse[d, j, 1, :] = (np.nan, np.nan)
        

    windows = np.array(windows)
    dpath = '/home/akumar/nse/neural_control/data/decodingvtfull'

    with open('%s/atype%d_didx%d_widx%d_%s.dat' % (dpath, atype, didx, comm.rank, filter_string), 'wb') as f:
        f.write(pickle.dumps(wr2))
        f.write(pickle.dumps(mse))
        f.write(pickle.dumps(error_filter))
        f.write(pickle.dumps(reach_filter))
        f.write(pickle.dumps(windows))
        f.write(pickle.dumps(filter_params))
        