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

def gen_run(name, didxs=np.arange(8), error_filt_params=[(1., 'le')], reach_filt_params=[(0, 0, 'le')]):
    combs = itertools.product(didxs, np.array([2, 6, 9]), ['S1', 'M1'])
    with open(name, 'w') as rsh:
        rsh.write('#!/bin/bash\n')
        for (di, dimval, reg) in combs:
            rsh.write('mpirun -n 8 python decodingvt_cv_ilmerge_cca.py %d %d %s\n'
                    % (di, dimval, reg))

# Filter reaches by:
# 0: Nothing
# 1: Top/Bottom filter_percentile in reach straightness
# 2: Top/Bottom filter_percentile in reach duration
# 3: Reach length (discrete category)
# 4: Number of peaks in velocity (n, equal, ge, le)
# Can add error threshold on top
def filter_reach_type(dat, reach_filter, error_percentile=0., error_op='ge', q=1., op='ge', windows=None):

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

    # Finally, filter such that for each recording session, the same reaches are assessed across all
    # windows. This requires taking the intersection of reaches that satisfy the window condition here.
    def valid_reach(t0, t1, w, measure_from_end):
        if measure_from_end:
            window_in_reach = t1 - w[1] > t0
        else:
            window_in_reach = t0 + w[1] < t1
        return window_in_reach

    if windows is not None:
        window_filter = []
        for i, window in enumerate(windows):
            window_filter.append([])
            for j, (t0, t1) in enumerate(transition_times):
                # Enforce that the previous reach must not have began after the window begins
                if valid_reach(t0,  t1, window, measure_from_end):
                    window_filter[i].append(j)
        
        # Take the intersection
        window_filter_int = set(window_filter[0])
        for wf in window_filter:
            window_filter_int = window_filter_int.intersection(set(wf))

        window_filter = list(window_filter_int)
        transition_times = transition_times[window_filter]
    else:
        window_filter = None

    print('%d Reaches' % len(transition_times))
    return transition_times, error_filter, reach_filter, window_filter

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('didx', type=int)
    parser.add_argument('dimval', type=int)
    parser.add_argument('region')

    args = parser.parse_args()    
    didx = args.didx
    comm = MPI.COMM_WORLD

    # Iterate over dimvals snce we neef
    dimval = args.dimval
    measure_from_end=False

    # Sliding windows
    window_width = 2
    #window_centers = np.linspace(0, 35, 25)[0:9]
    window_centers = np.arange(30)
    windows = [(int(wc - window_width//2), int(wc + window_width//2)) for wc in window_centers]

    if comm.rank == 0:

        # Load CCA fits
        with open('/mnt/Secondary/data/postprocessed/sabes_cca50_cvall.dat', 'rb') as f:
            sabes_df = pickle.load(f)

        data_files = np.unique(sabes_df['fl'].values)
        data_file = data_files[didx]

        # df = apply_df_filters(sabes_df, data_file=data_file, dim=dimval)
        # assert(df.shape[0] == 1)
        # coefpca.append(df.iloc[0]['pcacoef'])
        # coeffcca.append(df.iloc[0]['lqgcoef'])            

        dfcca = apply_df_filters(sabes_df, fl=data_file)
        data_file = data_file.split('/')[-1].split('.pkl')[0] + '.mat'
        assert(dfcca.shape[0] == 5)        

        if args.region == 'M1':
            coefcca = [apply_df_filters(dfcca, fold_idx=k).iloc[0]['ccamodel'].y_rotations_[:, 0:dimval] for k in range(5)]
        else:
            coefcca = [apply_df_filters(dfcca, fold_idx=k).iloc[0]['ccamodel'].x_rotations_[:, 0:dimval] for k in range(5)]

        dat = load_sabes('/mnt/Secondary/data/sabes/%s' % data_file, region=args.region)
        # dat = load_sabes(data_file)
        data_file = data_file.split('/')[-1]
        # dat = load_sabes('/mnt/sdb1/nc_data/sabes/%s' % data_file)
        dat = reach_segment_sabes(dat, data_file=data_file.split('.mat')[0])
        X = np.squeeze(dat['spike_rates'])
        Z = dat['behavior']
        # transition_times = dat['transition_times']

        # Hard coded defaults
        reach_filter = 0
        error_thresh = 1.
        error_op = 'le'
        q = 0
        filter_op = 'le'
        transition_times, error_filter, reach_filter, window_filter = filter_reach_type(dat, reach_filter, 
                                                                                        error_thresh, error_op, 
                                                                                        q=q, op=filter_op, windows=windows)
        # Encode the error_thresh, error_op, reach filter, q and op into a string
        filter_params = {'error_thresh':error_thresh, 'error_op':error_op,
                         'reach_filter':reach_filter, 'q':q, 'op':filter_op}
    else:
        dat = None
        data_files = None
        coefcca = None
        X = None
        Z = None
        transition_times = None
        error_filter = None
        reach_filter = None
        window_filter = None
        filter_params = None

    coefcca = comm.bcast(coefcca)

    X = comm.bcast(X)
    Z = comm.bcast(Z)

    transition_times = comm.bcast(transition_times)
    error_filter = comm.bcast(error_filter)
    reach_filter = comm.bcast(reach_filter)
    window_filter = comm.bcast(window_filter)
    filter_params = comm.bcast(filter_params)

    # S1 peaks at smaller lag
    if args.region == 'S1':
        lag = 1
    else:
        lag = 2
    decoding_window = 5

    # Distribute windows across ranks
    windows = np.array_split(windows, comm.size)[comm.rank]
    wr2 = np.zeros((len(windows), 5, 1, 6))

    # Apply projection

    # Cross-validate the prediction
    for j, window in enumerate(windows):
        for fold, (train_idxs, test_idxs) in tqdm(enumerate(KFold(n_splits=5).split(Z))): 
            xcca = X @ coefcca[fold]

            # Need to turn train/test_idxs returned by KFold into an indexing of the transition times
            tt_train_idxs = [idx for idx in range(len(transition_times)) if transition_times[idx][0] in train_idxs and transition_times[idx][1] in train_idxs]
            tt_test_idxs = [idx for idx in range(len(transition_times)) if transition_times[idx][0] in test_idxs and transition_times[idx][1] in test_idxs]

            r2pos, r2vel, r2acc, r2post, r2velt, r2acct, _, ntr, _, _, _, _ = lr_decode_windowed(xcca, Z, lag, [window], [window], transition_times, train_idxs=tt_train_idxs,
                                                                                                test_idxs=tt_test_idxs, decoding_window=decoding_window) 

            wr2[j, fold, 0, :] = (r2pos, r2vel, r2acc, r2post, r2velt, r2acct)

    windows = np.array(windows)
    dpath = '/home/akumar/nse/neural_control/data/decodingvt_cv_s1_w30'
    #dpath = '/mnt/sdb1/nc_data/decodingvt'
    with open('%s/didx%d_rank%d_dim%d_r%s.dat' % (dpath, didx, comm.rank, dimval, args.region), 'wb') as f:
        f.write(pickle.dumps(wr2))
        f.write(pickle.dumps(error_filter))
        f.write(pickle.dumps(reach_filter))
        f.write(pickle.dumps(window_filter))
        f.write(pickle.dumps(windows))
        f.write(pickle.dumps(filter_params))