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
from copy import deepcopy

import itertools
from sklearn.model_selection import KFold

sys.path.append('/home/akumar/nse/neural_control')
from utils import apply_df_filters, calc_loadings
from loaders import load_sabes
from segmentation import reach_segment_sabes
from decoders import lr_decode_windowed

from mpi4py import MPI

from decodingvt_cv_strvcorr import behavioral_metrics

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

def gen_run(name, didxs=np.arange(35), error_filt_params=[(1., 'le')], reach_filt_params=[(0, 0, 'le')]):
    combs = itertools.product(didxs, error_filt_params, reach_filt_params)
    with open(name, 'w') as rsh:
        rsh.write('#!/bin/bash\n')
        for (di, efp, rfp) in combs:
            rsh.write('mpirun -n 8 python decodingvt_cv_ilmerge.py %d %d --error_thresh=%.2f --error_op=%s --q=%.2f --filter_op=%s\n' % (di, rfp[0], efp[0], efp[1], rfp[1], rfp[2]))

# Filter reaches by:
# 0: Nothing
# 1: Top/Bottom filter_percentile in reach straightness
# 2: Top/Bottom filter_percentile in reach duration
# 3: Reach length (discrete category)
# 4: Number of peaks in velocity (n, equal, ge, le)
# Can add error threshold on top
def filter_reach_type(dat, reach_filter, error_percentile=0., error_op='ge', q=1., op='ge', windows=None):
    measure_from_end=False
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

    # Hard-coding window filter to only reflect length 20 reaches
    # Otherwise, using width 30 windows gives rise to too few reaches for good results
    if windows is not None:
        window_centers = np.arange(20)
        windows = [(int(wc - window_width//2), int(wc + window_width//2)) for wc in window_centers]
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
    parser.add_argument('reach_filter', type=int, default=0)
    parser.add_argument('--error_thresh', type=float, default=1.)
    parser.add_argument('--error_op', default='le')
    parser.add_argument('--q', type=float, default=0.5)
    parser.add_argument('--filter_op', default='ge')

    args = parser.parse_args()    
    didx = args.didx
    comm = MPI.COMM_WORLD

    #dimvals = np.array([2, 6, 10, 15])
    # Fix dimension to 6
    dimval = 6
    measure_from_end=False

    # Sliding windows
    window_width = 2
    #window_centers = np.linspace(0, 35, 25)[0:9]
    window_centers = np.arange(30)
    windows = [(int(wc - window_width//2), int(wc + window_width//2)) for wc in window_centers]

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

        transition_times, error_filter, reach_filter, window_filter = filter_reach_type(dat, args.reach_filter, 
                                                                                        args.error_thresh, args.error_op, 
                                                                                        q=args.q, op=args.filter_op, windows=windows)
        # Encode the error_thresh, error_op, reach filter, q and op into a string
        filter_params = {'error_thresh':args.error_thresh, 'error_op':args.error_op,
                         'reach_filter':args.reach_filter, 'q':args.q, 'op':args.filter_op}

        filter_string = 'rf_%d_op_%s_q_%d_et_%d_eop_%s' % (int(args.reach_filter), args.filter_op, int(100*args.q),
                                                           int(100*args.error_thresh), args.error_op)
        target_pairs = dat['target_pairs']
    else:
        dat = None
        data_files = None
        coefpca = None
        coeffcca = None
        X = None
        Z = None
        transition_times = None
        error_filter = None
        reach_filter = None
        window_filter = None
        filter_params = None
        filter_string = None
        target_pairs = None

    coefpca = comm.bcast(coefpca)
    coeffcca = comm.bcast(coeffcca)

    X = comm.bcast(X)
    Z = comm.bcast(Z)

    transition_times = comm.bcast(transition_times)
    error_filter = comm.bcast(error_filter)
    reach_filter = comm.bcast(reach_filter)
    window_filter = comm.bcast(window_filter)
    filter_params = comm.bcast(filter_params)
    filter_string = comm.bcast(filter_string)
    target_pairs = comm.bcast(target_pairs)

    lag = 2
    decoding_window = 5

    # Distribute windows across ranks
    windows = np.array_split(windows, comm.size)[comm.rank]
    wr2 = np.zeros((len(windows), 5, 2, 6))
    ntr = np.zeros((len(windows), 5, 2))

    # Apply projection
    MSEtr = np.zeros((len(windows), 5, 2), dtype=object)
    MSEte = np.zeros((len(windows), 5, 2), dtype=object)

    full_reaches_train = np.zeros((len(windows), 5), dtype=object)
    full_reaches_test = np.zeros((len(windows), 5), dtype=object)

    # Windows x folds x straight/corrective x train/test x metric
    behavioral_metrics_array = np.zeros((len(windows), 5, 2, 4), dtype=object)

    # Cross-validate the prediction
    for j, window in enumerate(windows):
        for fold, (train_idxs, test_idxs) in tqdm(enumerate(KFold(n_splits=5).split(Z))): 
            xpca = X @ coefpca[fold]
            xfcca = X @ coeffcca[fold]

            # Need to turn train/test_idxs returned by KFold into an indexing of the transition times
            tt_train_idxs = [idx for idx in range(len(transition_times)) if transition_times[idx][0] in train_idxs and transition_times[idx][1] in train_idxs]
            tt_test_idxs = [idx for idx in range(len(transition_times)) if transition_times[idx][0] in test_idxs and transition_times[idx][1] in test_idxs]

            r2pos, r2vel, r2acc, r2post, r2velt, r2acct, reg, ntr_, fitr, fite, msetr, msete  = lr_decode_windowed(xpca, Z, lag, [window], [window], transition_times, train_idxs=tt_train_idxs,
                                                                                                                test_idxs=tt_test_idxs, decoding_window=decoding_window) 
            
            # Narrow down the msetr and msete by the fitr/fite 
            msetr = [msetr[i] for i in range(len(msetr)) if i in fitr]
            msete = [msete[i] for i in range(len(msete)) if i in fite]

            wr2[j, fold, 0, :] = (r2pos, r2vel, r2acc, r2post, r2velt, r2acct)
            MSEtr[j, fold, 0] = msetr
            MSEte[j, fold, 0] = msete

            r2pos, r2vel, r2acc, r2post, r2velt, r2acct, reg, ntr_, fitr, fite, msetr, msete  = lr_decode_windowed(xfcca, Z, lag, [window], [window], transition_times, train_idxs=tt_train_idxs,
                                                                                                                   test_idxs=tt_test_idxs, decoding_window=decoding_window)

            wr2[j, fold, 1, :] = (r2pos, r2vel, r2acc, r2post, r2velt, r2acct)
            # Narrow down the msetr and msete by the fitr/fite 
            msetr = [msetr[i] for i in range(len(msetr)) if i in fitr]    
            msete = [msete[i] for i in range(len(msete)) if i in fite]
    
            MSEtr[j, fold, 1] = msetr
            MSEte[j, fold, 1] = msete

            # Convert fitr and fite to an indexing of the original transition times
            fitr = [tt_train_idxs[idx] for idx in fitr]
            fite = [tt_test_idxs[idx] for idx in fite]

            full_reaches_train[j, fold] = fitr
            full_reaches_test[j, fold] = fite

            # Calculate behavioral metrics. For some inexplicable reason, this modifies Z, so use deepcopy
            dftsecondphase_tr, maxperpd_tr, secondphaseduration_tr, perpdtr = behavioral_metrics(fitr, np.array(transition_times), target_pairs, deepcopy(Z))
            dftsecondphase_te, maxperpd_te, secondphaseduration_te, perpdte = behavioral_metrics(fite, np.array(transition_times), target_pairs, deepcopy(Z))

            behavioral_metrics_array[j, fold, 0, 0] = dftsecondphase_tr
            behavioral_metrics_array[j, fold, 0, 1] = maxperpd_tr
            behavioral_metrics_array[j, fold, 0, 2] = secondphaseduration_tr
            behavioral_metrics_array[j, fold, 0, 3] = perpdtr

            behavioral_metrics_array[j, fold, 1, 0] = dftsecondphase_te
            behavioral_metrics_array[j, fold, 1, 1] = maxperpd_te
            behavioral_metrics_array[j, fold, 1, 2] = secondphaseduration_te
            behavioral_metrics_array[j, fold, 1, 3] = perpdte

    windows = np.array(windows)
    dpath = '/home/akumar/nse/neural_control/data/decodingvt_cv_ttshift_lag2w30'
    #dpath = '/mnt/sdb1/nc_data/decodingvt'
    with open('%s/didx%d_rank%d_%s_%d.dat' % (dpath, didx, comm.rank, filter_string, measure_from_end), 'wb') as f:
        f.write(pickle.dumps(wr2))
        f.write(pickle.dumps(error_filter))
        f.write(pickle.dumps(reach_filter))
        f.write(pickle.dumps(window_filter))
        f.write(pickle.dumps(windows))
        f.write(pickle.dumps(filter_params))
        f.write(pickle.dumps(MSEtr))
        f.write(pickle.dumps(MSEte))
        f.write(pickle.dumps(full_reaches_train))
        f.write(pickle.dumps(full_reaches_test))
        f.write(pickle.dumps(behavioral_metrics_array))
