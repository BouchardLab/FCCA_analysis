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
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.preprocessing import StandardScaler

import itertools
from sklearn.model_selection import KFold

sys.path.append('/home/akumar/nse/neural_control')
from utils import apply_df_filters, calc_loadings
from loaders import load_sabes
from segmentation import reach_segment_sabes
from decoders import lr_decode_windowed, apply_window

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

# Loco files with decent decoding + indy session with S1 recording
fls = ['loco_20170210_03.mat',
 'loco_20170213_02.mat',
 'loco_20170215_02.mat',
 'loco_20170227_04.mat',
 'loco_20170228_02.mat',
 'loco_20170301_05.mat',
 'loco_20170302_02.mat',
 'indy_20160426_01.mat']

fls_ = ['loco_20170210_03.pkl',
 'loco_20170213_02.pkl',
 'loco_20170215_02.pkl',
 'loco_20170227_04.pkl',
 'loco_20170228_02.pkl',
 'loco_20170301_05.pkl',
 'loco_20170302_02.pkl',
 'indy_20160426_01.pkl']


def gen_run():
    name = '/home/akumar/nse/neural_control/analysis_scripts/run.sh'
    didxs = np.arange(8)

    with open(name, 'w') as rsh:
        rsh.write('#!/bin/bash\n')
        for di in didxs:
            rsh.write('mpirun -n 8 python s1m1vt.py %d %d --error_thresh=%.2f --error_op=%s --q=%.2f --filter_op=%s\n'
                    % (di, 0, 1, 'le', 0, 'le'))

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
    parser.add_argument('reach_filter', type=int, default=0)
    parser.add_argument('--error_thresh', type=float, default=1.)
    parser.add_argument('--error_op', default='le')
    parser.add_argument('--q', type=float, default=0.5)
    parser.add_argument('--filter_op', default='ge')

    cmd_args = parser.parse_args()    
    didx = cmd_args.didx
    comm = MPI.COMM_WORLD

    #dimvals = np.array([2, 6, 10, 15])
    ccadimval = 4
    # Fix dimension to 6
    dimval = 4
    measure_from_end=False

    # Sliding windows
    window_width = 2
    #window_centers = np.linspace(0, 35, 25)[0:9]
    window_centers = np.arange(30)
    windows = [(int(wc - window_width//2), int(wc + window_width//2)) for wc in window_centers]

    if comm.rank == 0:
        # Update 03/29/2023: Augment with FCCA/PCA fits in S1

        # CCA fit on all data files, 50 ms
        with open('/mnt/Secondary/data/postprocessed/sabes_cca50cv_df.dat', 'rb') as f:
            ccadf = pickle.load(f)
        if 'indy' in fls[didx]:
            with open('/mnt/Secondary/data/postprocessed/indy_decoding_df2.dat', 'rb') as f:
                sabes_df = pickle.load(f)

            sabes_df = pd.DataFrame(sabes_df)

            dffca = apply_df_filters(sabes_df, data_file=fls[didx], dim=dimval, dimreduc_method='LQGCA')

            dfpca = apply_df_filters(sabes_df, data_file=fls[didx], dim=dimval, dimreduc_method='PCA')

            with open('/mnt/Secondary/data/postprocessed/indy_S1_df.dat', 'rb') as f:
                sabes_dfS1 = pickle.load(f)

            sabes_dfS1 = pd.DataFrame(sabes_dfS1)
            dffcaS1 = apply_df_filters(sabes_dfS1, data_file=fls[didx], dim=dimval, dimreduc_method='LQGCA')
            dfpcaS1 = apply_df_filters(sabes_dfS1, data_file=fls[didx], dim=dimval, dimreduc_method='PCA')


        elif 'loco' in fls[didx]:
            with open('/mnt/Secondary/data/postprocessed/loco_decoding_df.dat', 'rb') as f:
                sabes_df = pickle.load(f)

            sabes_df = pd.DataFrame(sabes_df)

            dffca = apply_df_filters(sabes_df, data_file=fls[didx], dim=dimval, dimreduc_method='LQGCA', 
                                    dimreduc_args={'T':3, 'loss_type':'trace', 'n_init':10},
                                    loader_args={'bin_width': 50, 'filter_fn': 'none', 'filter_kwargs': {}, 'boxcox': 0.5, 'spike_threshold': 100, 'region': 'M1'},
                                    decoder_args={'trainlag': 4, 'testlag': 4, 'decoding_window': 5})

            dfpca = apply_df_filters(sabes_df, data_file=fls[didx], dim=dimval, dimreduc_method='PCA',
                                    loader_args={'bin_width': 50, 'filter_fn': 'none', 'filter_kwargs': {}, 'boxcox': 0.5, 'spike_threshold': 100, 'region': 'M1'},
                                    decoder_args={'trainlag': 4, 'testlag': 4, 'decoding_window': 5})

            dffcaS1 = apply_df_filters(sabes_df, data_file=fls[didx], dim=dimval, dimreduc_method='LQGCA', 
                                    dimreduc_args={'T':3, 'loss_type':'trace', 'n_init':10},
                                    loader_args={'bin_width': 50, 'filter_fn': 'none', 'filter_kwargs': {}, 'boxcox': 0.5, 'spike_threshold': 100, 'region': 'S1'},
                                    decoder_args={'trainlag': 4, 'testlag': 4, 'decoding_window': 5})

            dfpcaS1 = apply_df_filters(sabes_df, data_file=fls[didx], dim=dimval, dimreduc_method='PCA',
                                    loader_args={'bin_width': 50, 'filter_fn': 'none', 'filter_kwargs': {}, 'boxcox': 0.5, 'spike_threshold': 100, 'region': 'S1'},
                                    decoder_args={'trainlag': 4, 'testlag': 4, 'decoding_window': 5})


        # Only use valid files
        ccadf = apply_df_filters(ccadf, fl=fls_[didx], lag=0, win=1)


        # Will later need to be cross-validated
        assert(ccadf.shape[0] == 5)
        assert(dffca.shape[0] == 5)
        assert(dfpca.shape[0] == 5)
        assert(dffcaS1.shape[0] == 5)
        assert(dfpcaS1.shape[0] == 5)
        
        coefcca_x = [apply_df_filters(ccadf, fold_idx=k).iloc[0]['ccamodel'].x_rotations_[:, 0:ccadimval] for k in range(5)]
        coefcca = [apply_df_filters(ccadf, fold_idx=k).iloc[0]['ccamodel'].y_rotations_[:, 0:ccadimval] for k in range(5)]
        coefpca = [apply_df_filters(dfpca, fold_idx=k).iloc[0]['coef'][:, 0:dimval] for k in range(5)]
        coeffcca = [apply_df_filters(dffca, fold_idx=k).iloc[0]['coef'][:, 0:dimval] for k in range(5)]
        coefpcaS1 = [apply_df_filters(dfpcaS1, fold_idx=k).iloc[0]['coef'][:, 0:dimval] for k in range(5)]
        coeffccaS1 = [apply_df_filters(dffcaS1, fold_idx=k).iloc[0]['coef'][:, 0:dimval] for k in range(5)]

        # dat = load_sabes('/mnt/Secondary/data/sabes/%s' % data_file)
        dat = load_sabes('/mnt/Secondary/data/sabes/%s' % fls[didx], bin_width=ccadf.iloc[0]['bin_width'], filter_fn=ccadf.iloc[0]['filter_fn'], 
                         filter_kwargs=ccadf.iloc[0]['filter_kwargs'], region='S1')
        data_file = fls[didx].split('/')[-1].split('.mat')[0]
        # dat = load_sabes('/mnt/sdb1/nc_data/sabes/%s' % data_file)
        dat = reach_segment_sabes(dat, data_file=data_file)
        X = np.squeeze(dat['spike_rates'])
        
        dat = load_sabes('/mnt/Secondary/data/sabes/%s' % fls[didx], bin_width=ccadf.iloc[0]['bin_width'], filter_fn=ccadf.iloc[0]['filter_fn'], 
                         filter_kwargs=ccadf.iloc[0]['filter_kwargs'], region='M1')
        # dat = load_sabes('/mnt/sdb1/nc_data/sabes/%s' % data_file)
        dat = reach_segment_sabes(dat, data_file=data_file)
        Y = np.squeeze(dat['spike_rates'])
        
        transition_times, error_filter, reach_filter, window_filter = filter_reach_type(dat, cmd_args.reach_filter, 
                                                                                        cmd_args.error_thresh, cmd_args.error_op, 
                                                                                        q=cmd_args.q, op=cmd_args.filter_op, windows=windows)
        # Encode the error_thresh, error_op, reach filter, q and op into a string
        filter_params = {'error_thresh':cmd_args.error_thresh, 'error_op':cmd_args.error_op,
                         'reach_filter':cmd_args.reach_filter, 'q':cmd_args.q, 'op':cmd_args.filter_op}

        filter_string = 'rf_%d_op_%s_q_%d_et_%d_eop_%s' % (int(cmd_args.reach_filter), cmd_args.filter_op, int(100*cmd_args.q),
                                                           int(100*cmd_args.error_thresh), cmd_args.error_op)
    else:
        dat = None
        data_files = None
        coefpca = None
        coeffcca = None
        coefpcaS1 = None
        coeffccaS1 = None

        coefcca = None
        coefcca_x = None
        X = None
        Y = None
        transition_times = None
        error_filter = None
        reach_filter = None
        window_filter = None
        filter_params = None
        filter_string = None

    coefpca = comm.bcast(coefpca)
    coeffcca = comm.bcast(coeffcca)
    coefpcaS1 = comm.bcast(coefpcaS1)
    coeffccaS1 = comm.bcast(coeffccaS1)

    coefcca = comm.bcast(coefcca)
    coefcca_x = comm.bcast(coefcca_x)

    # S1 activity
    X = comm.bcast(X)
    # M1 activity
    Y = comm.bcast(Y)

    transition_times = comm.bcast(transition_times)
    error_filter = comm.bcast(error_filter)
    reach_filter = comm.bcast(reach_filter)
    window_filter = comm.bcast(window_filter)
    filter_params = comm.bcast(filter_params)
    filter_string = comm.bcast(filter_string)

    lag = 0
    decoding_window = 5

    # Distribute windows across ranks
    windows = np.array_split(windows, comm.size)[comm.rank]

    # Regressions that are independent of CCA dimension
    ccadims = np.array([4])

    # S1 -> M1 regressions
    wr2_S1M1 = np.zeros((len(windows), 5, 18, ccadims.size))
    # M1 -> S1 regressions
    wr2_M1S1 = np.zeros((len(windows), 5, 18, ccadims.size))
    
    # M1 -> M1 regressions
    wr2_M1 = np.zeros((len(windows), 5, 12, ccadims.size))
    # S1 -> S1 regressions
    wr2_S1 = np.zeros((len(windows), 5, 12, ccadims.size))

    # Iterate over several cca dims to get a sense of how this scales

    # Cross-validate the prediction
    for j, window in enumerate(windows):
        for fold, (train_idxs, test_idxs) in enumerate(KFold(n_splits=5).split(Y)): 
            for k, ccadim in enumerate(ccadims):
                # We have been given a list of windows for each transition
                if len(window) > 2:
                    W = [w for win in window for w in win]
                    win_min = min(W)
                else:
                    win_min = window[0]

                if win_min >= 0:
                    win_min = 0

                ypca = Y @ coefpca[fold]
                yfcca = Y @ coeffcca[fold]

                xpca = X @ coefpcaS1[fold]
                xfcca = X @ coeffccaS1[fold]

                if ccadim == -1:
                    ycca = Y @ coefcca[fold][:, 0:1]
                    xcca = X
                else:
                    ycca = Y @ coefcca[fold][:, 0:ccadim]
                    xcca = X @ coefcca_x[fold][:, 0:ccadim]
                    if ccadim == 1:
                        ycca = np.reshape(ycca, (-1, 1))
                        xcca = np.reshape(xcca, (-1, 1))

                print(ccadim)
                print(ycca.shape)
                print(xcca.shape)

                tt_train = [t for t in transition_times 
                            if t[0] >= min(train_idxs) and t[1] <= max(train_idxs) and t[0] > (lag + np.abs(win_min)) and t[1] < (X.shape[0] - lag - np.abs(win_min))]

                tt_test = [t for t in transition_times 
                        if t[0] >= min(test_idxs) and t[0] <= max(test_idxs) and t[0] > (lag + np.abs(win_min)) and t[1] < (X.shape[0] - lag - np.abs(win_min))]

                # S1 -> M1 regression
                idx = 0
                for features in [xcca, xpca, xfcca]:
                    for targets in[ycca, ypca, yfcca]:
                        xxtrain, yytrain, _, _ = apply_window(features, targets, lag, [window], tt_train, decoding_window, 
                                                        include_velocity=False, include_acc=False)
                        xxtest, yytest, _, _ = apply_window(features, targets, lag, [window], tt_test, decoding_window, 
                                                        include_velocity=False, include_acc=False)

                        regressor = RidgeCV().fit(StandardScaler().fit_transform(np.concatenate(xxtrain)),
                                                StandardScaler().fit_transform(np.concatenate(yytrain)))

                        r2train = regressor.score(StandardScaler().fit_transform(np.concatenate(xxtrain)),
                                                StandardScaler().fit_transform(np.concatenate(yytrain)))
                        r2test = regressor.score(StandardScaler().fit_transform(np.concatenate(xxtest)),
                                                StandardScaler().fit_transform(np.concatenate(yytest)))

                        wr2_S1M1[j, fold, idx, k] = r2train
                        wr2_S1M1[j, fold, idx + 1, k] = r2test
                        idx += 2

                # M1 -> S1 regression
                idx = 0
                for features in [ycca, ypca, yfcca]:
                    for targets in[xcca, xpca, xfcca]:
                        xxtrain, yytrain, _, _ = apply_window(features, targets, lag, [window], tt_train, decoding_window, 
                                                        include_velocity=False, include_acc=False)
                        xxtest, yytest, _, _ = apply_window(features, targets, lag, [window], tt_test, decoding_window, 
                                                        include_velocity=False, include_acc=False)

                        regressor = RidgeCV().fit(StandardScaler().fit_transform(np.concatenate(xxtrain)),
                                                StandardScaler().fit_transform(np.concatenate(yytrain)))

                        r2train = regressor.score(StandardScaler().fit_transform(np.concatenate(xxtrain)),
                                                StandardScaler().fit_transform(np.concatenate(yytrain)))
                        r2test = regressor.score(StandardScaler().fit_transform(np.concatenate(xxtest)),
                                                StandardScaler().fit_transform(np.concatenate(yytest)))

                        wr2_M1S1[j, fold, idx, k] = r2train
                        wr2_M1S1[j, fold, idx + 1, k] = r2test
                        idx += 2


                # M1 -> M1 regression
                idx = 0
                for features in [ycca, ypca, yfcca]:
                    for targets in[ycca, ypca, yfcca]:
                        if np.allclose(features, targets):
                            continue
                        xxtrain, yytrain, _, _ = apply_window(features, targets, lag, [window], tt_train, decoding_window, 
                                                        include_velocity=False, include_acc=False)
                        xxtest, yytest, _, _ = apply_window(features, targets, lag, [window], tt_test, decoding_window, 
                                                        include_velocity=False, include_acc=False)

                        regressor = RidgeCV().fit(StandardScaler().fit_transform(np.concatenate(xxtrain)),
                                                StandardScaler().fit_transform(np.concatenate(yytrain)))

                        r2train = regressor.score(StandardScaler().fit_transform(np.concatenate(xxtrain)),
                                                StandardScaler().fit_transform(np.concatenate(yytrain)))
                        r2test = regressor.score(StandardScaler().fit_transform(np.concatenate(xxtest)),
                                                StandardScaler().fit_transform(np.concatenate(yytest)))

                        wr2_M1[j, fold, idx, k] = r2train
                        wr2_M1[j, fold, idx + 1, k] = r2test
                        idx += 2

                # S1 -> S1 regression
                idx = 0
                for features in [xcca, xpca, xfcca]:
                    for targets in[xcca, xpca, xfcca]:
                        if np.allclose(features, targets):
                            continue
                        xxtrain, yytrain, _, _ = apply_window(features, targets, lag, [window], tt_train, decoding_window, 
                                                        include_velocity=False, include_acc=False)
                        xxtest, yytest, _, _ = apply_window(features, targets, lag, [window], tt_test, decoding_window, 
                                                        include_velocity=False, include_acc=False)

                        regressor = RidgeCV().fit(StandardScaler().fit_transform(np.concatenate(xxtrain)),
                                                StandardScaler().fit_transform(np.concatenate(yytrain)))

                        r2train = regressor.score(StandardScaler().fit_transform(np.concatenate(xxtrain)),
                                                StandardScaler().fit_transform(np.concatenate(yytrain)))
                        r2test = regressor.score(StandardScaler().fit_transform(np.concatenate(xxtest)),
                                                StandardScaler().fit_transform(np.concatenate(yytest)))

                        wr2_S1[j, fold, idx, k] = r2train
                        wr2_S1[j, fold, idx + 1, k] = r2test
                        idx += 2

    windows = np.array(windows)
    dpath = '/home/akumar/nse/neural_control/data/s1m1regvt_allpairsd4w2'
    #dpath = '/mnt/sdb1/nc_data/decodingvt'
    with open('%s/didx%d_rank%d_%s_%d.dat' % (dpath, didx, comm.rank, filter_string, measure_from_end), 'wb') as f:
        f.write(pickle.dumps(wr2_S1))
        f.write(pickle.dumps(wr2_M1))
        f.write(pickle.dumps(wr2_M1S1))
        f.write(pickle.dumps(wr2_S1M1))
        f.write(pickle.dumps(error_filter))
        f.write(pickle.dumps(reach_filter))
        f.write(pickle.dumps(window_filter))
        f.write(pickle.dumps(windows))
        f.write(pickle.dumps(filter_params))
