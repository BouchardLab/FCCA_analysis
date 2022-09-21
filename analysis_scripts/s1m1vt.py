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

def gen_run():
    name = '/home/akumar/nse/neural_control/analysis_scripts/run.sh'
    didxs = np.array([0, 1])
    lidxs = np.array([0, 1, 2, 3])

    combs = itertools.product(didxs, lidxs)
    with open(name, 'w') as rsh:
        rsh.write('#!/bin/bash\n')
        for (di, li) in combs:
            rsh.write('mpirun -n 8 python s1m1vt.py %d %d %d --error_thresh=%.2f --error_op=%s --q=%.2f --filter_op=%s\n'
                    % (di, li, 0, 1, 'le', 0, 'le'))

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
    parser.add_argument('lidx', type=int)
    parser.add_argument('reach_filter', type=int, default=0)
    parser.add_argument('--error_thresh', type=float, default=1.)
    parser.add_argument('--error_op', default='le')
    parser.add_argument('--q', type=float, default=0.5)
    parser.add_argument('--filter_op', default='ge')

    cmd_args = parser.parse_args()    
    didx = cmd_args.didx
    lidx = cmd_args.lidx
    comm = MPI.COMM_WORLD

    #dimvals = np.array([2, 6, 10, 15])
    # Fix dimension to 6
    dimval = 6
    measure_from_end=False

    # Sliding windows
    window_width = 10
    #window_centers = np.linspace(0, 35, 25)[0:9]
    window_centers = np.arange(12)
    windows = [(int(wc - window_width//2), int(wc + window_width//2)) for wc in window_centers]

    # Select the indy data file that has both M1/S1 and one loco datafile that seems to give decent decoding performance
    data_files = ['/mnt/Secondary/data/sabes/indy_20160426_01.mat', '/mnt/Secondary/data/sabes/loco_20170227_04.mat']

    if comm.rank == 0:

        # Form reference dataframe
        argfiles = glob.glob('/mnt/Secondary/data/timescale_dimreduc/arg*dat')
        rl = []
        for argfile in argfiles:
            with open(argfile, 'rb') as f:
                args = pickle.load(f)
            argno = argfile.split('arg')[1].split('.dat')[0]
            r = {}
            r['rf'] = args['results_file']
            r['data_file'] = args['data_file']
            r['didx'] = int(args['data_file'].split('didx_')[1].split('_')[0])
            r['dr_method'] = args['task_args']['dimreduc_method']
            r['dr_args'] = args['task_args']['dimreduc_args']
            rl.append(r)

        ref_df = pd.DataFrame(rl)

        with open('/mnt/Secondary/data/postprocessed/cca_timescales_df.pkl', 'rb') as f:
            cca_df = pickle.load(f)

        # Filter by the desired preprocesing parmaeters, and then have 2 sets of indices, one that
        # selects that data file, and another that selects the 
        ccavalid1 = apply_df_filters(cca_df, didx=didx, bin_width=25,
                                    win=1, lag=0, filter_fn='gaussian', filter_kwargs={'sigma':3})
        ccavalid2 = apply_df_filters(cca_df, didx=didx, bin_width=10,
                                    win=1, lag=0, filter_fn='gaussian', filter_kwargs={'sigma':3})
        ccavalid3 = apply_df_filters(cca_df, didx=didx, bin_width=25,
                                     win=1, lag=0, filter_fn='none')
        ccavalid4 = apply_df_filters(cca_df, didx=didx, bin_width=50,
                                     win=1, lag=0, filter_fn='none')

        if lidx == 0:
            ccadf = ccavalid1
        elif lidx == 1:
            ccadf = ccavalid2
        elif lidx == 2:
            ccadf = ccavalid3
        elif lidx == 3:
            ccadf = ccavalid4

        coefcca = ccadf.iloc[0]['ccamodel'].y_weights_

        dffca = apply_df_filters(ref_df, data_file=ccadf.iloc[0]['fl'], dr_method='LQGCA', dr_args={'T':3, 'loss_type':'trace', 'n_init':10})
        dfpca = apply_df_filters(ref_df, data_file=ccadf.iloc[0]['fl'], dr_method='PCA')

        assert(dffca.shape[0] == 1)
        assert(dfpca.shape[0] == 1)

        with open(dffca.iloc[0]['rf'], 'rb') as f:
            results = pickle.load(f)

        dffcca = pd.DataFrame(results)

        dffcca = apply_df_filters(dffcca, dim=dimval, fold_idx=0)
        coeffcca = dffcca.iloc[0]['coef']

        with open(dfpca.iloc[0]['rf'], 'rb') as f:
            results = pickle.load(f)

        dfpca = pd.DataFrame(results)
        dfpca = apply_df_filters(dfpca, dim=dimval, fold_idx=0)
        coefpca = dfpca.iloc[0]['coef']

        # dat = load_sabes('/mnt/Secondary/data/sabes/%s' % data_file)
        dat = load_sabes(data_files[didx], bin_width=cca_df.iloc[0]['bin_width'], filter_fn=cca_df.iloc[0]['filter_fn'], 
                         filter_kwargs=cca_df.iloc[0]['filter_kwargs'], region='S1')
        data_file = data_files[didx].split('/')[-1].split('.mat')[0]
        # dat = load_sabes('/mnt/sdb1/nc_data/sabes/%s' % data_file)
        dat = reach_segment_sabes(dat, data_file=data_file)
        X = np.squeeze(dat['spike_rates'])
        
        dat = load_sabes(data_files[didx], bin_width=cca_df.iloc[0]['bin_width'], filter_fn=cca_df.iloc[0]['filter_fn'], 
                         filter_kwargs=cca_df.iloc[0]['filter_kwargs'], region='M1')
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
        coefcca = None
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
    coefcca = comm.bcast(coefcca)

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
    decoding_window = 3

    # Distribute windows across ranks
    windows = np.array_split(windows, comm.size)[comm.rank]
    wr2 = np.zeros((len(windows), 1, 8))
    ypca = Y @ coefpca
    yfcca = Y @ coeffcca
    ycca = Y @ coefcca
    # Apply projection

    # Cross-validate the prediction
    for j, window in enumerate(windows):
        for fold, (train_idxs, test_idxs) in enumerate([list(KFold(n_splits=5).split(Y))[0]]): 
            # We have been given a list of windows for each transition
            if len(window) > 2:
                W = [w for win in window for w in win]
                win_min = min(W)
            else:
                win_min = window[0]

            if win_min >= 0:
                win_min = 0

            tt_train = [t for t in transition_times 
                        if t[0] >= min(train_idxs) and t[1] <= max(train_idxs) and t[0] > (lag + np.abs(win_min)) and t[1] < (X.shape[0] - lag - np.abs(win_min))]

            tt_test = [t for t in transition_times 
                    if t[0] >= min(test_idxs) and t[0] <= max(test_idxs) and t[0] > (lag + np.abs(win_min)) and t[1] < (X.shape[0] - lag - np.abs(win_min))]


            xxtrain, yytrain = apply_window(X, ypca, lag, window, tt_train, decoding_window, measure_from_end=measure_from_end,
                                        include_velocity=False, include_acc=False)
            xxtest, yytest = apply_window(X, ypca, lag, window, tt_test, decoding_window, measure_from_end=measure_from_end,
                                        include_velocity=False, include_acc=False)

            regressor = LinearRegression().fit(StandardScaler().fit_transform(np.concatenate(xxtrain)),
                                               StandardScaler().fit_transform(np.concatenate(yytrain)))

            r2train = regressor.score(StandardScaler().fit_transform(np.concatenate(xxtrain)),
                                    StandardScaler().fit_transform(np.concatenate(yytrain)))
            r2test = regressor.score(StandardScaler().fit_transform(np.concatenate(xxtest)),
                                    StandardScaler().fit_transform(np.concatenate(yytest)))

            wr2[j, fold, 0] = r2train
            wr2[j, fold, 1] = r2test


            xxtrain, yytrain = apply_window(X, yfcca, lag, window, tt_train, decoding_window, measure_from_end=measure_from_end,
                                        include_velocity=False, include_acc=False)
            xxtest, yytest = apply_window(X, yfcca, lag, window, tt_test, decoding_window, measure_from_end=measure_from_end,
                                        include_velocity=False, include_acc=False)

            regressor = LinearRegression().fit(StandardScaler().fit_transform(np.concatenate(xxtrain)),
                                        StandardScaler().fit_transform(np.concatenate(yytrain)))

            r2train = regressor.score(StandardScaler().fit_transform(np.concatenate(xxtrain)),
                                    StandardScaler().fit_transform(np.concatenate(yytrain)))
            r2test = regressor.score(StandardScaler().fit_transform(np.concatenate(xxtest)),
                                    StandardScaler().fit_transform(np.concatenate(yytest)))

            wr2[j, fold, 2] = r2train
            wr2[j, fold, 3] = r2test

            xxtrain, yytrain = apply_window(ycca, ypca, lag, window, tt_train, decoding_window, measure_from_end=measure_from_end,
                                        include_velocity=False, include_acc=False)
            xxtest, yytest = apply_window(ycca, ypca, lag, window, tt_test, decoding_window, measure_from_end=measure_from_end,
                                        include_velocity=False, include_acc=False)

            regressor = LinearRegression().fit(StandardScaler().fit_transform(np.concatenate(xxtrain)),
                                               StandardScaler().fit_transform(np.concatenate(yytrain)))

            r2train = regressor.score(StandardScaler().fit_transform(np.concatenate(xxtrain)),
                                    StandardScaler().fit_transform(np.concatenate(yytrain)))
            r2test = regressor.score(StandardScaler().fit_transform(np.concatenate(xxtest)),
                                    StandardScaler().fit_transform(np.concatenate(yytest)))

            wr2[j, fold, 4] = r2train
            wr2[j, fold, 5] = r2test


            xxtrain, yytrain = apply_window(ycca, yfcca, lag, window, tt_train, decoding_window, measure_from_end=measure_from_end,
                                        include_velocity=False, include_acc=False)
            xxtest, yytest = apply_window(ycca, yfcca, lag, window, tt_test, decoding_window, measure_from_end=measure_from_end,
                                        include_velocity=False, include_acc=False)

            regressor = LinearRegression().fit(StandardScaler().fit_transform(np.concatenate(xxtrain)),
                                               StandardScaler().fit_transform(np.concatenate(yytrain)))

            r2train = regressor.score(StandardScaler().fit_transform(np.concatenate(xxtrain)),
                                    StandardScaler().fit_transform(np.concatenate(yytrain)))
            r2test = regressor.score(StandardScaler().fit_transform(np.concatenate(xxtest)),
                                    StandardScaler().fit_transform(np.concatenate(yytest)))

            wr2[j, fold, 6] = r2train
            wr2[j, fold, 7] = r2test

    windows = np.array(windows)
    dpath = '/home/akumar/nse/neural_control/data/s1m1regvt'
    #dpath = '/mnt/sdb1/nc_data/decodingvt'
    with open('%s/didx%d_lidx%d_rank%d_%s_%d.dat' % (dpath, didx, lidx, comm.rank, filter_string, measure_from_end), 'wb') as f:
        f.write(pickle.dumps(wr2))
        f.write(pickle.dumps(error_filter))
        f.write(pickle.dumps(reach_filter))
        f.write(pickle.dumps(window_filter))
        f.write(pickle.dumps(windows))
        f.write(pickle.dumps(filter_params))
        