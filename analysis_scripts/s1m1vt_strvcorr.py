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
from copy import deepcopy
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.preprocessing import StandardScaler

import itertools
from sklearn.model_selection import KFold

sys.path.append('/home/akumar/nse/neural_control')
from utils import apply_df_filters, calc_loadings
from loaders import load_sabes
from segmentation import reach_segment_sabes
from decoders import lr_decode_windowed, apply_window

from decodingvt_cv_strvcorr import filter_reach_type, behavioral_metrics, get_recording_fold

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

    lag = 0
    decoding_window = 5

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
        Z = dat['behavior']
        target_pairs = dat['target_pairs']
        transition_times, straight_reaches, corrective_reaches = filter_reach_type(dat, lag, decoding_window)
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
        Z = None
        transition_times = None
        straight_reaches = None
        corrective_reaches = None
        target_pairs = None

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
    # Behavior
    Z = comm.bcast(Z)

    transition_times = comm.bcast(transition_times)
    straight_reaches = comm.bcast(straight_reaches)
    corrective_reaches = comm.bcast(corrective_reaches)
    target_pairs = comm.bcast(target_pairs)

    # Distribute windows across ranks
    windows = np.array_split(windows, comm.size)[comm.rank]

    # Regressions that are independent of CCA dimension
    ccadims = np.array([4])

    # S1 -> M1 regressions
    wr2_S1M1 = np.zeros((len(windows), 5, 18, ccadims.size, 2))
    # M1 -> S1 regressions
    wr2_M1S1 = np.zeros((len(windows), 5, 18, ccadims.size, 2))
    
    # M1 -> M1 regressions
    wr2_M1 = np.zeros((len(windows), 5, 12, ccadims.size, 2))
    # S1 -> S1 regressions
    wr2_S1 = np.zeros((len(windows), 5, 12, ccadims.size, 2))
    
    # windows x folds x straight/corrective x train/test x n_behavioral_metrics
    behavioral_metrics_array = np.zeros((len(windows), 5, 2, 2, 3), dtype=object)

    full_straight_reaches_train = np.zeros((len(windows), 5), dtype=object)
    full_straight_reaches_test = np.zeros((len(windows), 5), dtype=object)

    full_corrective_reaches_train = np.zeros((len(windows), 5), dtype=object)
    full_corrective_reaches_test = np.zeros((len(windows), 5), dtype=object)

    # Cross-validate the prediction
    for j, window in enumerate(windows):
        for str_corr_idx in range(2):

            if str_corr_idx == 0:
                valid_reaches = np.array(straight_reaches)
            if str_corr_idx == 1:
                valid_reaches = np.array(corrective_reaches)

            for fold, (train_idxs, test_idxs) in enumerate(KFold(n_splits=5).split(valid_reaches)): 
                for k, ccadim in enumerate(ccadims):

                    # We have been given a list of windows for each transition
                    if len(window) > 2:
                        W = [w for win in window for w in win]
                        win_min = min(W)
                    else:
                        win_min = window[0]

                    if win_min >= 0:
                        win_min = 0

                    recfold = get_recording_fold(valid_reaches[train_idxs], transition_times, X)

                    ypca = Y @ coefpca[recfold]
                    yfcca = Y @ coeffcca[recfold]

                    xpca = X @ coefpcaS1[recfold]
                    xfcca = X @ coeffccaS1[recfold]

                    if ccadim == -1:
                        ycca = Y @ coefcca[recfold][:, 0:1]
                        xcca = X
                    else:
                        ycca = Y @ coefcca[recfold][:, 0:ccadim]
                        xcca = X @ coefcca_x[recfold][:, 0:ccadim]
                        if ccadim == 1:
                            ycca = np.reshape(ycca, (-1, 1))
                            xcca = np.reshape(xcca, (-1, 1))

                    print(ccadim)
                    print(ycca.shape)
                    print(xcca.shape)


                    # Might need to rework this since we are doing the train/test split over valid reaches
                    tt_train = np.array(transition_times)[valid_reaches[train_idxs]]
                    tt_test = np.array(transition_times)[valid_reaches[test_idxs]]


                    # S1 -> M1 regression
                    idx = 0
                    for features in [xcca, xpca, xfcca]:
                        for targets in[ycca, ypca, yfcca]:
                            xxtrain, yytrain, _, fitr = apply_window(features, targets, lag, [window], tt_train, decoding_window, 
                                                            include_velocity=False, include_acc=False)
                            xxtest, yytest, _, fite = apply_window(features, targets, lag, [window], tt_test, decoding_window, 
                                                            include_velocity=False, include_acc=False)

                            regressor = RidgeCV().fit(StandardScaler().fit_transform(np.concatenate(xxtrain)),
                                                    StandardScaler().fit_transform(np.concatenate(yytrain)))

                            r2train = regressor.score(StandardScaler().fit_transform(np.concatenate(xxtrain)),
                                                    StandardScaler().fit_transform(np.concatenate(yytrain)))
                            r2test = regressor.score(StandardScaler().fit_transform(np.concatenate(xxtest)),
                                                    StandardScaler().fit_transform(np.concatenate(yytest)))

                            wr2_S1M1[j, fold, idx, k, str_corr_idx] = r2train
                            wr2_S1M1[j, fold, idx + 1, k, str_corr_idx] = r2test
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

                            wr2_M1S1[j, fold, idx, k, str_corr_idx] = r2train
                            wr2_M1S1[j, fold, idx + 1, k, str_corr_idx] = r2test
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

                            wr2_M1[j, fold, idx, k, str_corr_idx] = r2train
                            wr2_M1[j, fold, idx + 1, k, str_corr_idx] = r2test
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

                            wr2_S1[j, fold, idx, k, str_corr_idx] = r2train
                            wr2_S1[j, fold, idx + 1, k, str_corr_idx] = r2test
                            idx += 2

                    # Convert fitr and fite to an indexing of the original transition times
                    fitr = [valid_reaches[train_idxs][idx] for idx in fitr]
                    fite = [valid_reaches[test_idxs][idx] for idx in fite]

                    dftsecondphase_tr, maxperpd_tr, secondphaseduration_tr, _ = behavioral_metrics(fitr, np.array(transition_times), target_pairs, deepcopy(Z))
                    dftsecondphase_te, maxperpd_te, secondphaseduration_te, _ = behavioral_metrics(fite, np.array(transition_times), target_pairs, deepcopy(Z))

                    behavioral_metrics_array[j, fold, str_corr_idx, 0, 0] = dftsecondphase_tr
                    behavioral_metrics_array[j, fold, str_corr_idx, 0, 1] = maxperpd_tr
                    behavioral_metrics_array[j, fold, str_corr_idx, 0, 2] = secondphaseduration_tr

                    behavioral_metrics_array[j, fold, str_corr_idx, 1, 0] = dftsecondphase_te
                    behavioral_metrics_array[j, fold, str_corr_idx, 1, 1] = maxperpd_te
                    behavioral_metrics_array[j, fold, str_corr_idx, 1, 2] = secondphaseduration_te

                    if str_corr_idx == 0:
                        full_straight_reaches_train[j, fold] = fitr
                        full_straight_reaches_test[j, fold] = fite
                    else:
                        full_corrective_reaches_train[j, fold] = fitr
                        full_corrective_reaches_test[j, fold] = fite


    windows = np.array(windows)
    dpath = '/home/akumar/nse/neural_control/data/s1m1regvt_allpairs_strvcorr'
    #dpath = '/mnt/sdb1/nc_data/decodingvt'
    with open('%s/didx%d_rank%d.dat' % (dpath, didx, comm.rank), 'wb') as f:
        f.write(pickle.dumps(wr2_S1))
        f.write(pickle.dumps(wr2_M1))
        f.write(pickle.dumps(wr2_M1S1))
        f.write(pickle.dumps(wr2_S1M1))
        f.write(pickle.dumps(windows))
        f.write(pickle.dumps(straight_reaches))
        f.write(pickle.dumps(corrective_reaches))
        f.write(pickle.dumps(full_straight_reaches_train))
        f.write(pickle.dumps(full_straight_reaches_test))
        f.write(pickle.dumps(full_corrective_reaches_train))
        f.write(pickle.dumps(full_corrective_reaches_test))
        f.write(pickle.dumps(behavioral_metrics_array))