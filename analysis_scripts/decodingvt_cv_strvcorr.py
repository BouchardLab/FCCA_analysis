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
from sklearn.linear_model import LinearRegression

import itertools
from sklearn.model_selection import KFold

sys.path.append('/home/akumar/nse/neural_control')
from utils import apply_df_filters, calc_loadings
from loaders import load_sabes
from segmentation import reach_segment_sabes
from decoders import lr_decode_windowed, expand_state_space

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

def gen_run(name, didxs=np.arange(35)):
    with open(name, 'w') as rsh:
        rsh.write('#!/bin/bash\n')
        for di in didxs:
            rsh.write('mpirun -n 8 python decodingvt_cv_strvcorr.py %d\n' % di)


# Assign points to the closest peak
def closest_peak(pks, pnt):
    pk_dist = [np.abs(pk - pnt) for pk in pks]
    if len(pk_dist) > 0:
        return np.argmin(pk_dist)
    else:
        # Shouldn't be used as these fall outside of both single peak and multi peak reaches
        return np.nan

def max_straight_dev(trajectory, start, end):
    # Translate to the origin relative to the 1st target location
    trajectory -= start

    # straight line vector
    straight = end - start
    straight_norm = np.linalg.norm(straight)
    straight /= straight_norm

    if straight[0] == 0:
        perp = np.array([1, 0])
    elif straight[1] == 0:
        perp = np.array([0, 1])
    else:
        # Vector orthogonal to the straight line between targets
        x_orth = np.random.uniform(0, 1)
        y_orth = -1 * (straight[0] * x_orth)/straight[1]
        perp = np.array([x_orth, y_orth])
        perp /= np.linalg.norm(perp)
    
    if np.any(np.isnan(perp)):
        pdb.set_trace()
    
    m = straight[1]/straight[0]
    b = 0

    straight_dev = np.zeros(trajectory.shape[0])
    for j in range(trajectory.shape[0]):
        
        # transition is horizontal
        if m == 0:
            x_int = trajectory[j, 0]
            y_int = straight[1]
        # transition is vertical
        elif np.isnan(m) or np.isinf(m):
            x_int = straight[0]
            y_int = trajectory[j, 1]
        else:
            m1 = -1/m
            b1 = trajectory[j, 1] - m1 * trajectory[j, 0]
            # Find the intersection between the two lines
            x_int = (b - b1)/(m1 - m)
            y_int = m1 * x_int + b1
        
        straight_dev[j] = np.linalg.norm(np.array([x_int - trajectory[j, 0], y_int - trajectory[j, 1]]))

    return np.max(straight_dev), straight_dev

# Filter reaches by straight vs. corrective
def filter_reach_type(dat, lag, decoding_window):

    # Measure the distance from target
    # Calculate 
    # (1) The peaks in distance to target
    # (2) troughs in velocity
    # (3) Number of velocity peaks/velocity troughs
    dt = []
    vel = []
    dtpks = []
    dttrghs = []
    veltr = []
    velpks = []
    dtpkw = []
    velpkw = []

    # We calculate velocity by using expand state space, and shift the transition times accordingly
    Z, _ = expand_state_space([dat['behavior']], [dat['spike_rates'].squeeze()], True, True)
    # Flatten list structure imposed by expand_state_space
    Z = Z[0]

    # Shift transition times by 2: Note: this is only used to calcualte velocity peaks. apply_window is where
    # tt shift is handled for decodinvt
    transition_times = np.array([(t[0] - 2, t[1] - 2) for t in dat['transition_times']])
    for j, tt in enumerate(transition_times):        
        target_loc = dat['target_pairs'][j][1]

        vel_ = np.linalg.norm(Z[tt[0]:tt[1], 2:4], axis=1)
        dt_ = np.linalg.norm(Z[tt[0]:tt[1], 0:2] - dat['target_pairs'][j][1], axis=1)

        vel.append(vel_)
        dt.append(dt_)
        
        pks, _ = scipy.signal.find_peaks(dt_/np.max(dt_), height=0.1, prominence=0.1)

        # Require that the peak comes after the maximum value
        pks = pks[pks > np.argmax(dt_)]
        # Require that we have gotten at least halfway to the target, but not too close
        if len(pks) > 0:
            if np.any((dt_/np.max(dt_))[:pks[0]] < 0.5) and not np.any((dt_/np.max(dt_))[:pks[0]] < 0.1):
                # Get the FWHM of the peak widths
                w, _, l, r = scipy.signal.peak_widths(dt_/np.max(dt_), [pks[0]], rel_height=0.5)
                dtpkw.append(int(np.floor(l[0])))
            else:
                pks = []
                dtpkw.append(np.nan)
        else:
            dtpkw.append(np.nan)

        trghs, _ = scipy.signal.find_peaks(-1*vel_/np.max(vel_), height=-0.5)        
        dtpks.append(pks)
        veltr.append(trghs)

        pks, _ = scipy.signal.find_peaks(vel_/np.max(vel_), height=0.4, prominence=0.1)
        if len(pks) > 1:
            # Get the FWHM of the peak widths
            w, _, l, r = scipy.signal.peak_widths(vel_/np.max(vel_), [pks[1]], rel_height=0.0)
            velpkw.append(int(np.floor(l[0])))
        else:
            velpkw.append(np.nan)

        trghs, _ = scipy.signal.find_peaks(-1*dt_/np.max(dt_), height=-0.5)
        velpks.append(pks)
        dttrghs.append(trghs)

    Z = Z[:-lag, :]

    # Exclude any reaches that lie within +/- lag of the start/end of the session
    too_soon = [j for j in range(len(transition_times)) if transition_times[j][0] < lag]
    too_late = [j for j in range(len(transition_times)) if transition_times[j][1] > dat['behavior'].shape[0] - lag]

    # Straight/Direct vs. Corrective reaches
    # straight_reach = [idx for idx in range(len(dt)) if len(dtpks[idx]) == 0]
    # correction_reach = [idx for idx in range(len(dt)) if len(dtpks[idx]) > 0]
    straight_reach = [idx for idx in range(len(dt)) if len(velpks[idx]) == 1]
    correction_reach = [idx for idx in range(len(dt)) if len(velpks[idx]) > 1]

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


    print('%d Reaches' % len(transition_times))
    return dat['transition_times'], straight_reach, correction_reach

def get_recording_fold(reach_idxs, transition_times, X):

    # Folds of the recording session
    train_test_idxs = list(KFold(n_splits=5).split(X))
    fold_cnts = np.zeros(5)
    for reach_idx in reach_idxs:
        t0 = transition_times[reach_idx][0]
        t1 = transition_times[reach_idx][1]

        fold_membership = [len(set(np.arange(t0, t1)).intersection(set(train_test_idxs[j][0]))) for j in range(5)]
        fold_cnts[np.argmax(fold_membership)] += 1

    print(fold_cnts)
    return np.argmax(fold_cnts)

def behavioral_metrics(reach_set, transition_times, target_pairs, Z):
    # Calculate behavioral metrics
    dftsecondphase = []
    maxperpd = []
    perpd = []
    nonmono = []
    secondphaseduration = []

    for j, tt in enumerate(transition_times[reach_set]):

        target_loc = target_pairs[reach_set[j]][1]

        vel_ = np.linalg.norm(Z[tt[0]:tt[1], 2:4], axis=1)
        dt_ = np.linalg.norm(Z[tt[0]:tt[1], 0:2] - target_pairs[reach_set[j]][1], axis=1)
        
        pks, _ = scipy.signal.find_peaks(dt_/np.max(dt_), height=0.1, prominence=0.1)

        # Require that the peak comes after the maximum value
        dtpks = pks[pks > np.argmax(dt_)]
        # # Require that we have gotten at least halfway to the target, but not too close
        # if len(pks) > 0:
        #     if np.any((dt_/np.max(dt_))[:pks[0]] < 0.5) and not np.any((dt_/np.max(dt_))[:pks[0]] < 0.1):
        #         # Get the FWHM of the peak widths
        #         w, _, l, r = scipy.signal.peak_widths(dt_/np.max(dt_), [pks[0]], rel_height=0.5)
        #         dtpkw.append(int(np.floor(l[0])))
        #     else:
        #         pks = []
        #         dtpkw.append(np.nan)
        # else:
        #     dtpkw.append(np.nan)
        

        velpks, _ = scipy.signal.find_peaks(vel_/np.max(vel_), height=0.4, prominence=0.1)
        # if len(pks) > 1:
        #     # Get the FWHM of the peak widths
        #     w, _, l, r = scipy.signal.peak_widths(vel_/np.max(vel_), [pks[1]], rel_height=0.0)
        #     velpkw.append(int(np.floor(l[0])))
        # else:
        #     velpkw.append(np.nan)

        # velpks = pks

        # Assign transition indices according to whether they are second phase or not
        pkassign = np.array([closest_peak(velpks, t) for t in range(len(vel_))])

        # (1) Distance from target when second phase velocity peak
        dftsecondphase.append(dt_[np.argwhere(pkassign)[0][0]])

        # (2) Maximum perpendicular distance from target
        msd, sd = max_straight_dev(Z[tt[0]:tt[1], 0:2], target_pairs[reach_set[j]][0], target_pairs[reach_set[j]][1])

        maxperpd.append(msd)
        perpd.append(sd)

        # (3) Duration of second phase
        secondphaseduration.append(np.argwhere(pkassign).squeeze().size)

    return dftsecondphase, maxperpd, secondphaseduration, perpd

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('didx', type=int)

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
    window_centers = np.arange(35)
    windows = [(int(wc - window_width//2), int(wc + window_width//2)) for wc in window_centers]

    lag = 4
    decoding_window = 5


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
        target_pairs = dat['target_pairs']
        transition_times, straight_reaches, corrective_reaches = filter_reach_type(dat, lag, decoding_window)
    else:
        dat = None
        data_files = None
        coefpca = None
        coeffcca = None
        X = None
        Z = None
        transition_times = None
        straight_reaches = None
        corrective_reaches = None
        target_pairs=None

    coefpca = comm.bcast(coefpca)
    coeffcca = comm.bcast(coeffcca)

    X = comm.bcast(X)
    Z = comm.bcast(Z)

    transition_times = comm.bcast(transition_times)
    straight_reaches = comm.bcast(straight_reaches)
    corrective_reaches = comm.bcast(corrective_reaches)
    target_pairs = comm.bcast(target_pairs)

    # Distribute windows across ranks
    windows = np.array_split(windows, comm.size)[comm.rank]
    wr2 = np.zeros((len(windows), 5, 2, 6, 2))
    ntr = np.zeros((len(windows), 5, 2 ,2))

    # Store regressors to do post-hoc r2 vs behavioral features variaiblity analysis
    regressors = np.zeros((len(windows), 5, 2, 2), dtype=object)    

    # Windows x folds x straight/corrective x train/test x metric
    behavioral_metrics_array = np.zeros((len(windows), 5, 2, 2, 3), dtype=object)

    # Keep track of indices within transition time
    full_straight_reaches_train = np.zeros((len(windows), 5), dtype=object)
    full_straight_reaches_test = np.zeros((len(windows), 5), dtype=object)

    full_corrective_reaches_train = np.zeros((len(windows), 5), dtype=object)
    full_corrective_reaches_test = np.zeros((len(windows), 5), dtype=object)

    MSEtr = np.zeros((len(windows), 2, 5, 2), dtype=object)
    MSEte = np.zeros((len(windows), 2, 5, 2), dtype=object)

    # Cross-validate the prediction
    for j, window in enumerate(windows):
        # Iterate over straight vs corrective
        for k in range(2):

            # We want to cross-validate over a train/test split of the straight/corrective reaches, not the recording session. 
            # In order to choose which projection gets used, we identify the fold of the recording session with maximum overlap
            # with the training set of straight/corrective reaches
            if k == 0:
                valid_reaches = np.array(straight_reaches)
            else:
                valid_reaches = np.array(corrective_reaches)

            for fold, (train_idxs, test_idxs) in tqdm(enumerate(KFold(n_splits=5).split(valid_reaches))): 
                
                # Identify which fold of the recording session has most overlap with the train/test fold of the reaches
                recfold = get_recording_fold(valid_reaches[train_idxs], transition_times, X)
                xpca = X @ coefpca[recfold]
                xfcca = X @ coeffcca[recfold]
                r2pos, r2vel, r2acc, r2post, r2velt, r2acct, reg, ntr_, fitr, fite, msetr, msete = lr_decode_windowed(xpca, Z, lag, [window], [window], transition_times, train_idxs=valid_reaches[train_idxs],
                                                                                                    test_idxs=valid_reaches[test_idxs], decoding_window=decoding_window) 


                assert(len(fitr) == len(msetr))
                assert(len(fite) == len(msete))

                # Narrow down the msetr and msete by the fitr/fite 
                # msetr = [msetr[i] for i in range(len(msetr)) if i in fitr]
                # msete = [msete[i] for i in range(len(msete)) if i in fite]

                MSEtr[j, k, fold, 0] = msetr
                MSEte[j, k, fold, 0] = msete

                wr2[j, fold, 0, :, k] = (r2pos, r2vel, r2acc, r2post, r2velt, r2acct)
                ntr[j, fold, 0, k] = ntr_
                regressors[j, fold, 0, k] = reg

                r2pos, r2vel, r2acc, r2post, r2velt, r2acct, reg, ntr_, fitr2, fite2, msetr, msete = lr_decode_windowed(xfcca, Z, lag, [window], [window], transition_times, train_idxs=valid_reaches[train_idxs],
                                                                                                test_idxs=valid_reaches[test_idxs], decoding_window=decoding_window)
                wr2[j, fold, 1, :, k] = (r2pos, r2vel, r2acc, r2post, r2velt, r2acct)
                ntr[j, fold, 1, k] = ntr_
                regressors[j, fold, 1, k] = reg

                assert(fitr == fitr2)
                assert(fite == fite2)

                assert(len(fitr) == len(msetr))
                assert(len(fite) == len(msete))

                # Narrow down the msetr and msete by the fitr/fite 
                # msetr = [msetr[i] for i in range(len(msetr)) if i in fitr]
                # msete = [msete[i] for i in range(len(msete)) if i in fite]

                # try:
                #     assert(len(fitr) == len(msetr))
                #     assert(len(fite) == len(msete))
                # except:
                #     pdb.set_trace()

                MSEtr[j, k, fold, 1] = msetr
                MSEte[j, k, fold, 1] = msete

                # Convert fitr and fite to an indexing of the original transition times
                fitr = [valid_reaches[train_idxs][idx] for idx in fitr]
                fite = [valid_reaches[test_idxs][idx] for idx in fite]


                dftsecondphase_tr, maxperpd_tr, secondphaseduration_tr, _ = behavioral_metrics(fitr, np.array(transition_times), target_pairs, deepcopy(Z))
                dftsecondphase_te, maxperpd_te, secondphaseduration_te, _ = behavioral_metrics(fite, np.array(transition_times), target_pairs, deepcopy(Z))

                behavioral_metrics_array[j, fold, k, 0, 0] = dftsecondphase_tr
                behavioral_metrics_array[j, fold, k, 0, 1] = maxperpd_tr
                behavioral_metrics_array[j, fold, k, 0, 2] = secondphaseduration_tr

                behavioral_metrics_array[j, fold, k, 1, 0] = dftsecondphase_te
                behavioral_metrics_array[j, fold, k, 1, 1] = maxperpd_te
                behavioral_metrics_array[j, fold, k, 1, 2] = secondphaseduration_te

                if k == 0:
                    full_straight_reaches_train[j, fold] = fitr
                    full_straight_reaches_test[j, fold] = fite
                else:
                    full_corrective_reaches_train[j, fold] = fitr
                    full_corrective_reaches_test[j, fold] = fite


    windows = np.array(windows)
    dpath = '/home/akumar/nse/neural_control/data/decodingvt_cv_strcorr_behaviorsave'
    #dpath = '/mnt/sdb1/nc_data/decodingvt'
    with open('%s/didx%d_rank%d.dat' % (dpath, didx, comm.rank), 'wb') as f:
        f.write(pickle.dumps(wr2))
        f.write(pickle.dumps(ntr))
        f.write(pickle.dumps(straight_reaches))
        f.write(pickle.dumps(corrective_reaches))
        f.write(pickle.dumps(full_straight_reaches_train))
        f.write(pickle.dumps(full_straight_reaches_test))
        f.write(pickle.dumps(full_corrective_reaches_train))
        f.write(pickle.dumps(full_corrective_reaches_test))
        f.write(pickle.dumps(regressors))
        f.write(pickle.dumps(MSEtr))
        f.write(pickle.dumps(MSEte))   
        f.write(pickle.dumps(behavioral_metrics_array))