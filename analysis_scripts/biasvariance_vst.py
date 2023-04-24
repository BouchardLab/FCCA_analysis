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
        # Note the lower 1 threshold
        dat = reach_segment_sabes(dat, data_file=data_file.split('.mat')[0], err_thresh=0.9)

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

        # Intersection

        # We calculate velocity by using expand state space, and shift the transition times accordingly
        Z, _ = expand_state_space([dat['behavior']], [dat['spike_rates'].squeeze()], True, True)
        # Flatten list structure imposed by expand_state_space
        Z = Z[0]

        # Shift transition times by 2
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

        X = dat['spike_rates'].squeeze()

        # Apply lag
        X = X[lag:, :]
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

        # Segment the corrective reaches by pre/post corrective movement
        pkassign = get_peak_assignments_vel(vel)
        #pkassign = get_peak_assignments(velocity_seg, dtpkw)

        # Could add offsets so that time is measured with respect to different features for each reach
        offsets = np.zeros(len(transition_times))

        # for idx in correction_reach:
        #     # dt_ = dt[idx]
        #     # # Normalize by max
        #     # dt_ /= np.max(dt_)

        #     # dt_0 = dt_[:dtpks[idx][0]]
            
        #     # # Steepest decline
        #     # dt_00 = dt_0[np.argmin(np.diff(dt_0)):]
        #     # zero = np.argmin(dt_00) + np.argmin(np.diff(dt_0))
        #     pka = pkassign[idx]
        #     #offsets[idx] = np.argwhere(np.diff(pka))[0][0]
            
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
    train_windows = np.array_split(train_windows, comm.size)[comm.rank]
    test_windows = np.array_split(test_windows, comm.size)[comm.rank]

    print('%d straight, %d correction' % (len(straight_reach), len(correction_reach)))

    Xpca = X @ coefpca
    Xlqg = X @ coeffcca

    bias = np.zeros((len(test_windows), 8, 6))
    var = np.zeros((len(test_windows), 8, 6))
    mse = np.zeros((len(test_windows), 8, 6))
    wr2 = np.zeros((len(test_windows), 8, 6))
    ntr = np.zeros((len(test_windows), 8, 2))

    # Pool calculation of the decoder across multiple windows in the train set
    for j, train_window in enumerate(train_windows):
        print(j)
        mse_, bias_, var_, ntr_ =  lr_bv_windowed(Xpca, Z, 0, train_window, test_windows[j], transition_times, straight_reach, correction_reach, 
                                                    pkassign, decoding_window=decoding_window, n_boots=200, random_seed=500, offsets=offsets, norm=True)

        bias[j, 0, :] = bias_
        var[j, 0, :] = var_
        mse[j, 0, :] = mse_
        ntr[j, 0, 0] = ntr_

        mse_, bias_, var_, ntr_ =  lr_bv_windowed(Xlqg, Z, 0, train_window, test_windows[j], transition_times, straight_reach, correction_reach, 
                                            pkassign, decoding_window=decoding_window, n_boots=200, random_seed=500, offsets=offsets, norm=True)
        bias[j, 1, :] = bias_
        var[j, 1, :] = var_
        mse[j, 1, :] = mse_
        ntr[j, 1, 0] = ntr_

        # Also keep track of the r2
        r2pos, r2vel, r2acc, r2post, r2velt, r2acct, _, ntr_, _, _ = lr_decode_windowed(Xpca, Z, 0, train_window, test_windows[j], transition_times, train_idxs=straight_reach,
                                                                                            test_idxs=correction_reach, decoding_window=decoding_window, pkassign=pkassign, offsets=offsets) 
        wr2[j, 0, :] = (r2pos, r2vel, r2acc, r2post, r2velt, r2acct)
        ntr[j, 0, 1] = ntr_

        r2pos, r2vel, r2acc, r2post, r2velt, r2acct, _, ntr_, _, _ = lr_decode_windowed(Xlqg, Z, 0, train_window, test_windows[j], transition_times, train_idxs=straight_reach,
                                                                                        test_idxs=correction_reach, decoding_window=decoding_window, 
                                                                                        pkassign=pkassign, offsets=offsets)
        wr2[j, 1, :] = (r2pos, r2vel, r2acc, r2post, r2velt, r2acct)
        ntr[j, 1, 1] = ntr_

        ############################################################# Second, we train on both single and multi peak reaches and test on the latter half of multi peak
        # Feed into lr_decoder. Use lag of 0 since we already applied, but feed in the decoding window
        mse_, bias_, var_, ntr_ =  lr_bv_windowed(Xpca, Z, 0, train_window, test_windows[j], transition_times, np.sort(np.concatenate([straight_reach, correction_reach])), 
                                            correction_reach, pkassign, decoding_window=decoding_window, n_boots=200, random_seed=500, apply_pk_to_train=True, offsets=offsets, norm=True)
        bias[j, 2, :] = bias_
        var[j, 2, :] = var_
        mse[j, 2, :] = mse_
        ntr[j, 2, 0] = ntr_

        mse_, bias_, var_, ntr_ =  lr_bv_windowed(Xlqg, Z, 0, train_window, test_windows[j], transition_times, np.sort(np.concatenate([straight_reach, correction_reach])), 
                                            correction_reach, pkassign, decoding_window=decoding_window, n_boots=200, random_seed=500, apply_pk_to_train=True, offsets=offsets, norm=True)
        bias[j, 3, :] = bias_
        var[j, 3, :] = var_
        mse[j, 3, :] = mse_
        ntr[j, 3, 0] = ntr_

        # Also keep track of the r2

        r2pos, r2vel, r2acc, r2post, r2velt, r2acct, _, ntr_, _, _ = lr_decode_windowed(Xpca, Z, 0, train_window, test_windows[j], transition_times, train_idxs=np.sort(np.concatenate([straight_reach, correction_reach])),
                                                                                            test_idxs=correction_reach, decoding_window=decoding_window, 
                                                                                            pkassign=pkassign,  apply_pk_to_train=True, offsets=offsets) 
        wr2[j, 2, :] = (r2pos, r2vel, r2acc, r2post, r2velt, r2acct)
        ntr[j, 2, 1] = ntr_
        

        r2pos, r2vel, r2acc, r2post, r2velt, r2acct, _, ntr_, _, _ = lr_decode_windowed(Xlqg, Z, 0, train_window, test_windows[j], transition_times, train_idxs=np.sort(np.concatenate([straight_reach, correction_reach])),
                                                                                        test_idxs=correction_reach, decoding_window=decoding_window, 
                                                                                        pkassign=pkassign,  apply_pk_to_train=True, offsets=offsets)
        wr2[j, 3, :] = (r2pos, r2vel, r2acc, r2post, r2velt, r2acct)
        ntr[j, 3, 1] = ntr_

        ############################################################## Third, we train on multi peak only during the first portion and test on the latter half
        # Feed into lr_decoder. Use lag of 0 since we already applied, but feed in the decoding window
        mse_, bias_, var_, ntr_ =  lr_bv_windowed(Xpca, Z, 0, train_window, test_windows[j], transition_times, correction_reach, 
                                            correction_reach, pkassign, decoding_window=decoding_window, n_boots=200, random_seed=500,  apply_pk_to_train=True, offsets=offsets, norm=True)
        bias[j, 4, :] = bias_
        var[j, 4, :] = var_
        mse[j, 4, :] = mse_
        ntr[j, 4, 0] = ntr_

        mse_, bias_, var_, ntr_ =  lr_bv_windowed(Xlqg, Z, 0, train_window, test_windows[j], transition_times, correction_reach, 
                                            correction_reach, pkassign, decoding_window=decoding_window, n_boots=200, random_seed=500,  apply_pk_to_train=True, offsets=offsets, norm=True)
        bias[j, 5, :] = bias_
        var[j, 5, :] = var_
        mse[j, 5, :] = mse_
        ntr[j, 5, 0] = ntr_

        # Also keep track of the r2

        r2pos, r2vel, r2acc, r2post, r2velt, r2acct, _, ntr_, _, _ = lr_decode_windowed(Xpca, Z, 0, train_window, test_windows[j], transition_times, train_idxs=correction_reach,
                                                                                            test_idxs=correction_reach, decoding_window=decoding_window, 
                                                                                            pkassign=pkassign,  apply_pk_to_train=True, offsets=offsets) 
        wr2[j, 4, :] = (r2pos, r2vel, r2acc, r2post, r2velt, r2acct)
        ntr[j, 4, 1] = ntr_

        r2pos, r2vel, r2acc, r2post, r2velt, r2acct, _, ntr_, _, _ = lr_decode_windowed(Xlqg, Z, 0, train_window, test_windows[j], transition_times, train_idxs=correction_reach,
                                                                                        test_idxs=correction_reach, decoding_window=decoding_window, 
                                                                                        pkassign=pkassign,  apply_pk_to_train=True, offsets=offsets)
        wr2[j, 5, :] = (r2pos, r2vel, r2acc, r2post, r2velt, r2acct)
        ntr[j, 5, 1] = ntr_

        ############################################################## 
        # Lastly: Do a train/test split within ~~single phase reaches~~, multiphase only.
        # This requires 2 things: Set pkassign to 1 across the whole time series, and then split straight reaches into train and 
        # test indices
        pkassign_tmp = np.array([1 * np.invert(pka.astype(bool)) for pka in pkassign])

        # Do an 80/20 train test split
        # rnd = np.random.RandomState(seed=1234)
        # train_idxs = rnd.choice(np.arange(len(correction_reach)), int(0.8 * len(correction_reach)), replace=False)
        # train_single_phase = np.array(correction_reach)[train_idxs]
        # test_single_phase = np.array(correction_reach)[np.setdiff1d(np.arange(len(correction_reach)), train_idxs)]
        mse_, bias_, var_, ntr_ =  lr_bv_windowed(Xpca, Z, 0, train_window, test_windows[j], transition_times, straight_reach, 
                                                    correction_reach, pkassign_tmp, decoding_window=decoding_window, n_boots=200, 
                                                    random_seed=500,  apply_pk_to_train=False, offsets=offsets, norm=True)
        bias[j, 6, :] = bias_
        var[j, 6, :] = var_
        mse[j, 6, :] = mse_
        ntr[j, 6, 0] = ntr_

        mse_, bias_, var_, ntr_ =  lr_bv_windowed(Xlqg, Z, 0, train_window, test_windows[j], transition_times, straight_reach, 
                                                 correction_reach, pkassign_tmp, decoding_window=decoding_window, n_boots=200, random_seed=500,  apply_pk_to_train=False, offsets=offsets, norm=True)
        bias[j, 7, :] = bias_
        var[j, 7, :] = var_
        mse[j, 7, :] = mse_
        ntr[j, 7, 0] = ntr_

        # Also keep track of the r2

        r2pos, r2vel, r2acc, r2post, r2velt, r2acct, _, ntr_, _, _ = lr_decode_windowed(Xpca, Z, 0, train_window, test_windows[j], transition_times, train_idxs=straight_reach,
                                                                                            test_idxs=correction_reach, decoding_window=decoding_window, 
                                                                                            pkassign=pkassign_tmp,  apply_pk_to_train=False, offsets=offsets) 
        wr2[j, 6, :] = (r2pos, r2vel, r2acc, r2post, r2velt, r2acct)
        ntr[j, 6, 1] = ntr_

        r2pos, r2vel, r2acc, r2post, r2velt, r2acct, _, ntr_, _, _ = lr_decode_windowed(Xlqg, Z, 0, train_window, test_windows[j], transition_times, train_idxs=straight_reach,
                                                                                        test_idxs=correction_reach, decoding_window=decoding_window, 
                                                                                        pkassign=pkassign_tmp,  apply_pk_to_train=False, offsets=offsets)
        wr2[j, 7, :] = (r2pos, r2vel, r2acc, r2post, r2velt, r2acct)
        ntr[j, 7, 1] = ntr_
                        
    dpath = '/home/akumar/nse/neural_control/data/biasvariance_vst_norm2'
    #dpath = '/mnt/sdb1/nc_data/decodingvt'
    with open('%s/didx%d_rank%d.dat' % (dpath, didx, comm.rank), 'wb') as f:
        f.write(pickle.dumps(bias))
        f.write(pickle.dumps(var))
        f.write(pickle.dumps(mse))
        f.write(pickle.dumps(wr2))
        f.write(pickle.dumps(train_windows))
        f.write(pickle.dumps(test_windows))
        f.write(pickle.dumps(offsets))
        f.write(pickle.dumps(dimval))
        f.write(pickle.dumps(ntr))