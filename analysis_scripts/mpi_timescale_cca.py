import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import pickle
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.model_selection import KFold
from dca.cov_util import form_lag_matrix, calc_cross_cov_mats_from_data
import glob
import pdb
from statsmodels.tsa import stattools
from dca_research.lqg import LQGComponentsAnalysis as LQGCA

from mpi4py import MPI

comm = MPI.COMM_WORLD

# Analysis 4: Canonical correlation analysis
ccal = []
lags = np.array([0])
windows = np.array([1])

fls = ['loco_20170210_03.pkl',
 'loco_20170213_02.pkl',
 'loco_20170215_02.pkl',
 'loco_20170227_04.pkl',
 'loco_20170228_02.pkl',
 'loco_20170301_05.pkl',
 'loco_20170302_02.pkl',
 'indy_20160426_01.pkl']

fls = ['/mnt/Secondary/data/sabes_tmp50/%s' % f for f in fls]

fls = np.array_split(fls, comm.size)[comm.rank]

for fl in tqdm(fls):
    with open(fl, 'rb') as f:
        datM1 = pickle.load(f)
        datS1 = pickle.load(f)
#        lparam = pickle.load(f)

    Y = datM1['spike_rates'].squeeze()
    X = datS1['spike_rates'].squeeze()
    print('Hello')
    for k, lag in enumerate(lags):
        for w, window in enumerate(windows):
            for fold_idx, (train_idxs, test_idxs) in enumerate(KFold(n_splits=5).split(X)):
                t0 = time.time()
                r = {}
                # r['dfile'] = int(fl.split('didx_')[1].split('_')[0])
                # r['bin_width'] = lparam[0]
                # r['filter_fn'] = lparam[1]['filter_fn']
                # r['filter_kwargs'] = lparam[1]['filter_kwargs']

                # Hard-coded for 25 ms run
                r['dfile'] = fl.split('/')[1].split('.pkl')[0]
                print('Warning, hard-coding bin width')
                r['bin_width'] = 50
                r['filter_fn'] = 'none'
                r['filter_kwargs'] = {}

                r['lag'] = lag
                r['win'] = window
                r['fold_idx'] = fold_idx
                x = X[train_idxs]
                y = Y[train_idxs]

                # Apply window and lag relative to each other
                if lag != 0:
                    x = x[:-lag, :]
                    y = x[lag:, :]

                if window > 1:
                    x = form_lag_matrix(x, window)
                    y = form_lag_matrix(y, window)

                ccamodel = CCA(n_components=min(50, min(x.shape[-1], y.shape[-1])))
                t0 = time.time()
                ccamodel.fit(x, y)
                print(time.time() - t0)
                r['ccamodel'] = ccamodel
                r['fl'] = fl
                # ccal.append(r)

                # (1) Append the result to the open results file
                with open('/mnt/Secondary/data/mpi_cc50cv_all/rank%d.pkl' % comm.rank, 'ab') as f:
                    f.write(pickle.dumps(r))
