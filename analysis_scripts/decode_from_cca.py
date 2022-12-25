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
sys.path.append('/home/akumar/nse/neural_control')
from decoders import lr_decoder
from utils import apply_df_filters

from mpi4py import MPI

# Load cca dataframe

dw = 5
lags = [-6, -4, -2, -1, 0, 1, 2, 4, 6]
dims = np.arange(1, 31, 2)

with open('/mnt/Secondary/data/postprocessed/sabes_cca50_cvall.dat', 'rb') as f:
    ccadf = pickle.load(f)

fls = np.unique(ccadf['fl'].values)

r2 = np.zeros((fls.size, 5, len(lags), dims.size, 2, 3))
ss_coef = np.zeros((fls.size, 5, len(lags), dims.size, 2), dtype=object)
n_coef = np.zeros((fls.size, 5, len(lags), dims.size, 2), dtype=object)

for i, fl in tqdm(enumerate(fls)):
    
    with open(fl, 'rb') as f:
        datM1 = pickle.load(f)
        datS1 = pickle.load(f)

    x = datS1['spike_rates'].squeeze()
    y = datM1['spike_rates'].squeeze()
    z = datM1['behavior'].squeeze()

    train_test_split = list(KFold(n_splits=5).split(x))

    for f, fold in tqdm(enumerate(range(5))):
    
        train_idxs = train_test_split[f][0]
        test_idxs = train_test_split[f][1]

        dc_ = apply_df_filters(ccadf, fold_idx=fold, fl=fl)
        assert(dc_.shape[0] == 1)
        coef_x = dc_.iloc[0]['ccamodel'].x_rotations_
        coef_y = dc_.iloc[0]['ccamodel'].y_rotations_

        for j, lag_ in enumerate(lags):
            for k, dim in enumerate(dims):
                xtrain = x[train_idxs] @ coef_x[:, 0:dim]
                xtest = x[test_idxs] @ coef_x[:, 0:dim]

                ytrain = y[train_idxs] @ coef_y[:, 0:dim]
                ytest = y[test_idxs] @ coef_y[:, 0:dim]

                ztrain = z[train_idxs]
                ztest = z[test_idxs]
                
                r2pos_s1, r2vel_s1, r2acc_s1, models1 = lr_decoder(xtest, xtrain, ztest, ztrain, lag_, lag_, decoding_window=dw)
                r2pos_m1, r2vel_m1, r2acc_m1, modelm1 = lr_decoder(ytest, ytrain, ztest, ztrain, lag_, lag_, decoding_window=dw)

                r2[i, f, j, k, 0, :] = [r2pos_s1, r2vel_s1, r2acc_s1]
                r2[i, f, j, k, 1, :] = [r2pos_m1, r2vel_m1, r2acc_m1]

                # reshape coefficients as to take into account the decoding window. Take the norm contribution to velocity decoding only

                ss_coef[i, f, j, k, 0] = np.sqrt(np.sum(np.sum(np.power(models1.coef_.reshape((6, dim, dw), order='F')[2:4, ...], 2), axis=-1), axis=0))
                ss_coef[i, f, j, k, 1] = np.sqrt(np.sum(np.sum(np.power(modelm1.coef_.reshape((6, dim, dw), order='F')[2:4, ...], 2), axis=-1), axis=0))

                try:
                    cascaded_coef1 = np.einsum('ij,hjk->hik', coef_x[:, 0:dim], models1.coef_.reshape((6, dim, dw), order='F')[2:4, ...])
                    n_coef[i, f, j, k, 0] = np.sqrt(np.sum(np.sum(np.power(cascaded_coef1, 2), axis=-1), axis=0))
                    cascaded_coef2 = np.einsum('ij,hjk->hik', coef_y[:, 0:dim], modelm1.coef_.reshape((6, dim, dw), order='F')[2:4, ...])
                    n_coef[i, f, j, k, 1] = np.sqrt(np.sum(np.sum(np.power(cascaded_coef2, 2), axis=-1), axis=0))
                except:
                    pdb.set_trace()

    with open('cca_behavioral_decoding.dat', 'wb') as f:
        f.write(pickle.dumps(r2))
        f.write(pickle.dumps(ss_coef))
        f.write(pickle.dumps(n_coef))