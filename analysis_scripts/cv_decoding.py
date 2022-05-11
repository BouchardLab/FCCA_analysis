import pdb
import numpy as np
import matplotlib.pyplot as plt
import scipy 
import time
import glob
import pickle
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LinearRegression, LogisticRegression
import itertools
from sklearn.model_selection import KFold
import sys
sys.path.append('/home/akumar/nse/neural_control')
from utils import apply_df_filters, calc_loadings

from loaders import load_cv
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from mpi4py import MPI


with open('/home/akumar/nse/neural_control/data/cv_dimreduc_df2.dat', 'rb') as f:
    cv_df = pickle.load(f)

cv_df = pd.DataFrame(cv_df)

KCA_args = [{'T':10, 'causal_weights':(1, 0), 'n_init':5}, {'T':20, 'causal_weights':(1, 0), 'n_init':5}]
LQGCA_args = [{'T':10, 'loss_type':'trace', 'n_init':5}, {'T':20, 'loss_type':'trace', 'n_init':5}]
DCA_args = [{'T': 10, 'n_init': 5}, {'T':20, 'n_init':5}]


folds = np.arange(5)
dimvals = np.unique(cv_df['dim'].values)
dimreduc_methods = ['DCA10', 'DCA20', 'KCA10', 'KCA20', 'LQGCA10', 'LQGCA20', 'PCA']

# Do the decoding
decoding_list =[]

dat = load_cv('/mnt/Secondary/data/EC2_hg.h5')
comm = MPI.COMM_WORLD


dr_method = dimreduc_methods[comm.rank]
for f, fold in enumerate(folds):
    for d, dimval in tqdm(enumerate(dimvals)):            
        if 'KCA' in dr_method:
            df_ = apply_df_filters(cv_df, dimreduc_method='KCA', fold_idx=fold, dim=dimval)
            # Further filter by dimreduc_args
            if dr_method == 'KCA10':
                df_ = apply_df_filters(df_, dimreduc_args=KCA_args[0])
            elif dr_method == 'KCA20':
                df_ = apply_df_filters(df_, dimreduc_args=KCA_args[1])
        elif 'LQGCA' in dr_method:
            df_ = apply_df_filters(cv_df, dimreduc_method='LQGCA', fold_idx=fold, dim=dimval)
            # Further filter by dimreduc_args
            if dr_method == 'LQGCA10':
                df_ = apply_df_filters(df_, dimreduc_args=LQGCA_args[0])
            elif dr_method == 'LQGCA20':
                df_ = apply_df_filters(df_, dimreduc_args=LQGCA_args[1])
        elif 'DCA' in dr_method:
            df_ = apply_df_filters(cv_df, dimreduc_method='DCA', fold_idx=fold, dim=dimval)
            if dr_method == 'DCA10':
                df_ = apply_df_filters(df_, dimreduc_args=DCA_args[0])
            elif dr_method == 'DCA20':
                df_ = apply_df_filters(df_, dimreduc_args=DCA_args[1])

        else:
            df_ = apply_df_filters(cv_df, dimreduc_method=dr_method, fold_idx=fold, dim=dimval)

        assert(df_.shape[0] == 1)
        
        Xtrain = dat['spike_rates'][df_.iloc[0]['train_idxs']] @ df_.iloc[0]['coef']
        Ytrain = dat['behavior'][df_.iloc[0]['train_idxs']]

        Xtest = dat['spike_rates'][df_.iloc[0]['test_idxs']] @ df_.iloc[0]['coef']
        Ytest = dat['behavior'][df_.iloc[0]['test_idxs']]
        
        classifier = LogisticRegression(multi_class='multinomial', max_iter=500, solver='lbfgs', tol=1e-4).fit(Xtrain.reshape((Xtrain.shape[0], -1)), Ytrain)
        acc = classifier.score(Xtest.reshape((Xtest.shape[0], -1)), Ytest)
        
        result = {}
        result['dr_method'] = dr_method
        result['fold'] = fold
        result['dimval'] = d
        result['acc'] = acc
        result['classifier_coef'] = classifier.coef_
        decoding_list.append(result)

with open('decoding_list2_%d.dat' % comm.rank, 'wb') as f:
    f.write(pickle.dumps(decoding_list))