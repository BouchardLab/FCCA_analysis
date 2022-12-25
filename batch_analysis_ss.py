import sys, os
import gc
import argparse
import time
import pickle
import glob
import itertools
import numpy as np
import scipy
from sklearn.model_selection import KFold


#from decoders import kf_dim_analysis
from loaders import load_sabes, load_shenoy, load_peanut, load_cv
from mpi_loaders import mpi_load_shenoy

from subspaces import SubspaceIdentification, IteratedStableEstimator
from em import StableStateSpaceML


LOADER_DICT = {'sabes': load_sabes, 'shenoy': mpi_load_shenoy, 'peanut': load_peanut, 'cv':load_cv}

def main(args):
    total_start = time.time() 

    dat = LOADER_DICT[args['loader']](args['data_file'], **args['loader_args'])
    X = np.squeeze(dat['spike_rates'])

    split_idxs = list(KFold(5).split(X))
    train_idxs, test_idxs = split_idxs[args['task_args']['fold_idx']]
    savepath = args['results_file'].split('.dat')[0]

    result = {}
    result['fold_idx'] = args['task_args']['fold_idx']
    model_order = args['task_args']['model_order']
    result['model_order'] = model_order

    # Fit subspace identification explicitly first
    print('Fitting SSID')
    ssid = SubspaceIdentification(T=args['task_args']['T'], estimator=IteratedStableEstimator)
    A, C, Cbar, L0, Q, R, S = ssid.identify(X[train_idxs], order=model_order)        
    Sigma0 = scipy.linalg.solve_discrete_lyapunov(A, Q)
    result['ssid_coef'] = (A, C, Cbar, L0, Q, R, S) 

    print('Fitting StateSpaceML')            

    ssm = StableStateSpaceML(max_iter=50, init_strategy='manual')
    ssm.fit(X[train_idxs], state_dim=model_order, Ainit=A, Cinit=C, Qinit=Q, Rinit=R,
            x0 = np.zeros(A.shape[0]), Sigma0 = Sigma0)
    result['MLcoef'] = (ssm.A, ssm.C, ssm.R, ssm.x0, ssm.Sigma0)
    
    total_time = time.time() - total_start
    print(total_time)


if __name__ == '__main__':

    total_start = time.time()

    ###### Command line arguments #######
    
    # Dictionary with more detailed argument dictionary that is loaded via pickle
    arg_file = sys.argv[1]

    ####### Load arg file ################
    with open(arg_file, 'rb') as f:
        args = pickle.load(f)

    #######################################
    main(args)
