import sys
import scipy 
from scipy.stats import ortho_group
import pdb
import numpy as np
import os
import time
import pickle
import itertools
from tqdm import tqdm
from sklearn.model_selection import KFold
import argparse

from statsmodels.tsa.statespace.varmax import VARMAX

from neurosim.models.ssr import StateSpaceRealization as SSR
from subspaces import estimate_autocorrelation, SubspaceIdentification, IteratedStableEstimator
from pyuoi.linear_model.var import VAR, _form_var_problem

from mpi4py import MPI
from schwimmbad import MPIPool, SerialPool

from em import StableStateSpaceML
from subspaces import estimate_autocorrelation, IteratedStableEstimator, SubspaceIdentification


class PoolWorker():

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def VARfit(self, task_tuple):
        state_dim = self.state_dim
        obs_dim = self.obs_dim

        # Unpack task
        if len(task_tuple) == 2:
            task_tuple, comm = task_tuple
        else:
            comm = None               

        model_rep, trajectory_rep, \
        N, fold_idx, model_order = task_tuple
        
        # Generate synthetic data with the right seeds
        random_state = np.random.RandomState(seed=model_rep)
        A = random_state.normal(scale=1/(1.7 * np.sqrt(state_dim)), size=(state_dim, state_dim))
        while max(np.abs(np.linalg.eigvals(A))) > 0.99:
            A = random_state.normal(scale=1/(1.7 * np.sqrt(state_dim)), size=(state_dim, state_dim))

        scipy_randgen = scipy.stats.ortho_group
        scipy_randgen.random_state = random_state

        C = scipy_randgen.rvs(state_dim)[:, 0:obs_dim].T
        ssr = SSR(A=A, B=np.eye(A.shape[0]), C=C)
        ccm0 = ssr.autocorrelation(5)

        random_state = np.random.RandomState(seed=trajectory_rep)
        y = ssr.trajectory(N, rand_state=random_state)

        train_idxs, test_idxs = list(KFold(5).split(np.arange(y.shape[0])))[fold_idx]
        ccm1 = estimate_autocorrelation(y[train_idxs], 5)
        ccm2 = estimate_autocorrelation(y[test_idxs], 5)

        result = {}
        result['fold_idx'] = fold_idx
        result['N'] = N
        result['trajectory_rep'] = trajectory_rep
        result['model_rep'] = model_rep
        result['true_params'] = (A, C)
        result['autocorr_true'] = ccm0
        result['autocorr_train'] = ccm1
        result['autocorr_test'] = ccm2
        result['model_order'] = model_order

        print('Fitting VAR models')
        varmodel1 =  VAR(order=model_order, estimator='ols')
        varmodel1.fit(y)

        varmodel2 = VAR(order=model_order, estimator='uoi', penalty='scad', fit_type='union_only')
        varmodel2.fit(y)

        result['ols_coef'] = varmodel1.coef_
        result['uoi_coef'] = varmodel2.coef_
        with open('%s/%d_%d_%d_%d_%d.dat' % (self.result_dir, model_rep, trajectory_rep, N, fold_idx, model_order), 'wb') as f:
            f.write(pickle.dumps(result))

    def stableMLfit(self, task_tuple):

        state_dim = self.state_dim
        obs_dim = self.obs_dim

        # Unpack task
        if len(task_tuple) == 2:
            task_tuple, comm = task_tuple
        else:
            comm = None               

        model_rep, trajectory_rep, \
        N, fold_idx, model_order = task_tuple
        
        # Generate synthetic data with the right seeds
        random_state = np.random.RandomState(seed=model_rep)
        A = random_state.normal(scale=1/(1.7 * np.sqrt(state_dim)), size=(state_dim, state_dim))
        while max(np.abs(np.linalg.eigvals(A))) > 0.99:
            A = random_state.normal(scale=1/(1.7 * np.sqrt(state_dim)), size=(state_dim, state_dim))

        scipy_randgen = scipy.stats.ortho_group
        scipy_randgen.random_state = random_state

        C = scipy_randgen.rvs(state_dim)[:, 0:obs_dim].T
        ssr = SSR(A=A, B=np.eye(A.shape[0]), C=C)
        ccm0 = ssr.autocorrelation(5)

        random_state = np.random.RandomState(seed=trajectory_rep)
        y = ssr.trajectory(N, rand_state=random_state)

        train_idxs, test_idxs = list(KFold(5).split(np.arange(y.shape[0])))[fold_idx]
        ccm1 = estimate_autocorrelation(y[train_idxs], 5)
        ccm2 = estimate_autocorrelation(y[test_idxs], 5)

        result = {}
        result['fold_idx'] = fold_idx
        result['N'] = N
        result['trajectory_rep'] = trajectory_rep
        result['model_rep'] = model_rep
        result['true_params'] = (A, C)
        result['autocorr_true'] = ccm0
        result['autocorr_train'] = ccm1
        result['autocorr_test'] = ccm2
        result['model_order'] = model_order
        # Fit subspace identification explicitly first
        print('Fitting SSID')
        ssid = SubspaceIdentification(T=3, estimator=IteratedStableEstimator)
        A, C, Cbar, L0, Q, R, S = ssid.identify(y[train_idxs], order=model_order)        
        Sigma0 = scipy.linalg.solve_discrete_lyapunov(A, Q)
        result['ssid_coef'] = (A, C, Cbar, L0, Q, R, S) 

        print('Fitting StateSpaceML')            

        try:
            ssm = StableStateSpaceML(max_iter=50, init_strategy='manual')
            ssm.fit(y[train_idxs], state_dim=model_order, Ainit=A, Cinit=C, Qinit=Q, Rinit=R,
                    x0 = np.zeros(A.shape[0]), Sigma0 = Sigma0)
            result['MLcoef'] = (ssm.A, ssm.C, ssm.R, ssm.x0, ssm.Sigma0)
            print(time.time() - t0)
        except:
            result['coef'] = 'Fitting failed'

        with open('%s/%d_%d_%d_%d_%d.dat' % (self.result_dir, model_rep, trajectory_rep, N, fold_idx, model_order), 'wb') as f:
            f.write(pickle.dumps(result))
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('result_dir')
    parser.add_argument('--serial', action='store_true')
    parser.add_argument('--fit_type', dest='analysis_type', default='stableml')

    cmd_args = parser.parse_args()

    state_dim = 10
    obs_dim = 20

    model_reps = 1
    trajectory_reps = 1

    N = [int(5e3)]
    # Test (1) How the different model selection criteria perform in MOE identification 
    # (2) Whether forward vs. reverse time estimates give systematically better predictions
    # (3) OLS vs. Ridge vs. IteratedStability in terms of fit to cross-correlation matrices

    n_folds = 5
    #model_orders = np.arange(15, 27, 2)
    model_orders = np.array([5, 10])

    # Form outer product over tasks
    tasks = list(itertools.product(np.arange(model_reps), np.arange(trajectory_reps), N, range(n_folds), model_orders))

    worker = PoolWorker(state_dim=state_dim, obs_dim=obs_dim, result_dir=cmd_args.result_dir)

    # VERY IMPORTANT: Once pool is created, the workers wait for instructions, so must proceed directly to map
    if not cmd_args.serial:
        comm = MPI.COMM_WORLD
    else:
        comm = None

    results_dir = cmd_args.result_dir
    if comm is not None:
        # Create folder for processes to write in
        if comm.rank == 0:
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
    else: 
        if not os.path.exists(results_dir):
           os.makedirs(results_dir)

    if comm is not None:
        tasks = comm.bcast(tasks)
        print('%d Tasks Remaining' % len(tasks))
        pool = MPIPool(comm)
    else:
        pool = SerialPool()

    if len(tasks) > 0:
        pool.map(worker.stableMLfit, tasks)
    pool.close()