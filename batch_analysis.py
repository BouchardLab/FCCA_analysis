import os
import gc
import argparse
import time
import pickle
import glob
import itertools
import numpy as np
import scipy
from mpi4py import MPI
from mpi_utils.ndarray import Bcast_from_root
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from pyuoi.linear_model.var import VAR
from dca.dca import DynamicalComponentsAnalysis as DCA
from dca.cov_util import form_lag_matrix
from dca_research.kca import KalmanComponentsAnalysis as KCA
from dca_research.lqg import LQGComponentsAnalysis as LQGCA
from dca.methods_comparison import SlowFeatureAnalysis as SFA


# from neurosim.models.ssr import StateSpaceRealization as SSR

from schwimmbad import MPIPool, SerialPool

#from decoders import kf_dim_analysis
from loaders import load_sabes, load_shenoy, load_peanut, load_cv
from mpi_loaders import mpi_load_shenoy
from decoders import kf_decoder, lr_decoder
import pdb
import glob

def load_preprocessed(path, **kwargs):
    with open(path, 'rb') as f:
        dat = pickle.load(f)
    return dat

LOADER_DICT = {'sabes': load_sabes, 'shenoy': mpi_load_shenoy, 'peanut': load_peanut, 'cv':load_cv, 'preprocessed': load_preprocessed}
DECODER_DICT = {'lr': lr_decoder, 'kf': kf_decoder}

class PCA_wrapper():

    def __init__(self, d, lag=1, marginal_only=False):
        self.pcaobj = PCA()
        self.dim = d
        assert(lag > 0 and isinstance(lag, int))
        self.lag = lag
        self.marginal_only = marginal_only

    def fit(self, X):

        if self.lag > 1:
            X = form_lag_matrix(X, self.lag)

        if np.ndim(X) == 3:
            X = np.reshape(X, (-1, X.shape[-1]))

        # Relying only on the marginal variances, the method reduces to just returning a projection sorted along
        # these marginal variances
        if self.marginal_only:            
            var = np.var(X, axis=0)
            self.var = var

            var_ordering = np.argsort(var)[::-1]

            self.coef_ = np.zeros((X.shape[-1], self.dim))
            for i in range(self.dim):
                self.coef_[var_ordering[i], i] = 1
        else:
            self.pcaobj.fit(X)
            self.coef_ = self.pcaobj.components_.T[:, 0:self.dim]

    def score(self):
        if self.marginal_only:
            var_ordered = np.sort(self.var)[::-1]
            return sum(var_ordered[0:self.dim])/sum(self.var)
        else:
            return sum(self.pcaobj.explained_variance_ratio_[0:self.dim])

DIMREDUC_DICT = {'PCA': PCA_wrapper, 'DCA': DCA, 'KCA': KCA, 'LQGCA': LQGCA}

# Check which tasks have already been completed and prune from the task list
def prune_dimreduc_tasks(tasks, results_folder):
    # If the results file exists, there is nothing left to do
    if os.path.exists('%s.dat' % results_folder):
        return []

    completed_files = glob.glob('%s/*.dat' % results_folder)
    dim_and_folds = []
    for completed_file in completed_files:
        dim = int(completed_file.split('dim_')[1].split('_')[0])
        fold_idx = int(completed_file.split('fold_')[1].split('.dat')[0])

        dim_and_folds.append((dim, fold_idx))

    to_do = []
    for task in tasks:
        train_test_tuple, dim, method, method_args, results_folder = task
        fold_idx, train_idxs, test_idxs = train_test_tuple

        if (dim, fold_idx) not in dim_and_folds:
            to_do.append(task)

    return to_do

# Check which tasks have already been completed and prune from the task list
def prune_decoding_tasks(tasks, results_folder):

    completed_files = glob.glob('%s/*.dat' % results_folder)
    dim_and_folds = []
    for completed_file in completed_files:
        dim = int(completed_file.split('dim_')[1].split('_')[0])
        fold_idx = int(completed_file.split('fold_')[1].split('.dat')[0])

        dim_and_folds.append((dim, fold_idx))

    to_do = []
    for task in tasks:
        dim, fold_idx, \
        dimreduc_results, decoder, results_folder = task

        if (dim, fold_idx) not in dim_and_folds:
            to_do.append(task)

    return to_do

# Currently only thing we check for fold_idx
def prune_var_tasks(tasks, results_folder):
    completed_files = glob.glob('%s/*.dat' % results_folder)
    folds = []
    for completed_file in completed_files:
        pdb.set_trace()


# Tiered communicators for use with schwimmbad
def comm_split(comm, ncomms):

    if comm is not None:    
        subcomm = None
        split_ranks = None
    else:
        split_ranks = None

    return split_ranks

def init_comm(comm, split_ranks):

    ncomms = len(split_ranks)
    color = [i for i in np.arange(ncomms) if comm.rank in split_ranks[i]][0]
    return subcomm

def consolidate(results_folder, results_file, comm):
    # Consolidate files into a single data file
    if comm is not None:
        if comm.rank == 0:
            data_files = glob.glob('%s/*.dat' % results_folder)
            results_dict_list = []
            for data_file in data_files:
                with open(data_file, 'rb') as f:
                    results_dict = pickle.load(f)
                    results_dict_list.append(results_dict)

            with open(results_file, 'wb') as f:
                f.write(pickle.dumps(results_dict_list))
    else:
        data_files = glob.glob('%s/*.dat' % results_folder)
        results_dict_list = []
        for data_file in data_files:
            with open(data_file, 'rb') as f:
                results_dict = pickle.load(f)
                results_dict_list.append(results_dict)
        
        with open(results_file, 'wb') as f:    
            f.write(pickle.dumps(results_dict_list))

def load_data(loader, data_file, loader_args, comm, broadcast_behavior=False):

    print(loader_args)
    if comm is None:
        dat = LOADER_DICT[loader](data_file, **loader_args)
        spike_rates = np.squeeze(dat['spike_rates'])

    else:
        if comm.rank == 0:
            dat = LOADER_DICT[loader](data_file, **loader_args)
            if dat['spike_rates'].dtype == 'object':
                spike_rates = dat['spike_rates']
            else:
                spike_rates = np.ascontiguousarray(np.squeeze(dat['spike_rates']), dtype=float)
        else:
            spike_rates = None

        try:
            spike_rates = Bcast_from_root(spike_rates, comm)
        except KeyError:
            spike_rates = comm.bcast(spike_rates)

    # # Make global variable - saves memory when using Schwimmbad as the data can be accessed by workers without
    # being sent again (which duplicates it)
    globals()['X'] =  spike_rates

    # Repeat for behavior, if requested
    if broadcast_behavior:
        if comm is None:
            behavior = dat['behavior']
        else:
            if comm.rank == 0:
                behavior = dat['behavior']
            else:
                behavior = None
            behavior = comm.bcast(behavior)

        globals()['Y'] = behavior


class PoolWorker():

    # Initialize the worker with the data so it does not have to be broadcast by
    # pool.map
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def parametric_dimreduc(self, task_tuple):
        if len(task_tuple) == 2:
            task_tuple, comm = task_tuple
        else:
            comm = None             

        train_test_tuple, dim, method, method_args, results_folder = task_tuple
        A, fold_idx, train_idxs, test_idxs = train_test_tuple
        print('Dim: %d, Fold idx: %d' % (dim, fold_idx))

        if A.shape[1] <= dim:
            results_dict = {}
            results_dict['dim'] = dim
            results_dict['fold_idx'] = fold_idx
            results_dict['train_idxs'] = train_idxs
            results_dict['test_idxs'] = test_idxs

            results_dict['dimreduc_method'] = method
            results_dict['dimreduc_args'] = method_args
            results_dict['coef'] = np.nan
            results_dict['score'] = np.nan               
        else:

            ssr = SSR(A=A, B=np.eye(A.shape[0]), C=np.eye(A.shape[0]))
            if method == 'PCA':
                eig, U = np.linalg.eig(ssr.P)
                eigorder = np.argsort(np.abs(eigorder))[::-1]
                U = U[:, eigorder]
                coef = U[:, 0:dim]
                score = np.sum(eig[eigorder][0:dim])/np.trace(ssr.P)
            else:
                dimreducmodel = DIMREDUC_DICT[method](d=dim, **method_args)
                dimreducmodel.cross_covs = torch.tensor(ssr.autocorrelation(2 * method_args['T'] + 1))
                # Fit OLS VAR, DCA, PCA, and SFA
                dimreducmodel.fit()
                coef = dimreducmodel.coef_
                score = dimreducmodel.score()
            
        # Organize results in a dictionary structure
        results_dict = {}
        results_dict['dim'] = dim
        results_dict['fold_idx'] = fold_idx
        results_dict['train_idxs'] = train_idxs
        results_dict['test_idxs'] = test_idxs

        results_dict['dimreduc_method'] = method
        results_dict['dimreduc_args'] = method_args
        results_dict['coef'] = coef
        results_dict['score'] = score

        # Write to file, will later be concatenated by the main process
        file_name = 'dim_%d_fold_%d.dat' % (dim, fold_idx)
        with open('%s/%s' % (results_folder, file_name), 'wb') as f:
            f.write(pickle.dumps(results_dict))
        # Cannot return None or else schwimmbad with hang (lol!)
        return 0

    def dimreduc(self, task_tuple):
        ### NOTE: Somehow we have modified schwimmbad to pass in comm here
        # This could be useful if subcommunicators are needed
        if len(task_tuple) == 2:
            task_tuple, comm = task_tuple
        else:
            comm = None             

        train_test_tuple, dim, method, method_args, results_folder = task_tuple
        fold_idx, train_idxs, test_idxs = train_test_tuple
        print('Dim: %d, Fold idx: %d' % (dim, fold_idx))

        # X is either of shape (n_time, n_dof) or (n_trials,). In the latter case
        X = globals()['X']

        # dim_val is too high
        dim_error = False
        if np.ndim(X) == 2:
            if X.shape[1] <= dim:
                dim_error = True
        else:
            if X[0].shape[1] <= dim:
                dim_error = True

        if dim_error:
            results_dict = {}
            results_dict['dim'] = dim
            results_dict['fold_idx'] = fold_idx
            results_dict['train_idxs'] = train_idxs
            results_dict['test_idxs'] = test_idxs

            results_dict['dimreduc_method'] = method
            results_dict['dimreduc_args'] = method_args
            results_dict['coef'] = np.nan
            results_dict['score'] = np.nan               
        else:
            X_train = X[train_idxs, ...]

            if X.dtype == 'object':
                # subtract the cross condition mean
                cross_cond_mean = np.mean([np.mean(x_, axis=0) for x_ in X_train], axis=0)      
                X_train = [x_ - cross_cond_mean for x_ in X_train]
            else:            
                # Save memory
                X_train -= np.concatenate(X_train).mean(axis=0, keepdims=True)
                # X_mean = np.concatenate(X_train).mean(axis=0, keepdims=True)
                # X_train_ctd = np.array([Xi - X_mean for Xi in X_train])

            # Fit OLS VAR, DCA, PCA, and SFA

            dimreducmodel = DIMREDUC_DICT[method](d=dim, **method_args)
            dimreducmodel.fit(X_train)

            coef = dimreducmodel.coef_
            score = dimreducmodel.score()
            
        # Organize results in a dictionary structure
        results_dict = {}
        results_dict['dim'] = dim
        results_dict['fold_idx'] = fold_idx
        results_dict['train_idxs'] = train_idxs
        results_dict['test_idxs'] = test_idxs
        results_dict['dimreduc_method'] = method
        results_dict['dimreduc_args'] = method_args
        results_dict['coef'] = coef
        results_dict['score'] = score

        # Write to file, will later be concatenated by the main process
        file_name = 'dim_%d_fold_%d.dat' % (dim, fold_idx)
        with open('%s/%s' % (results_folder, file_name), 'wb') as f:
            f.write(pickle.dumps(results_dict))
        # Cannot return None or else schwimmbad with hang (lol!)
        return 0

    def decoding(self, task_tuple):

        # Unpack task tuple
        if len(task_tuple) == 2:
            task_tuple, comm = task_tuple
        else:
            comm = None               

        dim_val, fold_idx, \
        dimreduc_results, decoder, results_folder = task_tuple

        print('Working on %d, %d' % (dim_val, fold_idx))

        # Find the index in dimreduc_results that matches the fold_idx and dim_vals
        # that have been assigned to us
        dim_fold_tuples = [(result['dim'], result['fold_idx']) for result in dimreduc_results]
        dimreduc_idx = dim_fold_tuples.index((dim_val, fold_idx))
        coef_ = dimreduc_results[dimreduc_idx]['coef']

        X = globals()['X']
        Y = globals()['Y']

        # Project the (train and test) data onto the subspace and train and score the requested decoder
        train_idxs = dimreduc_results[dimreduc_idx]['train_idxs']
        test_idxs = dimreduc_results[dimreduc_idx]['test_idxs']

        Ytrain = Y[train_idxs]
        Ytest = Y[test_idxs]

        Xtrain = X[train_idxs]
        Xtest = X[test_idxs]

        # if np.ndim(Xtrain) == 2:
        #     Xtrain = form_lag_matrix(Xtrain, lag)
        #     Xtest = form_lag_matrix(Xtest, lag)
        # else:
        #     Xtrain = np.array([form_lag_matrix(xx, lag) for xx in Xtrain])
        #     Xtest = np.array([form_lag_matrix(xx, lag) for xx in Xtest])
        if np.ndim(Xtrain) == 2:
            Xtrain = Xtrain @ coef_
            Xtest = Xtest @ coef_
        else:
            Xtrain = [xx @ coef_ for xx in Xtrain]
            Xtest = [xx @ coef_ for xx in Xtest]
            Ytrain = [yy for yy in Ytrain]
            Ytest = [yy for yy in Ytest]

        r2_pos, r2_vel, r2_acc, decoder_obj = DECODER_DICT[decoder['method']](Xtest, Xtrain, Ytest, Ytrain, **decoder['args'])

        # Compile results into a dictionary. First copy over everything from the dimreduc results so that we no longer
        # have to refer to the dimreduc results
        results_dict = {}

        for key, value in dimreduc_results[dimreduc_idx].items(): 
            results_dict[key] = value

        results_dict['dim'] = dim_val
        results_dict['fold_idx'] = fold_idx
        results_dict['decoder'] = decoder['method']
        results_dict['decoder_args'] = decoder['args']
        results_dict['decoder_obj'] = decoder_obj 
        results_dict['r2'] = [r2_pos, r2_vel, r2_acc]

        # Save to file
        with open('%s/dim_%d_fold_%d.dat' % \
                (results_folder, dim_val, fold_idx), 'wb') as f:
            f.write(pickle.dumps(results_dict))

def parametric_dimreduc_(X, dim_vals, 
                        n_folds, comm,
                        method, method_args, 
                        split_ranks, results_file,
                        resume=False):

    # Follow the same logic as dimreduc_, but generate the autocorrelation sequence from SSR
    if comm is not None:
        # Create folder for processes to write in
        results_folder = results_file.split('.')[0]
        if comm.rank == 0:
            if not os.path.exists(results_folder):
                os.makedirs(results_folder)
    else: 
        results_folder = results_file.split('.')[0]        
        if not os.path.exists(results_folder):
           os.makedirs(results_folder)

    if ((comm is not None) and comm.rank == 0) or (comm is None):
        # Do cv_splits here
        cv = KFold(n_folds, shuffle=False)

        train_test_idxs = list(cv.split(X))

        # Fit VAR(1) on rank 0
        A = []
        for i in range(len(train_test_idxs)):
            Xtrain = X[train_test_idxs[i][0]]
            varmodel = VAR(order=1, estimator='ols')
            varmodel.fit(Xtrain)
            A.append(np.squeeze(varmodel.coef_))            

        data_tasks = [(A[idx], idx) + train_test_split for idx, train_test_split
                    in enumerate(train_test_idxs)]    
        tasks = itertools.product(data_tasks, dim_vals)
        tasks = [task + (method, method_args, results_folder) for task in tasks]    
        # Check which tasks have already been completed
        if resume:
            tasks = prune_tasks(tasks, results_folder)
    else:
        tasks = None
        A = None

    # Initialize Pool worker with data
    worker = PoolWorker()

    # VERY IMPORTANT: Once pool is created, the workers wait for instructions, so must proceed directly to map
    if comm is not None:
        tasks = comm.bcast(tasks)
        print('%d Tasks Remaining' % len(tasks))
        pool = MPIPool(comm, subgroups=split_ranks)
    else:
        pool = SerialPool()

    if len(tasks) > 0:
        pool.map(worker.dimreduc, tasks)
    pool.close() 

    # Consolidate files into a single data file
    if comm is not None:
        if comm.rank == 0:
            data_files = glob.glob('%s/*.dat' % results_folder)
            results_dict_list = []
            for data_file in data_files:
                with open(data_file, 'rb') as f:
                    results_dict = pickle.load(f)
                    results_dict_list.append(results_dict)

            with open(results_file, 'wb') as f:
                f.write(pickle.dumps(results_dict_list))
    else:
        data_files = glob.glob('%s/*.dat' % results_folder)
        results_dict_list = []
        for data_file in data_files:
            with open(data_file, 'rb') as f:
                results_dict = pickle.load(f)
                results_dict_list.append(results_dict)

        with open(results_file, 'wb') as f:
            f.write(pickle.dumps(results_dict_list))


def dimreduc_(dim_vals, 
              n_folds, comm,
              method, method_args, 
              split_ranks, results_file,
              resume=False):

    print(method)
    if comm is not None:
        # Create folder for processes to write in
        results_folder = results_file.split('.')[0]
        if comm.rank == 0:
            if not os.path.exists(results_folder):
                os.makedirs(results_folder)
    else: 
        results_folder = results_file.split('.')[0]        
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)

    if comm is None:
        # X is either of shape (n_time, n_dof) or (n_trials, n_time, n_dof)
        X = globals()['X']

        # Do cv_splits here
        if n_folds > 1:
            cv = KFold(n_folds, shuffle=False)
            train_test_idxs = list(cv.split(X))
        else:
            # No cross-validation split
            train_test_idxs = [(list(range(X.shape[0])), [])]

        data_tasks = [(idx,) + train_test_split for idx, train_test_split
                    in enumerate(train_test_idxs)]    
        tasks = itertools.product(data_tasks, dim_vals)
        tasks = [task + (method, method_args, results_folder) for task in tasks]
        # Check which tasks have already been completed
        if resume:
            tasks = prune_dimreduc_tasks(tasks, results_folder)

    else:
        if comm.rank == 0:
            # X is either of shape (n_time, n_dof) or (n_trials, n_time, n_dof)
            X = globals()['X']

            # Do cv_splits here
            if n_folds > 1:
                cv = KFold(n_folds, shuffle=False)
                train_test_idxs = list(cv.split(X))
            else:
                # No cross-validation split
                train_test_idxs = [(list(range(X.shape[0])), [])]

            data_tasks = [(idx,) + train_test_split for idx, train_test_split
                        in enumerate(train_test_idxs)]    
            tasks = itertools.product(data_tasks, dim_vals)
            tasks = [task + (method, method_args, results_folder) for task in tasks]
            # Check which tasks have already been completed
            if resume:
                tasks = prune_dimreduc_tasks(tasks, results_folder)
        else:
            tasks = None

    # Initialize Pool worker with data
    worker = PoolWorker()

    # VERY IMPORTANT: Once pool is created, the workers wait for instructions, so must proceed directly to map
    if comm is not None:
        tasks = comm.bcast(tasks)
        print('%d Tasks Remaining' % len(tasks))
        pool = MPIPool(comm, subgroups=split_ranks)
    else:
        pool = SerialPool()

    if len(tasks) > 0:
        pool.map(worker.dimreduc, tasks)
    pool.close()

    consolidate(results_folder, results_file, comm)

def decoding_(dimreduc_file, decoder, data_path,
              comm, split_ranks, results_file, 
              resume=False):

    if comm is not None:
        # Create folder for processes to write in
        results_folder = results_file.split('.')[0]
        if comm.rank == 0:
            if not os.path.exists(results_folder):
                os.makedirs(results_folder)
    else: 
        results_folder = results_file.split('.')[0]        
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)


    # Look for an arg file in the same folder as the dimreduc_file
    dimreduc_path = '/'.join(dimreduc_file.split('/')[:-1])
    dimreduc_fileno = int(dimreduc_file.split('_')[-1].split('.dat')[0])
    argfile_path = '%s/arg%d.dat' % (dimreduc_path, dimreduc_fileno)

    # Dimreduc args provide loader information
    with open(argfile_path, 'rb') as f:
        args = pickle.load(f) 

    data_file_name = args['data_file'].split('/')[-1]
    data_file_path = '%s/%s' % (data_path, data_file_name)

    # Don't do this one
    if data_file_name == 'trialtype0.dat':
        return

    load_data(args['loader'], args['data_file'], args['loader_args'], comm, broadcast_behavior=True)
    
    if comm is None:
        with open(dimreduc_file, 'rb') as f:
            dimreduc_results = pickle.load(f)

        # Pass in for manual override for use in cleanup
        if dim_vals is None:
            dim_vals = args['task_args']['dim_vals']
        if fold_idxs is None:
            n_folds = args['task_args']['n_folds']
            fold_idxs = np.arange(n_folds)

        # Assemble task arguments
        tasks = itertools.product(dim_vals, fold_idxs)
        tasks = [task + (dimreduc_results, decoder, results_folder) 
                for task in tasks]
        if resume:
            tasks = prune_decoding_tasks(tasks, results_folder)
    else:
        if comm.rank == 0:
            with open(dimreduc_file, 'rb') as f:
                dimreduc_results = pickle.load(f)

            # Pass in for manual override for use in cleanup
            dim_vals = args['task_args']['dim_vals']
            n_folds = args['task_args']['n_folds']
            fold_idxs = np.arange(n_folds)

            # Assemble task arguments
            tasks = itertools.product(dim_vals, fold_idxs)
            tasks = [task + (dimreduc_results, decoder, results_folder) 
                    for task in tasks]
            if resume:
                tasks = prune_decoding_tasks(tasks, results_folder)
        else:
            tasks = None

    # Initialize Pool worker with data
    worker = PoolWorker()

    # VERY IMPORTANT: Once pool is created, the workers wait for instructions, so must proceed directly to map
    if comm is not None:
        tasks = comm.bcast(tasks)
        print('%d Tasks Remaining' % len(tasks))
        pool = MPIPool(comm, subgroups=split_ranks)
    else:
        pool = SerialPool()

    if len(tasks) > 0:
        pool.map(worker.decoding, tasks)
    pool.close()

    consolidate(results_folder, results_file, comm)

def main(cmd_args, args):
    total_start = time.time() 

    # MPI split
    if not cmd_args.serial:
        comm = MPI.COMM_WORLD
        ncomms = cmd_args.ncomms
    else:
        comm = None
        ncomms = None

    if cmd_args.analysis_type == 'var':

        # ncomms=comm.size

        # If resume, check whether the completed .dat file exists, and if so, skip
        if cmd_args.resume:
            if os.path.exists(args['results_file']):
                print('Nothing to do')
                return

        load_data(args['loader'], args['data_file'], args['loader_args'], comm)

        X = globals()['X']
        if args['task_args']['fold_idx'] > 0:
            split_idxs = list(KFold(5).split(X))
            train_idxs, test_idxs = split_idxs[args['task_args']['fold_idx']]
        else:
            train_idxs = np.arange(X.shape[0])
        savepath = args['results_file'].split('.dat')[0]

        # Pop off fold_idx from task_args - the rest can be passed into the VAR object
        del args['task_args']['fold_idx']

        args['task_args']['savepath'] = savepath

        estimator = VAR(comm=comm, ncomms=ncomms, **args['task_args'])  
        # Need to do distributed save and provide filepath
        t0 = time.time()
        estimator.fit(X[train_idxs])

        # Need to save at this point as the var object did not
        if not estimator.distributed_save:
            if comm.rank == 0:
                with open(args['results_file'], 'wb') as f:
                    f.write(pickle.dumps(estimator.coef_))            

    elif cmd_args.analysis_type == 'dimreduc':
        load_data(args['loader'], args['data_file'], args['loader_args'], comm)        
        split_ranks = comm_split(comm, ncomms)
        dimreduc_(dim_vals = args['task_args']['dim_vals'],
                  n_folds = args['task_args']['n_folds'], 
                  method = args['task_args']['dimreduc_method'],
                  method_args = args['task_args']['dimreduc_args'],
                  comm=comm, split_ranks=split_ranks,
                  results_file = args['results_file'],
                  resume=cmd_args.resume)

    elif cmd_args.analysis_type == 'decoding':

        split_ranks = comm_split(comm, ncomms)
        decoding_(dimreduc_file=args['task_args']['dimreduc_file'], 
                  decoder=args['task_args']['decoder'],
                  data_path = args['data_path'],
                  comm=comm, split_ranks=split_ranks,
                  results_file=args['results_file'],
                  resume=cmd_args.resume)

    total_time = time.time() - total_start
    print(total_time)

if __name__ == '__main__':

    total_start = time.time()

    ###### Command line arguments #######
    parser = argparse.ArgumentParser()

    # Dictionary with more detailed argument dictionary that is loaded via pickle
    parser.add_argument('arg_file')
    parser.add_argument('--analysis_type', dest='analysis_type')
    # parser.add_argument('--data_file', dest='data_file')
    parser.add_argument('--serial', dest='serial', action='store_true')
    parser.add_argument('--ncomms', type=int, default=1)
    parser.add_argument('--resume', action='store_true')
    # parser.add_argument('--var_order', type=int, default=-1)
    # parser.add_argument('--self_regress', default=False)
    # parser.add_argument('--bin_size', type=float, default=0.05)
    # parser.add_argument('--spike_threshold', type=float, default=100)
    # parser.add_argument('--decimate', default=False)
    # parser.add_argument('--lag', type=int, default=0)
    cmd_args = parser.parse_args()

    ####### Load arg file ################
    with open(cmd_args.arg_file, 'rb') as f:
        args = pickle.load(f)

    #######################################
    # If provided a list of arguments, call main for each entry
    if type(args) == dict:
        main(cmd_args, args)
    else:
        for arg in args:
            try:
                main(cmd_args, arg)
            except:
                continue
