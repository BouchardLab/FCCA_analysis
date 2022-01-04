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

from dca.dca import DynamicalComponentsAnalysis as DCA
from dca_research.kca import KalmanComponentsAnalysis as KCA
from dca.methods_comparison import SlowFeatureAnalysis as SFA
from pyuoi.linear_model.var import VAR, form_lag_matrix

from schwimmbad import MPIPool, SerialPool

#from decoders import kf_dim_analysis
from loaders import load_sabes, load_shenoy, load_peanut
from mpi_loaders import mpi_load_shenoy
from decoders import kf_decoder, lr_decoder
import pdb
import glob

LOADER_DICT = {'sabes': load_sabes, 'shenoy': mpi_load_shenoy, 'peanut': load_peanut}
DECODER_DICT = {'lr': lr_decoder, 'kf': kf_decoder}

# Check which tasks have already been completed and prune from the task list
def prune_dimreduc_tasks(tasks, results_folder):

    completed_files = glob.glob('%s/*.dat' % results_folder)
    dim_and_folds = []
    for completed_file in completed_files:
        dim = int(completed_file.split('dim_')[1].split('_')[0])
        fold_idx = int(completed_file.split('fold_')[1].split('.dat')[0])

        dim_and_folds.append((dim, fold_idx))

    to_do = []
    for task in tasks:
        train_test_tuple, dim_tuple, T, results_folder = task
        fold_idx, train_idxs, test_idxs = train_test_tuple
        dim, fit_all = dim_tuple

        if (dim, fold_idx) not in dim_and_folds:
            to_do.append(task)

    return to_do

def form_companion(w, u=None):
    order = w.shape[0]
    size = w.shape[1]
    I = np.eye(size * (order - 1))
    wcomp = np.block([list(w), [I, np.zeros((size * (order - 1), size))]])

    if u is not None:
        ucomp = [[u]]
        for i in range(order - 1):
            ucomp.append([np.zeros((size, size))])
        ucomp = np.block(ucomp)

        return wcomp, ucomp
    else:
        return wcomp

# Tiered communicators for use with schwimmbad
def comm_split(comm, ncomms):

    if comm is not None:
        if ncomms == 1:
            split_ranks = None
        else:
            rank = comm.rank
            numproc = comm.Get_size()
            ranks = np.arange(numproc)
            split_ranks = np.array_split(ranks, ncomms)
    else:
        subcomm = None
        split_ranks = None

    return split_ranks

def init_comm(comm, split_ranks):

    ncomms = len(split_ranks)
    color = [i for i in np.arange(ncomms) if comm.rank in split_ranks[i]][0]
    subcomm_roots = [split_ranks[i][0] for i in np.arange(ncomms)]
    subcomm = comm.Split(color, comm.rank)

    return subcomm

class PoolWorker():

    # Initialize the worker with the data so it does not have to be broadcast by
    # pool.map
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def dimreduc(self, task_tuple):
        ### NOTE: Somehow we have modified schwimmbad to pass in comm here
        # This could be useful if subcommunicators are needed
        if len(task_tuple) == 2:
            task_tuple, comm = task_tuple
        else:
            comm = None             

        train_test_tuple, dim_tuple, T, results_folder = task_tuple
        fold_idx, train_idxs, test_idxs = train_test_tuple
        dim, fit_all = dim_tuple
        print('Dim: %d, Fold idx: %d' % (dim, fold_idx))

        # X is either of shape (n_time, n_dof) or (n_trials, n_time, n_dof)
        X = globals()['X']

        # dim_val is too high
        if X.shape[1] <= dim:
            dcacoef = np.nan
            pi = np.nan
            kcacoef = np.nan
            mmse = np.nan
        else:
            X_train = X[train_idxs, ...]
            # Save memory
            X_train -= np.concatenate(X_train).mean(axis=0, keepdims=True)
            # X_mean = np.concatenate(X_train).mean(axis=0, keepdims=True)
            # X_train_ctd = np.array([Xi - X_mean for Xi in X_train])
            # Fit OLS VAR, DCA, PCA, and SFA
            dcamodel = DCA(d=dim, T=T).fit(X_train)
            dcacoef = dcamodel.coef_
            pi = dcamodel.score(X_train)

            kcamodel = KCA(d=dim, T=T, causal_weights=(1, 0), project_mmse=False).fit(X_train)
            kcacoef = kcamodel.coef_
            mmse = kcamodel.score()

            # Don't need to fit the rest of these for all dimensions

            if fit_all:
                # varmodel = VAR(estimator='ols', order=ols_order)
                # varmodel.fit(np.array(X_train_ctd))
                # A = form_companion(varmodel.coef_)
                # W = scipy.linalg.solve_discrete_lyapunov(A, A.shape[0])

                # For SFA and PCA, we concatenate trial structure (if any)
                if np.ndim(X_train) == 3:
                    sfa_model = SFA(1).fit(np.reshape(X_train, (-1, X.shape[-1])))        
                    pca_model = PCA(svd_solver='full').fit(np.reshape(X_train, (-1, X.shape[-1])))
                elif np.ndim(X_train) == 2:
                    sfa_model = SFA(1).fit(X_train)
                    pca_model = PCA(svd_solver='full').fit(X_train)
                else:
                    raise ValueError('Something weird happened with dimension of X')

        # Organize results in a dictionary structure
        results_dict = {}
        results_dict['dim'] = dim
        results_dict['fold_idx'] = fold_idx
        results_dict['train_idxs'] = train_idxs
        results_dict['test_idxs'] = test_idxs
        results_dict['DCA'] = {}
        results_dict['DCA']['coef'] = dcacoef
        results_dict['DCA']['PI'] = pi
        results_dict['KCA'] = {}
        results_dict['KCA']['coef'] = kcacoef
        results_dict['KCA']['mmse'] = mmse

        results_dict['fit_all'] = fit_all

        if fit_all:
            # results_dict['OLS'] = {}
            # results_dict['OLS']['coef'] = varmodel.coef_
            # results_dict['OLS']['A'] = A
            # results_dict['OLS']['W'] = W

            results_dict['PCA'] = {}
            results_dict['PCA']['coef'] = pca_model.components_.T
            results_dict['PCA']['variance'] = pca_model.explained_variance_
            results_dict['SFA'] = sfa_model.coef_

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

        dim_val, fold_idx, dimreduc_method, \
        decoder, dimreduc_results, X, Y, decoder_args, results_folder = task_tuple

        print('Working on %d, %d, %s' % (dim_val, fold_idx, dimreduc_method))

        # Find the index in dimreduc_results that matches the fold_idx and dim_vals
        # that have been assigned to us
        dim_fold_tuples = [(result['dim'], result['fold_idx']) for result in dimreduc_results]
        # Segment the analysis based on the dimreduc_method
        if dimreduc_method == 'PCA':
            dimreduc_idx = dim_fold_tuples.index((1, fold_idx))
            # Grab the coefficients
            coef_ = dimreduc_results[dimreduc_idx]['PCA']['coef'][:, 0:dim_val]

        elif dimreduc_method == 'DCA':
            dimreduc_idx = dim_fold_tuples.index((dim_val, fold_idx))
            coef_ = dimreduc_results[dimreduc_idx]['DCA']['coef']

        elif dimreduc_method == 'KCA':
            dimreduc_idx = dim_fold_tuples.index((dim_val, fold_idx))
            coef_ = dimreduc_results[dimreduc_idx]['KCA']['coef']

        elif 'OLS' in dimreduc_method:
            dimreduc_idx = dim_fold_tuples.index((1, fold_idx))
            W = dimreduc_results[dimreduc_idx]['OLS']['W']
            eigvals, U = np.linalg.eig(W)
            eigorder = np.argsort(np.abs(eigvals))[::-1]
            U = U[:, eigorder]
            coef_ = U[:, 0:dim_val]

        # Project the (train and test) data onto the subspace and train and score the requested decoder
        train_idxs = dimreduc_results[dimreduc_idx]['train_idxs']
        test_idxs = dimreduc_results[dimreduc_idx]['test_idxs']

        Ytrain = Y[train_idxs, ...]
        Ytest = Y[test_idxs, ...]

        # Need to lag the data before applying dimensionality reduction
        if dimreduc_method == 'OLS':
            T = coef_.shape[0] // X.shape[-1]
            Xtrain = X[train_idxs, ...]
            Xtest = X[test_idxs, ...]

            Xtrain, Ytrain = form_lag_matrix(Xtrain, T, y=Ytrain)
            Xtest, Ytest = form_lag_matrix(Xtest, T, y=Ytest)

            # Convert to array and force real valued entries
            Xtrain = np.array(Xtrain)
            Xtest = np.array(Xtest)
            Ytrain = np.array(Ytrain)
            Ytest = np.array(Ytest)

            Xtrain = (Xtrain @ coef_).astype(np.float)
            Xtest = (Xtest @ coef_).astype(np.float)

        else:
            Xtrain = X[train_idxs, ...] @ coef_
            Xtest = X[test_idxs, ...] @ coef_

        r2_pos, r2_vel, r2_acc, decoder_obj = DECODER_DICT[decoder](Xtest, Xtrain, Ytest, Ytrain, **decoder_args)

        # Compile results into a dictionary
        results_dict = {}
        results_dict['dim'] = dim_val
        results_dict['fold_idx'] = fold_idx
        results_dict['dr_method'] = dimreduc_method
        results_dict['decoder'] = decoder
        results_dict['decoder_args'] = decoder_args
        results_dict['decoder_obj'] = decoder 
        results_dict['r2'] = [r2_pos, r2_vel, r2_acc]

        # Save to file
        with open('%s/dim_%d_fold_%d_%s_%s.dat' % \
                (results_folder, dim_val, fold_idx, dimreduc_method, decoder), 'wb') as f:
            f.write(pickle.dumps(results_dict))


def dimreduc_(X, dim_vals, T, 
              n_folds, comm, split_ranks, results_file,
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

    if ((comm is not None) and comm.rank == 0) or (comm is None):
        # Do cv_splits here
        cv = KFold(n_folds, shuffle=False)

        train_test_idxs = list(cv.split(X))
        data_tasks = [(idx,) + train_test_split for idx, train_test_split
                    in enumerate(train_test_idxs)]    

        # Fit OLS/SFA/PCA for d = 1 (evens out times)
        min_dim_val = np.min(dim_vals)
        dim_vals = [(dim, True if dim == min_dim_val else False) for dim in dim_vals]
        tasks = itertools.product(data_tasks, dim_vals)

        # Send the data itself as well
        tasks = [task + (T, results_folder) for task in tasks]
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

def decoding_(dimreduc_file, data_path,
              dimreduc_methods, decoders,
              comm, split_ranks, 
              decoder_args, results_file, 
              dim_vals=None, fold_idxs=None):
    
    # Look for an arg file in the same folder as the dimreduc_file
    dimreduc_path = '/'.join(dimreduc_file.split('/')[:-1])
    dimreduc_fileno = int(dimreduc_file.split('_')[-1].split('.dat')[0])
    argfile_path = '%s/arg%d.dat' % (dimreduc_path, dimreduc_fileno)

    with open(argfile_path, 'rb') as f:
        args = pickle.load(f) 

    data_file_name = args['data_file'].split('/')[-1]
    data_file_path = '%s/%s' % (data_path, data_file_name)

    # Don't do this one
    if data_file_name == 'trialtype0.dat':
        return

    # Load data and dimreduc file
    if comm is None:
        dat = LOADER_DICT[args['loader']](data_file_path, **args['loader_args'])
        with open(dimreduc_file, 'rb') as f:
            dimreduc_results = pickle.load(f)
    else:
        if comm.rank == 0:
            dat = LOADER_DICT[args['loader']](data_file_path, **args['loader_args'])
            with open(dimreduc_file, 'rb') as f:
                dimreduc_results = pickle.load(f)
        else:
            dat = None
            dimreduc_results = None

        dat = comm.bcast(dat)
        dimreduc_results = comm.bcast(dimreduc_results)

    X = np.squeeze(dat['spike_rates'])
    Y = dat['behavior']
    
    # Pass in for manual override for use in cleanup
    if dim_vals is None:
        dim_vals = args['task_args']['dim_vals']
    if fold_idxs is None:
        n_folds = args['task_args']['n_folds']
        fold_idxs = np.arange(n_folds)

    results_folder = results_file.split('.')[0]   
    # Assemble task arguments
    task_tuples = itertools.product(dim_vals, fold_idxs, 
                                    dimreduc_methods, decoders)

    task_tuples = [tup + (dimreduc_results, X, Y, decoder_args, results_folder) 
                    for tup in task_tuples]

    if comm is not None:
        # Create folder for processes to write in
        if comm.rank == 0:
            if not os.path.exists(results_folder):
                os.makedirs(results_folder)

        # pool = MPIPool(comm, subgroups=split_ranks)
        # Try simple, even split
        
        task_tuples = np.array_split(task_tuples, comm.size)[comm.rank]
        print(len(task_tuples))
        for tup_ in task_tuples:
            t0 = time.time()
            #print(len(tup_))
            decoding_main(tup_) 
            print(t0 - time.time())      
        
    else: 
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)

        pool = SerialPool()

        pool.map(decoding_main, task_tuples)
        print('Hello 1')
        pool.close()
        print('Hello 2')

def main(cmd_args, args):
    total_start = time.time() 
    # MPI split`
    if not cmd_args.local:
        comm = MPI.COMM_WORLD
        ncomms = cmd_args.ncomms
    else:
        comm = None
        ncomms = None

    if cmd_args.analysis_type == 'var':
        if cmd_args.local:
            dat = LOADER_DICT[args['loader']](args['data_file'], **args['loader_args'])
        else:
            if comm.rank == 0:
                dat = LOADER_DICT[args['loader']](args['data_file'], **args['loader_args'])
            else:
                dat = None

            dat = comm.bcast(dat)

        varmodel = VAR(estimator=args['task_args']['estimator'], 
                        penalty=args['task_args']['penalty'], 
                        order=args['task_args']['order'],
                        fit_type=args['task_args']['fit_type'],
                        self_regress = args['task_args']['self_regress'],
                        estimation_score=args['task_args']['estimation_score'],
                        estimation_frac=1.,
                        n_boots_est=1,
                        comm=comm, ncomms=ncomms)

        savepath = args['results_file'].split('.dat')[0]

        # Which indices should we fit to?
        if args['task_args']['idxs'] == 'all':
            X = np.squeeze(dat['spike_rates'])
        else:
            X = np.squeeze(dat['spike_rates'])
            cv = KFold(args['task_args']['n_folds'], shuffle=False)
            train_test_idxs = list(cv.split(X))
            train_idxs = train_test_idxs[args['task_args']['idxs']][0]
            X = X[train_idxs, ...]
        
        varmodel.fit(X, distributed_save=True, savepath=savepath, resume=cmd_args.resume)

#        if comm is not None:
#            if comm.rank == 0:
#                with open(cmd_args.results_file, 'wb') as f:
#                    f.write(pickle.dumps(args))
#                    f.write(pickle.dumps(varmodel.coef_))
#                    f.write(pickle.dumps(varmodel.scores_))
#        else:
#            with open(cmd_args.results_file, 'wb') as f:
#                f.write(pickle.dumps(args))
#                f.write(pickle.dumps(varmodel.coef_))
#                f.write(pickle.dumps(varmodel.scores_))

    elif cmd_args.analysis_type == 'dimreduc':

        if cmd_args.local:
            dat = LOADER_DICT[args['loader']](args['data_file'], **args['loader_args'])
        else:
            dat = LOADER_DICT[args['loader']](comm, args['data_file'], **args['loader_args'])
            spike_rates = np.ascontiguousarray(np.squeeze(dat['spike_rates']), dtype=float)
            spike_rates = Bcast_from_root(spike_rates, comm)
        
        # # Make global variable
        globals()['X'] =  spike_rates
        
        split_ranks = comm_split(comm, ncomms)
        dimreduc_(spike_rates, 
                  dim_vals = args['task_args']['dim_vals'],
                  T = args['task_args']['T'],
                  n_folds = args['task_args']['n_folds'], 
                  comm=comm, split_ranks=split_ranks,
                  results_file = args['results_file'],
                  resume=cmd_args.resume)

    elif cmd_args.analysis_type == 'decoding':

        split_ranks = comm_split(comm, ncomms)
        results = decoding_per_dim(args['task_args']['dimreduc_file'], 
                                   args['data_path'],
                                   args['task_args']['dimreduc_methods'],
                                   args['task_args']['decoders'],
                                   comm, split_ranks, 
                                   args['task_args']['decoder_args'],
                                   args['results_file'])

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
    parser.add_argument('--local', dest='local', action='store_true')
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