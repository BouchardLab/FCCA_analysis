import os
import gc
import argparse
import time
import pickle
import glob
import itertools
import numpy as np
import scipy
from copy import deepcopy
from mpi4py import MPI
from mpi_utils.ndarray import Bcast_from_root
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from statsmodels.tsa import stattools
#from pyuoi.linear_model.var import VAR
from dca.cov_util import form_lag_matrix

# from neurosim.models.ssr import StateSpaceRealization as SSR

from schwimmbad import MPIPool, SerialPool
import pdb
import glob

def load_preprocessed(path, **kwargs):
    with open(path, 'rb') as f:
        datM1 = pickle.load(f)
        datS1 = pickle.load(f)
        lparam = pickle.load(f)
    return datM1, datS1, lparam

# Check which tasks have already been completed and prune from the task list
def prune_tasks(tasks, progress_file):
    if not os.path.exists(progress_file):
        return tasks
    else:
        pf = open(progress_file, 'rb')

    completed_combos = []
    while True:
        try:
            completed_combos.append(pickle.load(pf))
        except EOFError:
            break

    to_do = []
    for task in tasks:
        if task not in completed_combos:
            to_do.append(task)

    pf.close()

    return to_do

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

def load_data(data_file, comm):

    if comm is None:
        datM1, datS1, lparam = load_preprocessed(data_file)
        XM1 = np.ascontiguousarray(np.squeeze(datM1['spike_rates']), dtype=float)
        XS1 = np.ascontiguousarray(np.squeeze(datS1['spike_rates']), dtype=float)
    else:
        if comm.rank == 0:
            datM1, datS1, lparam = load_preprocessed(data_file)
            XM1 = np.ascontiguousarray(np.squeeze(datM1['spike_rates']), dtype=float)
            XS1 = np.ascontiguousarray(np.squeeze(datS1['spike_rates']), dtype=float)
        else:
            XM1 = None
            XS1 = None

        XM1 = Bcast_from_root(XM1, comm)
        XS1 = Bcast_from_root(XS1, comm)

    # # Make global variable - saves memory when using Schwimmbad as the data can be accessed by workers without
    # being sent again (which duplicates it)
    globals()['XM1'] =  XM1
    globals()['XS1'] = XS1


class PoolWorker():
    def __init__(self, results_file, progress_file, task_type):
        self.results_file = results_file
        self.progress_file = progress_file
        self.task_type = task_type

    def ccf(self, task):

        if isinstance(task[-1], MPI.Intracomm):
            task, comm = task

        S1idx, M1idx = task
        xs1 = globals()['XS1'][:, S1idx]
        xm1 = globals()['XM1'][:, M1idx]
        crosscorrfunc = stattools.ccf(xs1, xm1)

        # Only keep the first segment
        crosscorrfunc = crosscorrfunc[0:int(10 * np.log10(xs1.size))]

        r = {}
        r['S1idx'] = S1idx
        r['M1idx'] = M1idx
        r['ccf'] = crosscorrfunc

        return (r, task)

    def callback(self, result):
        rdict, task = result

        # (1) Append the result to the open results file
        with open(self.results_file, 'ab') as f:
            f.write(pickle.dumps(result))

        # (2) Save the task parameters to the progress file
        with open(self.progress_file, 'ab') as f:
            f.write(pickle.dumps(task))

    def cca(self, task):
        if isinstance(task[-1], MPI.Intracomm):
            task, comm = task

        lag, w = task
        X = globals()['XS1']
        Y = globals()['XM1']

        r = {}
        r['lag'] = lag
        r['win'] = w

        # Apply window and lag relative to each other
        if lag != 0:
            x = X[:-lag, :]
            y = Y[lag:, :]
        else:
            x = X
            y = Y

        if w > 1:
            x = form_lag_matrix(x, w)
            y = form_lag_matrix(y, w)

        ccamodel = CCA(n_components=6)
        ccamodel.fit(x, y)
        r['ccamodel'] = ccamodel

        return (r, task)

def main(cmd_args, args):
    total_start = time.time() 

    # MPI split
    if not cmd_args.serial:           
        comm = MPI.COMM_WORLD
        ncomms = cmd_args.ncomms
    else:
        comm = None
        ncomms = None

    load_data(args['data_file'], comm)

    results_file = args['results_file']

    if args['task_args']['task_type'] == 'ccf':

        # create or append toa results file and progress file
        results_file = results_file.split('/')
        results_file[-1] = 'ccf_' + results_file[-1]
        progress_file = deepcopy(results_file)
        progress_file[-1] = 'progress_' + progress_file[-1]

        results_file = '/'.join(results_file)
        progress_file = '/'.join(progress_file)

        worker = PoolWorker(results_file=results_file, progress_file=progress_file, task_type='ccf')

        if comm is not None:
            if comm.rank == 0:            
                nM1 = globals()['XM1'].shape[-1]
                nS1 = globals()['XS1'].shape[-1]

                tasks = list(itertools.product(np.arange(nS1), np.arange(nM1)))
                tasks = prune_tasks(tasks, progress_file)
            else:
                tasks = None


            tasks = comm.bcast(tasks)
            print('%d Tasks Remaining' % len(tasks))

            pool = MPIPool(comm)
        else:
            nM1 = globals()['XM1'].shape[-1]
            nS1 = globals()['XS1'].shape[-1]

            tasks = list(itertools.product(np.arange(nS1), np.arange(nM1)))
            tasks = prune_tasks(tasks, progress_file)
            pool = SerialPool()

        if len(tasks) > 0:
            pool.map(worker.ccf, tasks, callback=worker.callback)

        pool.close()


    elif args['task_args']['task_type'] == 'cca':
        # create or append toa results file and progress file
        results_file = results_file.split('/')
        results_file[-1] = 'cca_' + results_file[-1]
        progress_file = deepcopy(results_file)
        progress_file[-1] = 'progress_' + progress_file[-1]

        results_file = '/'.join(results_file)
        progress_file = '/'.join(progress_file)

        worker = PoolWorker(results_file=results_file, progress_file=progress_file, task_type='cca')

        if comm is not None:
            if comm.rank == 0:            
                tasks = list(itertools.product(args['task_args']['task_args']['lags'], args['task_args']['task_args']['windows']))
                tasks = prune_tasks(tasks, progress_file)
            else:
                tasks = None


            tasks = comm.bcast(tasks)
            print('%d Tasks Remaining' % len(tasks))

            pool = MPIPool(comm)
        else:
            nM1 = globals()['XM1'].shape[-1]
            nS1 = globals()['XS1'].shape[-1]

            tasks = list(itertools.product(args['task_args']['task_args']['lags'], args['task_args']['task_args']['windows']))
            tasks = prune_tasks(tasks, progress_file)
            pool = SerialPool()

        if len(tasks) > 0:
            pool.map(worker.cca, tasks, callback=worker.callback)

        pool.close()


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
