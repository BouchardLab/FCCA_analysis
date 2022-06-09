import numpy as np
import scipy
import pdb
import glob
import os
import sys
import pickle
import itertools
import importlib
from mpi4py import MPI
from schwimmbad import MPIPool, SerialPool

from dca_research.lqg import LQGComponentsAnalysis as LQGCA
from dca.dca import DynamicalComponentsAnalysis as DCA

dr_dict = {'FCCA': LQGCA, 'DCA':DCA}

def gen_sbatch(arg_array, sbatch_params, local=False, shifter=True):

    # We are going to submit a *single* sbatch script that requests multiple nodes
    # Put common stuff up top
    if 'sbname' not in sbatch_params.keys():
        sbname = 'sbatch.sh'
    else:
        sbname = sbatch_params['sbname']

    jobdir = sbatch_params['jobdir']
    sbname = '%s/%s' % (jobdir, sbname)
    jobname = sbatch_params['jobname']
    qos = sbatch_params['qos'] 

    with open(sbname, 'w') as sb:
        if local:
            for i, arg in enumerate(arg_array):
                arg_file = '%s/arg%d.dat' % (jobdir, sbatch_params['jobnos'][i])
                # sb.write('sbcast -f --compress %s/%s /tmp/%s\n' % (script_dir, script, script))
                sb.write('mpirun -n 8 python3 -u %s %s' % (sbatch_params['script_path'], arg_file) + '\n')
        else:
            sb.write('#!/bin/bash\n')
            sb.write('#SBATCH --qos=%s\n' % qos)
            sb.write('#SBATCH --constraint=knl\n')
            if shifter:
                sb.write('#SBATCH --image=docker:akumar25/nersc_conda_base:latest\n')
            sb.write('#SBATCH -N %d\n' % sbatch_params['total_nodes'])
            sb.write('#SBATCH -t %s\n' % sbatch_params['job_time'])
            sb.write('#SBATCH --job-name=%s%d\n' % (jobname, sbatch_params['jobnos'][0]))
            sb.write('#SBATCH --out=%s/%s%d.o\n' % (jobdir, jobname, sbatch_params['jobnos'][0]))
            sb.write('#SBATCH --error=%s/%s%d.e\n' % (jobdir, jobname, sbatch_params['jobnos'][0]))
            sb.write('#SBATCH --mail-user=ankit_kumar@berkeley.edu\n')
            sb.write('#SBATCH --mail-type=FAIL\n')

            sb.write('source ~/anaconda3/bin/activate\n')
            sb.write('source activate dyn\n')

            # Critical to prevent threads competing for resources
            sb.write('export OMP_NUM_THREADS=1\n')
            sb.write('export KMP_AFFINITY=disabled\n')

            for i, arg in enumerate(arg_array):

                if sbatch_params['n_nodes'][i] == 0:
                    continue

                arg_file = '%s/arg%d.dat' % (jobdir, sbatch_params['jobnos'][i])
                # sb.write('sbcast -f --compress %s/%s /tmp/%s\n' % (script_dir, script, script))
 
                if shifter:
                    sb.write('srun -N %d -n %d -c %d ' % (sbatch_params['n_nodes'][i], 
                                                          sbatch_params['n_nodes'][i] * sbatch_params['tpn'][i], 
                                                          sbatch_params['cpt']) + \
                             'shifter --entrypoint python3 -u %s %s ' % (sbatch_params['script_path'],
                                                                         arg_file))


                else:
                     sb.write('srun -N %d -n %d -c %d ' % (sbatch_params['n_nodes'][i], 
                                                           sbatch_params['n_nodes'][i] * sbatch_params['tpn'][i], 
                                                           sbatch_params['cpt']) + \
                              'python3 -u %s %s ' % (sbatch_params['script_path'],
                                                        arg_file))

                if sbatch_params['sequential']:
                    sb.write('\n')
                else:
                    sb.write(' &\n')
            if not sbatch_params['sequential']:
                sb.write('wait')

def gen_argfiles(jobdir, arg_array, fname):

    for i, arg_ in enumerate(arg_array):
        with open('%s/%s%d.dat' % (jobdir, fname, i), 'wb') as f:
            f.write(pickle.dumps(arg_))

def init_batch(submit_file, jobdir, job_time='04:00:00', qos='regular', local=False, shifter=False, 
               sequential=False, split_sbatch=False, **kwargs):

    if not os.path.exists(jobdir):
        os.makedirs(jobdir)

    jobname = jobdir.split('/')[-1]

    path = '/'.join(submit_file.split('/')[:-1])
    name = submit_file.split('/')[-1]
    
    name = os.path.splitext(name)[0]
    sys.path.append(path)

    args = importlib.import_module(name)

    # Copy submit file to jobdir
    os.system('cp %s %s/' % (submit_file, jobdir))

    script_path = args.script_path
    dbfile = args.dbfile
    task_args = args.task_args
    nsplits = args.nsplits
    idxs = args.idxs
    idxs = np.array_split(idxs, nsplits)

    if hasattr(args, 'desc'):
        desc = args.desc
    else:
        desc = 'No description available.'

    arg_array = []
    for i, param_comb in enumerate(itertools.product(idxs, task_args)):
        arg_array.append({'idxs':param_comb[0], 'task_args':param_comb[1],  'dbfile':dbfile,
                          'results_file': '%s/%s_%d.dat' % (jobdir, jobname, i)})
    cpt = 4

    if 'n_nodes' in kwargs.keys():
        n_nodes = kwargs['n_nodes']
    else:
        n_nodes = 10


    if 'tpn' in kwargs.keys():
        tpn = kwargs['tpn']
    else:
        tpn = 64

    cpt = 4  
    total_nodes = len(arg_array) * n_nodes

    if 'ncomms' in kwargs.keys():
        ncomms = kwargs['ncomms']
    else:
        ncomms = 1


    # Generate a set of argfiles that correspond to each unique param_comb. 
    pdb.set_trace() 
    gen_argfiles(jobdir, arg_array, fname='arg')

    n_nodes = n_nodes * np.ones(len(arg_array))
    tpn = tpn * np.ones(len(arg_array))
    ncomms = ncomms * np.ones(len(arg_array))
    total_nodes = np.sum(n_nodes)

    # Assemble sbatch params
    if split_sbatch:
        arg_array_split = np.array_split(arg_array, split_sbatch)
        n_nodes = np.array_split(n_nodes, split_sbatch)
        jobnos = np.array_split(np.arange(len(arg_array)), split_sbatch)
        for i, arg_array_ in enumerate(arg_array_split):
            sbatch_params = {'qos':qos, 'jobname':jobname, 'tpn': tpn, 'script_path':script_path,
                             'cpt': cpt, 'jobdir': jobdir, 'jobnos': jobnos[i],
                             'job_time': job_time, 'n_nodes': n_nodes[i], 'total_nodes': sum(n_nodes[i]),
                             'sequential': sequential, 'ncomms': ncomms, 'sbname':'sbatch_%d.sh' % i}

            gen_sbatch(arg_array_, sbatch_params, local, shifter)

    else:
        sbatch_params = {'qos':qos, 'jobname': jobname, 'tpn': tpn, 
                         'cpt': cpt, 'script_path':script_path, 'jobnos': np.arange(len(arg_array)),
                         'jobdir': jobdir, 'job_time': job_time, 'n_nodes': n_nodes,
                         'total_nodes': total_nodes, 'sequential' : sequential,  
                         'ncomms':ncomms}

        gen_sbatch(arg_array, sbatch_params, local, shifter)


    

def prune_tasks(tasks, results_folder):
    completed_files = glob.glob('%s/*.dat' % results_folder)
    dim_and_idxs = []
    for completed_file in completed_files:
        dim = int(completed_file.split('dim_')[1].split('_')[0])
        idx = int(completed_file.split('idx_')[1].split('.dat')[0])
        dim_and_idxs.append((idx, dim))

    to_do = []
    for task in tasks:
        idx, dim = task
        if (idx, dim) not in dim_and_idxs:
            to_do.append(task)

    return to_do

def work(task):
    # Unpack and print
    idx, dim, dimreduc_args, results_folder = task
    print('Rank %d working on (%d, %d)' % (comm.rank, idx, dim))
    A = globals()['A'][idx]
    B = globals()['B'][idx]

    W = scipy.linalg.solve_continuous_lyapunov(A, -B @ B.T)
    print(W)
    dimreduc_model = dr_dict[dimreduc_args['method']](**dimreduc_args['method_args'])
    ccm = np.array([scipy.linalg.expm(A * k) @ W for k in range(2 * dimreduc_args['method_args']['T'] + 1)])
    dimreduc_model.cross_covs = ccm
    
    # Keep track of all scores and coefs
    coefs = []  
    scores = []
    for i in range(dimreduc_args['method_args']['n_init']):
        coef, score = dimreduc_model._fit_projection(d=dim)
        coefs.append(coef)
        scores.append(score)

    # Save
    # Organize results in a dictionary structure
    results_dict = {}
    results_dict['W'] = W
    results_dict['coefs'] = coefs
    results_dict['scores'] = scores
    results_dict['A'] = A
    results_dict['B'] = B
    results_dict['idx'] = globals()['idxs'][idx]

    for k, v in dimreduc_args['method_args'].items():
        results_dict[k] = v

    # Write to file, will later be concatenated by the main process
    file_name = 'dim_%d_idx_%d.dat' % (dim, idx)
    with open('%s/%s' % (results_folder, file_name), 'wb') as f:
        f.write(pickle.dumps(results_dict))
    # Cannot return None or else schwimmbad with hang (lol!)
    return 0    

def main(args, results_file):

    # Split off results_file to create a results_folder
    results_folder = results_file.split('.dat')[0]

    # Assemble tasks, prune them, and then distribute to worker pool
    if comm.rank == 0:
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)

        dims = args['task_args']['dims']
        tasks = itertools.product(np.arange(len(globals()['A'])), dims)
        tasks = prune_tasks(tasks, results_folder)
        print('%d Tasks Remaining' % len(tasks))
    else:
        tasks = None
    
    tasks = comm.bcast(tasks)

    dimreduc_args = args['task_args']['dimreduc_args']
    tasks = [t + (dimreduc_args, results_folder) for t in tasks]

    # VERY IMPORTANT: Once pool is created, the workers wait for instructions, so must proceed directly to map
    pool = MPIPool(comm)

    if len(tasks) > 0:
        pool.map(work, tasks)
    pool.close()

    # Gather and save
    if comm.rank == 0:
        data_files = glob.glob('%s/*.dat' % results_folder)
        results_dict_list = []
        for data_file in data_files:
            with open(data_file, 'rb') as f:
                results_dict = pickle.load(f)
                results_dict_list.append(results_dict)

        with open(results_file, 'wb') as f:
            f.write(pickle.dumps(results_dict_list))


# Use a similar format to batch_analysis. Provide an args file. The args should be
# (1) which portion of the itertools.product(A, B) list should be processed 
# (2) dimvals and task args associated with LQGCA/DCA
# Observables of interest:
# (1) Subspaces, Controllability Gramian
# (2) score(s)
# (3) Also assess score associated with the true LQGCA cost function for both the implied cost
#  matrices as well as more "sensible" matrices (compared to random projections perhaps?)
if __name__ == '__main__':

    argfile = sys.argv[1]

    with open(argfile, 'rb') as f:
        args = pickle.load(f)

    results_file = args['results_file']
    idxs = args['idxs']
    comm = MPI.COMM_WORLD
    # Load A/B matrices, select the relevant slice, and broadcast
    if comm.rank == 0:
        with open(args['dbfile'], 'rb') as f:
            _ = pickle.load(f)
            _ = pickle.load(f)
            AA = pickle.load(f)
            BB = pickle.load(f)
            
        A = [AA[idx[0]] for idx in idxs]
        B = [BB[idx[1]] for idx in idxs]
    else:
        A = None
        B = None

    A = comm.bcast(A)
    B = comm.bcast(B)

    globals()['A'] = A
    globals()['B'] = B
    globals()['idxs'] = idxs

    main(args, results_file)