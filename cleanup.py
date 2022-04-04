import os
import glob
import numpy as np 
import pickle
import marshal
import itertools
import pdb
from tqdm import tqdm
import importlib
from sklearn.model_selection import KFold

#from batch_analysis import comm_split, dca_main, decoding_per_dim
from loaders import load_sabes, load_shenoy, load_peanut
# from schwimmbad import MPIPool, SerialPool

LOADER_DICT = {'sabes': load_sabes, 'shenoy': load_shenoy, 'peanut': load_peanut}

# Utility to go through all the arg files in a directory and change the data path
def change_data_paths(root_dir, data_path=None, results_path=None):

    # Grab arg files
    argfiles = glob.glob('%s/arg*' % root_dir)

    # Filter out arg_
    # for argfile in argfiles_:
    #     if not 'arg_' in argfile:
    #         argfiles.append(argfile)
    for j, argfile in enumerate(argfiles):
        with open(argfile, 'rb') as f:
            args = pickle.load(f)

        if data_path is not None:
            # replace the root directory
            fname = args['data_file'].split('/')[-1]
            args['data_file'] = '%s/%s' % (data_path, fname)
            args['data_path'] = data_path

        if results_path is not None:

            fname = args['results_file'].split('/')[-1]
            args['results_file'] = '%s/%s' % (results_path, fname)

        with open(argfile, 'wb') as f:
            f.write(pickle.dumps(args))

def check_root(comm):
    if comm is None:
        return True
    else:
        if comm.rank == 0:
            return True
        else:
            return False

def cleanup_var(root_dir, job_name, submit_file):

    to_do = {}

    # Grab arg files
    argfiles_ = glob.glob('%s/arg*' % root_dir)

    # Filter out arg_
    argfiles = []
    for argfile in argfiles_:
        if not 'arg_' in argfile:
            argfiles.append(argfile)

    # Load the submit file to get the data files.
    name = submit_file.split('/')[-1]    
    name = os.path.splitext(name)[0]
    sys.path.append(path)
    submit_args = importlib.import_module(name)

    # Load each data file and for each set of loader args, get the dof.
    data_files = submit_args.data_files
    n_dof = np.zeros((len(data_files), len(submit_args.loader_args)))
    for i, data_file in enumerate(data_files):
        for j, loader_arg in enumerate(submit_args.loader_args):
            dat = LOADER_DICT[submit_args.loader](data_file, loader_arg])
            n_dof[i, j] = dat['spike_rates'].shape[-1]

    # For each arg file, get the number and then find the
    # directory
    completed_files = []

    for j, argfile in enumerate(argfiles):
        print('%d out of %d' % (j + 1, len(argfiles)))
        jobno = int(argfile.split('/')[-1].split('arg')[-1].split('.')[0])

        jobdir = '%s/%s_%d' % (root_dir, job_name, jobno)

        results_file = '%s/%s_%d.dat' % (root_dir, job_name, jobno)

        # First check if the results_file already exists. If so, continue
        if os.path.exists(results_file):
            continue

        # Load args and check if the job was completed
        with open(argfile, 'rb') as f:
            args = pickle.load(f)

        # Expected output files
        # First lookup dof given the data file and loader_args corresponding to this argfile
        idx1 = data_files.index(args['data_file'])
        idx2 = next((index for (index, d) in enumerate(submit_args.loader_args) if d == args['loader_args']))

        n_dof = n_dof[idx1, idx2]

        # Scale by the var order
        n_dof *= args['task_args']['order']

        expected_files = ['%d.dat' % i for i in np.arange(n_dof)]
        data_files = glob.glob('%s/*.dat' % jobdir)
        found_files = [file.split('/')[-1] for file in data_files]

        to_do[jobno] = []
        for expected_file in expected_files:
            if expected_file not in found_files:
                to_do[jobno].append(expected_file)

        # Concatenate results
        if len(to_do[jobno]) == 0:

            coefs = np.zeros((n_dof, n_dof, args['task_args']['order']))
            scores_and_supports = {}

            for data_file in data_files:
                rowno = int(data_file.split('/')[-1].split('.dat')[0])
                with open(data_file, 'rb') as f:
                    coef_ = pickle.load(f)
                    #ss = pickle.load(f) 

                if args['task_args']['self_regress']:
                    coef__ = np.reshape(coef_, (args['task_args']['order'], n_dof)).T
                else:
                    coef__ = np.zeros((args['task_args']['order'], n_dof))
                    coef__[:, np.arange(n_dof) != rowno] = np.reshape(coef_, (args['task_args']['order'], n_dof - 1))
                    coef__ = coef__.T

                coefs[rowno, ...] = np.fliplr(coef__) 

                #scores_and_supports[rowno] = ss

            # This mirrors the sequence in pyuoi var
            coefs = np.transpose(coefs, axes=(2, 0, 1))
            
            with open(results_file, 'wb') as f:
                f.write(pickle.dumps(args))
                f.write(pickle.dumps(coefs))
                #f.write(pickle.dumps(scores_and_supports))

            completed_files.append(results_file)

    return to_do, completed_files

# Take subfolders of DCA results and properly
# combine into single files
def cleanup_dca(root_dir, job_name):

    to_do = {}

    # Grab arg files
    argfiles_ = glob.glob('%s/arg*' % root_dir)

    # Filter out arg_
    argfiles = []
    for argfile in argfiles_:
        if not 'arg_' in argfile:
            argfiles.append(argfile)

    # For each arg file, get the number and then find the
    # directory
    for argfile in tqdm(argfiles):

        jobno = int(argfile.split('/')[-1].split('arg')[-1].split('.')[0])

        jobdir = '%s/%s_%d' % (root_dir, job_name, jobno)

        # Load args and check if the job was completed
        with open(argfile, 'rb') as f:
            args = pickle.load(f)

        # Get dim_vals and n_folds
        dim_vals = args['task_args']['dim_vals']
        n_folds = args['task_args']['n_folds']

        # Expected output files
        outputs = itertools.product(dim_vals, np.arange(n_folds))
        expected_files = ['dim_%d_fold_%d.dat' % tup for tup in outputs]
        data_files = glob.glob('%s/*.dat' % jobdir)
        found_files = [file.split('/')[-1] for file in data_files]

        to_do[jobno] = []
        for expected_file in expected_files:
            if expected_file not in found_files:
                to_do[jobno].append(expected_file)
        if len(to_do[jobno]) != 0:
            print('%d:%s' % (jobno, args['task_args']['dimreduc_method']))


        if len(to_do[jobno]) == 0:
            results_file = '%s/%s_%d.dat' % (root_dir, job_name, jobno)
            results_dict_list = []
            for data_file in data_files:
                with open(data_file, 'rb') as f:
                    results_dict = pickle.load(f)

                for key, value in args['loader_args'].items():
                    results_dict[key] = value
                for key, value in args['task_args'].items():
                    results_dict[key] = value

                results_dict['data_file'] = args['data_file'].split('/')[-1] 
                results_dict_list.append(results_dict)

            with open(results_file, 'wb') as f:
                pickle.dump(results_dict_list, f, protocol=-1)

    return to_do

def cleanup_decoding(root_dir, job_name, complete=True, 
                     dimreduc_path=None, data_path=None):

    to_do = {}

    # Grab arg files
    argfiles_ = glob.glob('%s/arg*' % root_dir)

    # Filter out arg_ - these are used to flexibly distribute tasks on NERSC
    argfiles = []
    for argfile in argfiles_:
        if not 'arg_' in argfile:
            argfiles.append(argfile)

    # For each arg file, get the number and then find the
    # directory

    # Also need to backtrack and include the original data file in the final results

    for argfile in argfiles:

        jobno = int(argfile.split('/')[-1].split('arg')[-1].split('.')[0])

        jobdir = '%s/%s_%d' % (root_dir, job_name, jobno)

        # Load args and check if the job was completed
        with open(argfile, 'rb') as f:
            args = pickle.load(f)

        dimreduc_file = args['task_args']['dimreduc_file']
        if dimreduc_path is None:
            dimreduc_path = '/'.join(dimreduc_file.split('/')[:-1])
        dimreduc_fileno = int(dimreduc_file.split('_')[-1].split('.dat')[0])
        dr_argfile_path = '%s/arg%d.dat' % (dimreduc_path, dimreduc_fileno)

        dimreduc_file = '%s/%s' % (dimreduc_path, dimreduc_file.split('/')[-1])

        # Load the dr_argfile and get the name of the original data file
        with open(dr_argfile_path, 'rb') as f:
            dr_args = pickle.load(f)

        original_data_file = dr_args['data_file'].split('/')[-1]


        # Get dim_vals and n_folds
        dim_vals = dr_args['task_args']['dim_vals']
        n_folds = dr_args['task_args']['n_folds']
        dimreduc_methods = args['task_args']['dimreduc_methods']
        decoders = args['task_args']['decoders']

        # Expected output files
        outputs = itertools.product(dim_vals, np.arange(n_folds), dimreduc_methods, decoders)
        expected_files = ['dim_%d_fold_%d_%s_%s.dat' % tup for tup in outputs]
        data_files = glob.glob('%s/*.dat' % jobdir)
        found_files = [file.split('/')[-1] for file in data_files]

        to_do[jobno] = []
        for expected_file in expected_files:
            if expected_file not in found_files:
                to_do[jobno].append(expected_file)

        if len(to_do[jobno]) == 0:
            results_file = '%s/%s_%d.dat' % (root_dir, job_name, jobno)
            results_dict_list = []
            for data_file in data_files:
                with open(data_file, 'rb') as f:
                    results_dict = pickle.load(f)

                results_dict['data_file'] = original_data_file
                # Add preprocessing parameters
                for key, value in dr_args['loader_args'].items():
                    results_dict[key] = value
                for key, value in args['task_args']['decoder_args'].items():
                    results_dict[key] = value
 
                results_dict_list.append(results_dict)

            with open(results_file, 'wb') as f:
                pickle.dump(results_dict_list, f, protocol=-1)

        elif complete:

            for task in to_do[jobno]:
                dim = int(task.split('_')[1])
                fold_idx = int(task.split('_')[3])
                dimreduc_method = [task.split('_')[4]]
                decoder = [task.split('_')[-1].split('.dat')[0]]

                if data_path is None:
                    data_path = args['data_path']


                # Finish off any stragglers
                decoding_per_dim(dimreduc_file, data_path, dimreduc_method,
                                 decoder, None, None, args['task_args']['decoder_args'],
                                 '%s/%s_%d.dat' % (root_dir, job_name, jobno), dim_vals=[dim],
                                 fold_idxs=[fold_idx])


    return to_do

###### CODE for "resuming" if significant amount of tasks not completed ##### 
'''
            # Assemble a task tuple compatible with batch_analysis_sabes.dca_main
            argfile = '%s/arg%d.dat' % (root_dir, key)
            with open(argfile, 'rb') as f:
                args = pickle.load(f)

            # Replace the data file path as we are likely on a different machine
            data_file_name = args['data_file'].split('/')[-1]
            data_file_path = '%s/%s' % (data_path, data_file_name)
            
            # Load the data
            if check_root(comm):
                dat = LOADER_DICT[args['loader']](data_file_path, **args['loader_args'])
            else:
                dat = None

            if comm is not None:
                dat = comm.bcast(dat)

            X = np.squeeze(dat['spike_rates'])
            T = args['task_args']['T']
            
            # Outstanding dim vals and fold_idxs
            dim_fold_combs = [(int(s.split('_')[1]), int(s.split('_')[-1].split('.dat')[0])) for s in value]

            # Do cv_splits here

            cv = KFold(n_folds, shuffle=False)
            pdb.set_trace()
            train_test_idxs = list(cv.split(X))

            data_tasks = [(fold_idx,) + train_test_idxs[fold_idx] for (_, fold_idx) in dim_fold_combs]

            # Fit OLS/SFA/PCA for d = 1 (evens out times)
            min_dim_val = 1
            dim_vals = [(dim, True if dim == min_dim_val else False) for (dim, _) in dim_fold_combs]

            pdb.set_trace()
            # # Send the data itself as well
            # tasks = [task + (X, T, results_folder) for task in tasks]
'''
