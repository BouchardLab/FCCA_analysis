import numpy as np 
import glob
import sys 
import os
import importlib
import itertools
import pickle
import pdb

def submit_sbatch(path, sbatch_string):

    sbatch_files = glob.glob('%s/%s' % (path, sbatch_string))
    # Check
    pdb.set_trace()
    for file_ in sbatch_files:
        os.system('sbatch %s' % file_)

def gen_sbatch(arg_array, sbatch_params, local=False, 
               shifter=True, resume=False):

    # We are going to submit a *single* sbatch script that requests multiple nodes
    # Put common stuff up top
    if 'sbname' not in sbatch_params.keys():
        if resume:
            sbname = 'sbatch_resume.sh'
        else:
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
                cmd_args0 = ' '.join([' --%s=%s ' % (key, value) for key, value in sbatch_params['cmd_args0'].items()])
                cmd_args1 = ' '.join([' --%s ' % key for key, value in sbatch_params['cmd_args1'].items() if value]) 
                # sb.write('sbcast -f --compress %s/%s /tmp/%s\n' % (script_dir, script, script))
                sb.write('mpirun -n 8 python3 -u %s %s' % (sbatch_params['script_path'], arg_file) + cmd_args0 + cmd_args1 + '\n')
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
                cmd_args0 = '--ncomms=%d ' % sbatch_params['ncomms'][i]
                cmd_args0 += ' '.join([' --%s=%s ' % (key, value) for key, value in sbatch_params['cmd_args0'].items()])
                cmd_args1 = ' '.join([' --%s ' % key for key, value in sbatch_params['cmd_args1'].items() if value]) 
 
                if shifter:
                    sb.write('srun -N %d -n %d -c %d ' % (sbatch_params['n_nodes'][i], 
                                                          sbatch_params['n_nodes'][i] * sbatch_params['tpn'][i], 
                                                          sbatch_params['cpt']) + \
                             'shifter --entrypoint python3 -u %s %s ' % (sbatch_params['script_path'],
                                                                         arg_file) + cmd_args0 + cmd_args1)


                else:
                     sb.write('srun -N %d -n %d -c %d ' % (sbatch_params['n_nodes'][i], 
                                                           sbatch_params['n_nodes'][i] * sbatch_params['tpn'][i], 
                                                           sbatch_params['cpt']) + \
                              'python3 -u %s %s ' % (sbatch_params['script_path'],
                                                        arg_file) + cmd_args0 + cmd_args1)

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

# Possible kwargs in the case default job size estimation is to be overridden:
# numtasks: number of total MPI processes desired
# cpu_per_task: number of cpus to allocate per MPI process
# n_nodes: numer of nodes to request
def init_batch(submit_file, jobdir, job_time='04:00:00', qos='regular', local=False, shifter=False, 
               sequential=False, resume=True, split_sbatch=False, **kwargs):

    if not os.path.exists(jobdir):
        os.makedirs(jobdir)

    jobname = jobdir.split('/')[-1]

    path = '/'.join(submit_file.split('/')[:-1])
    name = submit_file.split('/')[-1]
    
    name = os.path.splitext(name)[0]
    sys.path.append(path)
    args = importlib.import_module(name)
    analysis_type = args.analysis_type

    # Copy submit file to jobdir
    os.system('cp %s %s/' % (submit_file, jobdir))
    script_path = args.script_path

    loader_args = args.loader_args
    task_args = args.task_args

    if hasattr(args, 'desc'):
        desc = args.desc
    else:
        desc = 'No description available.'

    arg_array = []
    for i, param_comb in enumerate(itertools.product(args.data_files, loader_args, task_args)):
        arg_array.append({'data_file':param_comb[0], 'loader':args.loader, 'loader_args':param_comb[1],
                          'task_args':param_comb[2], 'data_path':args.data_path, 
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


    # Generate a set of argfiles that correspond to each unique param_comb. these are potentially references
    # by downstream analysis, but NOT the arg files that are loaded in when we run this job. This allows for
    # flexibility with arg_splits
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
            sbatch_params = {'qos':qos, 'jobname':jobname, 'tpn': tpn,
                             'cpt': cpt, 'script_path':script_path, 'jobdir': jobdir, 'jobnos': jobnos[i],
                             'job_time': job_time, 'n_nodes': n_nodes[i], 'total_nodes': sum(n_nodes[i]),
                             'sequential': sequential, 'ncomms': ncomms, 'sbname':'sbatch_%d.sh' % i,
                             'cmd_args0': {'analysis_type':analysis_type},
                             'cmd_args1': {'resume':resume}}

            gen_sbatch(arg_array_, sbatch_params, local, shifter, resume)

    else:
        sbatch_params = {'qos':qos, 'jobname': jobname, 'tpn': tpn, 
                         'cpt': cpt, 'script_path':script_path, 'jobnos': np.arange(len(arg_array)),
                         'jobdir': jobdir, 'job_time': job_time, 'n_nodes': n_nodes,
                         'total_nodes': total_nodes, 'sequential' : sequential,  
                         'ncomms':ncomms,
                         'cmd_args0': {'analysis_type':analysis_type},
                         'cmd_args1': {'resume':resume}}

        gen_sbatch(arg_array, sbatch_params, local, shifter, resume)
