import glob
import os
import numpy as np
from sklearn.model_selection import KFold

#script_path = '/global/homes/a/akumar25/repos/neural_control/batch_analysis.py'
script_path = '/home/akumar/nse/neural_control/batch_analysis.py'

desc = 'Examining effect of initialization'
#desc = 'Fits of dimreduc methods to loco data'

#data_path = os.environ['SCRATCH'] + '/sabes'
data_path = '/mnt/Secondary/data/sabes'    
 
data_files = glob.glob('%s/*.mat' % data_path)

# Only fit every third data file to save time
data_files = data_files[::3]

loader = 'sabes'
analysis_type = 'dimreduc'
 
seeds = np.arange(20)
loader_args = [{'bin_width':50, 'filter_fn':'none', 'filter_kwargs':{}, 'boxcox':0.5, 'spike_threshold':100, 'region':'M1'}]
dimvals = np.arange(1, 31, 2)
task_args = [{'dim_vals':dimvals, 'n_folds':5, 'dimreduc_method':'LQGCA', 'dimreduc_args': {'T':3, 'loss_type':'trace', 'n_init':1, 'rng_or_seed':seed}} for seed in seeds]
