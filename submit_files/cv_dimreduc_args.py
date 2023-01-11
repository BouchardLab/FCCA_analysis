import glob
import os
import numpy as np


script_path = '/home/akumar/nse/neural_control/batch_analysis.py'

desc = 'Looking at variability over initializations'
# data_path = '/mnt/Secondary/data/peanut'
# data_files = ['%s/data_dict_peanut_day14.obj' % data_path]
#data_path = '/media/akumar/Data/nse/data/neuraldata'

data_path = '/mnt/Secondary/data/cv'
#data_files = glob.glob('%s/*.h5' % data_path)
data_files = ['%s/EC2_hg.h5' % data_path]

loader = 'cv'
analysis_type = 'dimreduc'

# Each of these can be made into a list whose outer product is taken
loader_args = [{}]

# loader_args = [{'bin_width':50, 'filter_fn':'none', 'filter_kwargs':{}, 'boxcox':0.5, 'spike_threshold':100}]
dim_vals = np.arange(1, 31, 2)
seeds = np.arange(20)
task_args = [{'dim_vals':dim_vals, 'n_folds':1, 'dimreduc_method':'LQGCA', 'dimreduc_args': {'T':40, 'loss_type':'trace', 'n_init':1, 'rng_or_seed':seed}} 
             for seed in seeds]