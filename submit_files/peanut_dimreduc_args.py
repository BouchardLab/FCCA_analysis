import glob
import os
import numpy as np

#script_path = '/global/homes/a/akumar25/repos/localization/batch_analysis_sabes.py'
script_path = '/home/akumar/nse/neural_control/batch_analysis.py'

desc = 'Peanut dimreduc looking at variability over initializations'
data_path = '/mnt/Secondary/data/peanut'
data_files = ['%s/data_dict_peanut_day14.obj' % data_path]

loader = 'peanut'
analysis_type = 'dimreduc'

# Each of these can be made into a list whose outer product is taken
loader_args = [{'bin_width':50, 'epoch': epoch, 'filter_fn':'none', 'filter_kwargs':{}, 'boxcox':0.5, 'spike_threshold':200, 'speed_threshold':4}
               for epoch in np.arange(2, 18, 2)]

# loader_args = [{'bin_width':50, 'filter_fn':'none', 'filter_kwargs':{}, 'boxcox':0.5, 'spike_threshold':100}]
dimvals = np.arange(1, 31, 2)
seeds = np.arange(20)
task_args = [{'dim_vals':dimvals, 'n_folds':5, 'dimreduc_method':'LQGCA', 'dimreduc_args': {'T':3, 'loss_type':'trace', 'n_init':1, 'rng_or_seed':seed}} for seed in seeds]

