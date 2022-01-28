import glob
import os
import numpy as np

#script_path = '/global/homes/a/akumar25/repos/localization/batch_analysis_sabes.py'

script_path = '/home/akumar/nse/neural_control/batch_analysis.py'

desc = 'Optimality Principle based dimreduc on Peanut Data, methods operate on the autocorrelation sequence'
data_path = '/mnt/Secondary/data/peanut'
data_files = ['%s/data_dict_peanut_day14.obj' % data_path]

loader = 'peanut'
analysis_type = 'parametric_dimreduc'

# Each of these can be made into a list whose outer product is taken
loader_args = [{'bin_width':25, 'epoch': epoch, 'filter_fn':'none', 'filter_kwargs':{}, 'boxcox':0.5, 'spike_threshold':200, 'speed_threshold':4}
               for epoch in np.arange(2, 18, 2)]

# loader_args = [{'bin_width':50, 'filter_fn':'none', 'filter_kwargs':{}, 'boxcox':0.5, 'spike_threshold':100}]
task_args = [{'dim_vals':np.arange(1, 31), 'n_folds':5, 'dimreduc_method':'DCA', 'dimreduc_args': {'T':3}},
             {'dim_vals':np.arange(1, 31), 'n_folds':5, 'dimreduc_method':'KCA', 'dimreduc_args': {'T':3, 'causal_weights':(1, 0)}},
             {'dim_vals':np.arange(1, 31), 'n_folds':5, 'dimreduc_method':'KCA', 'dimreduc_args': {'T':3, 'causal_weights':(0, 1)}},
             {'dim_vals':np.arange(1, 31), 'n_folds':5, 'dimreduc_method':'KCA', 'dimreduc_args': {'T':3, 'causal_weights':(1, 1)}},
             {'dim_vals':np.arange(1, 31), 'n_folds':5, 'dimreduc_method':'LQGCA', 'dimreduc_args': {'T':3, 'loss_type':'trace'}},
             {'dim_vals':np.arange(1, 31), 'n_folds':5, 'dimreduc_method':'LQGCA', 'dimreduc_args': {'T':3, 'loss_type':'fro'}},
             {'dim_vals':np.arange(1, 31), 'n_folds':5, 'dimreduc_method':'LQGCA', 'dimreduc_args': {'T':3, 'loss_type':'logdet'}},
             {'dim_vals':np.arange(1, 31), 'n_folds':5, 'dimreduc_method':'LQGCA', 'dimreduc_args': {'T':3, 'loss_type':'additive'}},
             {'dim_vals':np.arange(1, 31), 'n_folds':5, 'dimreduc_method':'PCA', 'dimreduc_args': {}}]

