import glob
import os
import numpy as np
from sklearn.model_selection import KFold

#script_path = '/global/homes/a/akumar25/repos/neural_control/batch_analysis.py'
script_path = '/home/akumar/nse/neural_control/batch_analysis.py'

desc = 'Sabes dimreduc in the lagged state space to enable comparison of subspace angles with the supervised subspace'

#data_path = os.environ['SCRATCH'] + '/sabes'
data_path = '/mnt/Secondary/data/sabes'
 
data_files = glob.glob('%s/*.mat' % data_path)
 
 # Load the data files and determine how many dof (neurons) there are in each recording
 # data_files = ['%s/%s' % (data_path, data_file) for data_file in data_files]
 
loader = 'sabes'
analysis_type = 'dimreduc'
 
 # Each of these can be made into a list whose outer product is taken
loader_args = [{'bin_width':50, 'filter_fn':'none', 'filter_kwargs':{}, 'boxcox':0.5, 'spike_threshold':100}]

n_folds=5
# loader_args = [{'bin_width':50, 'filter_fn':'none', 'filter_kwargs':{}, 'boxcox':0.5, 'spike_threshold':100}]
task_args = [{'dim_vals':np.arange(1, 31), 'n_folds':5, 'lag':5, 'dimreduc_method':'DCA', 'dimreduc_args': {'T':1, 'n_init':5}},
             {'dim_vals':np.arange(1, 31), 'n_folds':5, 'lag':5, 'dimreduc_method':'KCA', 'dimreduc_args': {'T':1, 'causal_weights':(1, 0), 'n_init':5}},
             {'dim_vals':np.arange(1, 31), 'n_folds':5, 'lag':5, 'dimreduc_method':'KCA', 'dimreduc_args': {'T':1, 'causal_weights':(0, 1), 'n_init':5}},
             {'dim_vals':np.arange(1, 31), 'n_folds':5, 'lag':5, 'dimreduc_method':'KCA', 'dimreduc_args': {'T':1, 'causal_weights':(1, 1), 'n_init':5}},
             {'dim_vals':np.arange(1, 31), 'n_folds':5, 'lag':5, 'dimreduc_method':'LQGCA', 'dimreduc_args': {'T':1, 'loss_type':'trace', 'n_init':5}},
             {'dim_vals':np.arange(1, 31), 'n_folds':5, 'lag':5, 'dimreduc_method':'LQGCA', 'dimreduc_args': {'T':1, 'loss_type':'fro', 'n_init':5}},
             {'dim_vals':np.arange(1, 31), 'n_folds':5, 'lag':5, 'dimreduc_method':'LQGCA', 'dimreduc_args': {'T':1, 'loss_type':'logdet', 'n_init':5}},
             {'dim_vals':np.arange(1, 31), 'n_folds':5, 'lag':5, 'dimreduc_method':'LQGCA', 'dimreduc_args': {'T':1, 'loss_type':'additive', 'n_init':5}},
             {'dim_vals':np.arange(1, 31), 'n_folds':5, 'lag':5, 'dimreduc_method':'PCA', 'dimreduc_args': {}}]

