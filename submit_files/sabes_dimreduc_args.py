import glob
import os
import numpy as np
from sklearn.model_selection import KFold

#script_path = '/global/homes/a/akumar25/repos/neural_control/batch_analysis.py'
script_path = '/home/akumar/nse/neural_control/batch_analysis.py'

desc = 'Fits of dimreduc methods to loco data'

#data_path = os.environ['SCRATCH'] + '/sabes'
data_path = '/mnt/Secondary/data/sabes'    
 
# These are the data files that contain both M1 and S1 recordings.
data_files = glob.glob('%s/loco*' % data_path)
#data_files = [data_files[0], data_files[5]]
  
 # Load the data files and determine how many dof (neurons) there are in each recording
 # data_files = ['%s/%s' % (data_path, data_file) for data_file in data_files]
 
loader = 'sabes'
analysis_type = 'dimreduc'
 
 # Each of these can be made into a list whose outer product is taken
loader_args = [{'bin_width':50, 'filter_fn':'none', 'filter_kwargs':{}, 'boxcox':0.5, 'spike_threshold':100, 'region':'M1'},
               {'bin_width':50, 'filter_fn':'none', 'filter_kwargs':{}, 'boxcox':0.5, 'spike_threshold':100, 'region': 'S1'}]

n_folds=5
# loader_args = [{'bin_width':50, 'filter_fn':'none', 'filter_kwargs':{}, 'boxcox':0.5, 'spike_threshold':100}]
task_args = [{'dim_vals':np.arange(1, 31), 'n_folds':5, 'dimreduc_method':'DCA', 'dimreduc_args': {'T':3, 'n_init':5}},
             {'dim_vals':np.arange(1, 31), 'n_folds':5, 'dimreduc_method':'KCA', 'dimreduc_args': {'T':3, 'causal_weights':(1, 0), 'n_init':5}},
             {'dim_vals':np.arange(1, 31), 'n_folds':5, 'dimreduc_method':'KCA', 'dimreduc_args': {'T':3, 'causal_weights':(0, 1), 'n_init':5}},
             {'dim_vals':np.arange(1, 31), 'n_folds':5, 'dimreduc_method':'LQGCA', 'dimreduc_args': {'T':3, 'loss_type':'trace', 'n_init':5}},
             {'dim_vals':np.arange(1, 31), 'n_folds':5, 'dimreduc_method':'PCA', 'dimreduc_args': {}}]

