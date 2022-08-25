import glob
import os
import numpy as np
from sklearn.model_selection import KFold

script_path = '/home/akumar/nse/neural_control/batch_analysis.py'

desc = 'dimreduc across loader_params'
#desc = 'Fits of dimreduc methods to loco data'

#data_path = os.environ['SCRATCH'] + '/sabes'
data_path = '/mnt/Secondary/data/sabes_tmp'
data_files = glob.glob('%s/*.pkl' % data_path) 

loader = 'preprocessed'
analysis_type = 'dimreduc'
  
 # Each of these can be made into a list whose outer product is taken
loader_args = [{}]

n_folds=5
# loader_args = [{'bin_width':50, 'filter_fn':'none', 'filter_kwargs':{}, 'boxcox':0.5, 'spike_threshold':100}]
task_args = [{'dim_vals':np.arange(1, 31), 'n_folds':5, 'dimreduc_method':'LQGCA', 'dimreduc_args': {'T':3, 'loss_type':'trace', 'n_init':10}},
             {'dim_vals':np.arange(1, 31), 'n_folds':5, 'dimreduc_method':'LQGCA', 'dimreduc_args': {'T':1, 'loss_type':'trace', 'n_init':10}},
             {'dim_vals':np.arange(1, 31), 'n_folds':5, 'dimreduc_method':'LQGCA', 'dimreduc_args': {'T':6, 'loss_type':'trace', 'n_init':10}},  
             {'dim_vals':np.arange(1, 31), 'n_folds': 5, 'dimreduc_method':'PCA', 'dimreduc_args':{}}]