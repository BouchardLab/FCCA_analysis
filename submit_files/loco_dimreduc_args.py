import glob
import os
import numpy as np
from sklearn.model_selection import KFold

script_path = '/home/akumar/nse/neural_control/batch_analysis.py'

desc = 'Loco dimreduc at 25 ms across all recording sessions'
#desc = 'Fits of dimreduc methods to loco data'

#data_path = os.environ['SCRATCH'] + '/sabes'
data_path = '/mnt/Secondary/data/sabes'    
 
data_files = glob.glob('%s/loco*' % data_path)

# This is the one indy file that has S1 data
data_files.append('/mnt/Secondary/data/sabes/indy_20160426_01.mat')

#data_files = glob.glob('%s/loco*' % data_path)
#data_files = [data_files[0], data_files[5]]
  
 # Load the data files and determine how many dof (neurons) there are in each recording
 # data_files = ['%s/%s' % (data_path, data_file) for data_file in data_files]
 
loader = 'sabes'
analysis_type = 'dimreduc'
  
 # Each of these can be made into a list whose outer product is taken
loader_args = [{'bin_width':25, 'filter_fn':'none', 'filter_kwargs':{}, 'region':'M1'}]

n_folds=5
# loader_args = [{'bin_width':50, 'filter_fn':'none', 'filter_kwargs':{}, 'boxcox':0.5, 'spike_threshold':100}]
task_args = [{'dim_vals':np.arange(1, 31), 'n_folds':5, 'dimreduc_method':'LQGCA', 'dimreduc_args': {'T':3, 'loss_type':'trace', 'n_init':10}},
             {'dim_vals':np.arange(1, 31), 'n_folds':5, 'dimreduc_method':'LQGCA', 'dimreduc_args': {'T':5, 'loss_type':'trace', 'n_init':10}},  
             {'dim_vals':np.arange(1, 31), 'n_folds': 5, 'dimreduc_method':'PCA', 'dimreduc_args':{}}]