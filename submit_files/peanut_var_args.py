import glob
import os
import numpy as np
from sklearn.model_selection import KFold

#script_path = '/global/homes/a/akumar25/repos/neural_control/batch_analysis.py'
script_path = '/home/akumar/nse/neural_control/batch_analysis.py'
desc = 'VAR fits to peanut, using both regions, cross-validated ccm accuracy based selection'

#data_path = os.environ['SCRATCH'] + '/peanut'
data_path = '/mnt/Secondary/data/peanut'
 

#data_files = glob.glob('%s/*.mat' % data_path)
data_files = ['%s/data_dict_peanut_day14.obj' % data_path]

 # Load the data files and determine how many dof (neurons) there are in each recording
 # data_files = ['%s/%s' % (data_path, data_file) for data_file in data_files]
 
loader = 'peanut'
analysis_type = 'var'
 
 # Each of these can be made into a list whose outer product is taken
# Bin widths were selected based on initial decoding results in FrankLab notebook (in grant_notebooks repo)
loader_args = [{'bin_width':25, 'epoch': epoch, 'filter_fn':'none', 'filter_kwargs':{}, 'boxcox':0.5, 'spike_threshold':100, 'region':r}
               for epoch in np.arange(2, 18, 2) for r in ['HPc', 'OFC', 'both']]

n_folds=5
# Estimation score not important since we save all estimates and do selection subsequently
task_args = [{'estimator': 'ols', 'self_regress':False, 'order':order, 'estimation_score':'null', 'fold_idx':idx, 'distributed_save':False} 
              for order in [1, 2, 3, 4, 5] for idx in range(5)]

# task_args = [{'estimator': 'uoi', 'penalty': 'scad', 'self_regress':False, 'continuous':True,
#               'fit_type':'union_only', 'idxs':idx, 'order':1, 'estimation_score':'gMDL', 'n_folds':n_folds}
#               for idx in np.arange(n_folds)]
