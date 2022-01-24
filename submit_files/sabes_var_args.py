import glob
import os
import numpy as np
from sklearn.model_selection import KFold

script_path = '/global/homes/a/akumar25/repos/neural_control/batch_analysis.py'
#script_path = '/home/akumar/nse/localization/batch_analysis_sabes.py'
desc = 'gMDL selection, only order 1, cross validated for prediction, self regress FALSE, and CONTINUOUS TIME'

data_path = os.environ['SCRATCH'] + '/sabes'
 #data_path = '/media/akumar/Secondary/data/sabes'
 
data_files = glob.glob('%s/*.mat' % data_path)
 
 # Load the data files and determine how many dof (neurons) there are in each recording
 # data_files = ['%s/%s' % (data_path, data_file) for data_file in data_files]
 
loader = 'sabes'
analysis_type = 'var'
 
 # Each of these can be made into a list whose outer product is taken
loader_args = [{'bin_width':50, 'filter_fn':'none', 'filter_kwargs':{}, 'boxcox':0.5, 'spike_threshold':100}]

n_folds=5
task_args = [{'estimator': 'uoi', 'penalty': 'scad', 'self_regress':False, 
              'fit_type':'union_only', 'idxs':idx, 'order':1, 'estimation_score':'gMDL', 'n_folds':n_folds}
              for idx in np.arange(n_folds)]

