import glob
import os
import numpy as np
from sklearn.model_selection import KFold

script_path = '/global/homes/a/akumar25/repos/neural_control/batch_analysis.py'
desc = 'Fitting continuous time VAR models to peanut data"'

data_path = os.environ['SCRATCH'] + '/peanut'
 #data_path = '/media/akumar/Secondary/data/sabes'
 
#data_files = glob.glob('%s/*.mat' % data_path)
data_files = ['%s/data_dict_peanut_day14.obj' % data_path]

 # Load the data files and determine how many dof (neurons) there are in each recording
 # data_files = ['%s/%s' % (data_path, data_file) for data_file in data_files]
 
loader = 'peanut'
analysis_type = 'var'
 
 # Each of these can be made into a list whose outer product is taken
# Bin widths were selected based on initial decoding results in FrankLab notebook (in grant_notebooks repo)
loader_args = [{'bin_width':25, 'epoch': epoch, 'filter_fn':'none', 'filter_kwargs':{}, 'boxcox':0.5, 'spike_threshold':200}
               for epoch in np.arange(2, 18, 2)]

n_folds=5
task_args = [{'estimator': 'uoi', 'penalty': 'scad', 'self_regress':False, 'continuous':True,
              'fit_type':'union_only', 'idxs':idx, 'order':1, 'estimation_score':'gMDL', 'n_folds':n_folds}
              for idx in np.arange(n_folds)]
