import glob
import os
import numpy as np

#script_path = '/global/homes/a/akumar25/repos/neural_control/batch_analysis.py'
script_path = '/home/akumar/nse/neural_control/batch_analysis.py'
desc = 'Revsiting shenoy dataset with trial aggregation and KCA'
#data_path = os.environ['SCRATCH'] + '/shenoy'
data_path = '/mnt/Secondary/data/shenoy'
data_files = ['%s/RC,2009-09-18,1-2,good-ss.mat' % data_path]

# Load the data files and determine how many dof (neurons) there are in each recording
# data_files = ['%s/%s' % (data_path, data_file) for data_file in data_files]

loader = 'shenoy'
analysis_type = 'dimreduc'

# Explore different bin widths, smoothing timescales, and whether all trialVersions are included
#loader_args = [{'bin_width':1, 'filter_fn':'gaussian', 'filter_kwargs':{'sigma':sigma}, 
#                'boxcox':0.5, 'spike_threshold':None, 'region':region, 'trialVersions':version}
#                for sigma in [2, 5, 10, 40] for bin_width in [1, 2, 5, 20] 
#                for region in ['both', 'M1'] for version in ['all', 0, 1, 2]]

# No smoothing, only varied bin widths
#loader_args.extend([{'bin_width':1, 'filter_fn':'none', 'filter_kwargs':{}, 
#                     'boxcox':0.5, 'spike_threshold':None, 'region':region, 
#                     'trialVersions':version}
#                     for bin_width in [1, 2, 5, 20] 
#                     for region in ['both', 'M1'] for version in ['all', 0, 1, 2]])

# Test run : Just the most straightforward parameters
loader_args = [{'bin_width': 1, 'filter_fn':'gaussian', 'filter_kwargs': {'sigma':20}, 'boxcox':0.5,
                'spike_threshold':None, 'region':region, 'trialVersions':version}
                for region in ['both', 'M1'] for version in ['all', 0, 1, 2]]

# Each of these can be made into a list whose outer product is taken
task_args = [{'dim_vals':np.arange(1, 50), 'n_folds':5, 'T':T_} for T_ in [1, 3, 5]]
