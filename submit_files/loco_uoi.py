import glob
import os

script_path = '/global/homes/a/akumar25/repos/neural_control/batch_analysis.py'
#script_path = '/home/akumar/nse/neural_control/batch_analysis.py'
desc = 'UoI on Loco'

data_path = os.environ['SCRATCH'] + '/sabes/preprocessed'
# data_path = '/mnt/Secondary/data/sabes/preprocessed'    
 
# These are the data files that contain both M1 and S1 recordings.
data_files = glob.glob('%s/loco*' % data_path)
#data_files = [data_files[0], data_files[5]]
#data_files.append('%s/indy_20160426_01.mat' % data_path)
#data_files = ['/lol%d' % i for i in range(15)]

 # Load the data files and determine how many dof (neurons) there are in each recording
 # data_files = ['%s/%s' % (data_path, data_file) for data_file in data_files]
 
loader = 'preprocessed'
analysis_type = 'var'
 
 # Each of these can be made into a list whose outer product is taken
loader_args = [{'bin_width':50, 'filter_fn':'none', 'filter_kwargs':{}, 'boxcox':0.5, 'spike_threshold':100, 'region':'S1'}]
ft = ['uoi', 'union_only']
penalty = ['l1', 'scad']
# Estimation score not important since we save all estimates and do selection subsequently
task_args = [{'estimator': 'uoi', 'self_regress':False, 'order':order, 'estimation_score':'null', 'fold_idx':idx, 'distributed_save':True,
              'fit_type': ft[i], 'penalty': penalty[i]} 
              for order in [1, 2, 3]
              for idx in range(5) for i in range(2)]
