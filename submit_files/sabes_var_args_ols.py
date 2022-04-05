import glob
import os

<<<<<<< Updated upstream
script_path = '/global/homes/a/akumar25/repos/neural_control/batch_analysis.py'
#script_path = '/home/akumar/nse/neural_control/batch_analysis.py'
desc = 'cross validate, disregard selection, want to use cross validated ccm estimation as the criteria. Fit on M1 and S1, and marginal models'
=======
#script_path = '/global/homes/a/akumar25/repos/neural_control/batch_analysis.py'
script_path = '/home/akumar/nse/neural_control/batch_analysis.py'
desc = 'cross-validated indy fits'
>>>>>>> Stashed changes

data_path = os.environ['SCRATCH'] + '/sabes'
#data_path = '/mnt/Secondary/data/sabes'    
 
# These are the data files that contain both M1 and S1 recordings.
data_files = glob.glob('%s/indy*' % data_path)
data_files = data_files[0:3]
#data_files.append('%s/indy_20160426_01.mat' % data_path)
#data_files = ['/lol%d' % i for i in range(15)]

 # Load the data files and determine how many dof (neurons) there are in each recording
 # data_files = ['%s/%s' % (data_path, data_file) for data_file in data_files]
 
loader = 'sabes'
analysis_type = 'var'
 
 # Each of these can be made into a list whose outer product is taken
loader_args = [{'bin_width':50, 'filter_fn':'none', 'filter_kwargs':{}, 'boxcox':0.5, 'spike_threshold':100, 'region':'M1'}]

# Estimation score not important since we save all estimates and do selection subsequently
task_args = [{'estimator': 'ols', 'self_regress':False, 'order':order, 'estimation_score':'null', 'fold_idx':idx, 'distributed_save':False} 
              for order in [1, 2, 3, 4, 5]
              for idx in range(5)]
