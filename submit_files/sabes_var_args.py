import glob
import os

script_path = '/global/homes/a/akumar25/repos/neural_control/batch_analysis.py'
#script_path = '/home/akumar/nse/localization/batch_analysis_sabes.py'
#script_path = '/home/akumar/nse/neural_control/batch_analysis.py'
desc = 'cross validate, disregard selection, want to use cross validated ccm estimation as the criteria. Fit on M1 and S1, and marginal models'

data_path = os.environ['SCRATCH'] + '/sabes'
#data_path = '/mnt/Secondary/data/sabes'    
 
# These are the data files that contain both M1 and S1 recordings.
data_files = glob.glob('%s/loco*' % data_path)
data_files.append('%s/indy_20160426_01.mat' % data_path)
#data_files = ['/lol%d' % i for i in range(15)]

 # Load the data files and determine how many dof (neurons) there are in each recording
 # data_files = ['%s/%s' % (data_path, data_file) for data_file in data_files]
 
loader = 'sabes'
analysis_type = 'var'
 
 # Each of these can be made into a list whose outer product is taken
loader_args = [{'bin_width':50, 'filter_fn':'none', 'filter_kwargs':{}, 'boxcox':0.5, 'spike_threshold':100, 'region':'M1'},
               {'bin_width':50, 'filter_fn':'none', 'filter_kwargs':{}, 'boxcox':0.5, 'spike_threshold':100, 'region':'S1'},
               {'bin_width':50, 'filter_fn':'none', 'filter_kwargs':{}, 'boxcox':0.5, 'spike_threshold':100, 'region':'both'}]

# Estimation score not important since we save all estimates and do selection subsequently
task_args = [{'estimator': 'uoi', 'penalty': 'scad', 'self_regress':sr, 'continuous':False, 'n_boots_sel':1, 'selection_frac':1., 'n_boots_est': 1, 'estimation_frac':0.9, 
              'fit_type':'union_only', 'order':order, 'estimation_score':'null', 'fold_idx':idx, 'distributed_save':True} 
              for order in [1, 2, 3, 4, 5] for sr in [True, False] for fold_idx in range(5)]
