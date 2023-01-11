import glob
import os
import numpy as np
import itertools

#script_path = '/global/homes/a/akumar25/repos/localization/batch_analysis_sabes.py'
script_path = '/home/akumar/nse/neural_control/batch_analysis.py'

desc = 'Decode from peanut_dimreduc_initvar'
#data_path = os.environ['SCRATCH'] + '/shenoy_split'
data_path = '/mnt/Secondary/data/peanut'

# Data files specified by dimreduc files
data_files = ['']

# Load the data files and determine how many dof (neurons) there are in each recording
# data_files = ['%s/%s' % (data_path, data_file) for data_file in data_files]

loader = 'peanut'
analysis_type = 'decoding'

# Loader args are taken from the dimreduc files
loader_args = [[]]

# Grab all the dimreduc files. 
dimreduc_files = glob.glob('/mnt/Secondary/data/peanut_dimreduc_initvar/peanut_dimreduc_initvar*.dat')

# Create separate set of task args for each dimreduc file and
# each set of decoder_args. The rest of the iterables are handled
# in parallel at execution

decoders = [{'method': 'lr', 'args':{'trainlag': 0, 'testlag': 0, 'decoding_window':6}}]

task_args = []
for param_comb in itertools.product(dimreduc_files, decoders):
	task_args.append({'dimreduc_file':param_comb[0],
					  'decoder':param_comb[1]})
