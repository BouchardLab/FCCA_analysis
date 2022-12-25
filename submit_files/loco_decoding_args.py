import glob
import os
import numpy as np
import itertools

#script_path = '/global/homes/a/akumar25/repos/neural_control/batch_analysis.py'
script_path = '/home/akumar/nse/neural_control/batch_analysis.py'

desc = 'Decode from loco dimreduc with some additional lags to fully compare to the decoding from CCA. We use dimreduc files from dimreduc_reduced which includes only bw 50, spike thresh 100'
data_path = '/mnt/Secondary/data/sabes'
#data_path = '/global/cscratch1/sd/akumar25/sabes'


# Data files specified by dimreduc files
data_files = [' ']

# Load the data files and determine how many dof (neurons) there are in each recording
# data_files = ['%s/%s' % (data_path, data_file) for data_file in data_files]

loader = 'sabes'
analysis_type = 'decoding'

# Loader args are taken from the dimreduc files
loader_args = [[]]

# Grab all the dimreduc files. 
dimreduc_files = glob.glob('/mnt/Secondary/data/loco_dimreduc_norm_reduced/loco_dimreduc_norm_*.dat')
#dimreduc_files = glob.glob(os.environ['SCRATCH'] + '/indy_dimreduc_parametric/indy_dimreduc_parametric_*.dat')

# each set of decoder_args. The rest of the iterables are handled
# in parallel at execution

decoders = [{'method': 'lr', 'args':{'trainlag': 1, 'testlag': 1, 'decoding_window':5}},p
            {'method': 'lr', 'args':{'trainlag': -1, 'testlag': -1, 'decoding_window':5}}]

task_args = []
for param_comb in itertools.product(dimreduc_files, decoders):
	task_args.append({'dimreduc_file':param_comb[0],
					  'decoder':param_comb[1]})
