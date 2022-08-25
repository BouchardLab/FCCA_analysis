import glob
import os
import numpy as np
from sklearn.model_selection import KFold

#script_path = '/global/homes/a/akumar25/repos/neural_control/batch_analysis.py'
script_path = '/home/akumar/nse/neural_control/batch_cc_analysis.py'

desc = 'Assessing how various analyses are impacted by choice of loader parameters, with an eye towards detecting M1/S1 interactions'

# Pre-processed M1/S1 data
data_path = '/mnt/Secondary/data/sabes_tmp'
data_files = glob.glob('%s/*.pkl' % data_path)
 
analysis_type = ''

# Encode each analysis time and an interation over the distinct combination of task parameters so we can parallelize over them
# Use the existing infrastructure to do VAR and dimreduc
loader = 'preprocessed'
loader_args = [{}]
lags = np.array([4, 2, 0])
windows = np.array([5, 3, 1])
task_args = [{'task_type':'cca', 'task_args':{'lags':lags, 'windows':windows}}]
