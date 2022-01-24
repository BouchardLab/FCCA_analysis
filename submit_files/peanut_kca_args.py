import glob
import os
import numpy as np

#script_path = '/global/homes/a/akumar25/repos/localization/batch_analysis_sabes.py'
script_path = '/home/akumar/nse/localization/batch_analysis_sabes.py'
desc = 'Peanut dimreduc using (causal) KCA for cosyne'

# data_path = '/media/akumar/Secondary/data/sabes'
# data_files = glob.glob('%s/*.mat' % data_path)

data_path = '/media/akumar/Secondary/data/peanut'
data_files = ['%s/data_dict_peanut_day14.obj' % data_path]

loader = 'peanut'
analysis_type = 'dca'

# Each of these can be made into a list whose outer product is taken
loader_args = [{'bin_width':25, 'epoch': epoch, 'filter_fn':'none', 'filter_kwargs':{}, 'boxcox':0.5, 'spike_threshold':200, 'speed_threshold':4}
               for epoch in np.arange(2, 18, 2)]
# loader_args = [{'bin_width':50, 'filter_fn':'none', 'filter_kwargs':{}, 'boxcox':0.5, 'spike_threshold':100}]
task_args = [{'dim_vals':np.arange(1, 31), 'n_folds':5, 'T':3, 'ols_order':3}]

### NOTE: WE HARDCODED KCA in batch_analysis_sabes with loss_reg_vals (1, 0), project_mmse=False

