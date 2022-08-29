import os
import pickle
import pandas as pd
import glob
import numpy as np
from deepdiff import DeepDiff

import sys

sys.path.append('/home/akumar/nse/neural_control')
from loaders import load_sabes
from utils import apply_df_filters

# Consolidate decoding results. Keep track of the directory where results are from and the original 
def consolidate_decoding(src_path, save_path):

    argfiles = glob.glob('%s/arg*.dat' % src_path)
    
    result_list = []
    for argfile in argfiles:

        # Open up the arg files
        with open(argfile, 'rb') as f:
            args = pickle.load(f)

        rfile = args['results_file']
        with open(rfile, 'rb') as f:
            result = pickle.load(f)

        for result_ in result:
            for k, v in args.items():
                if type(k) == dict:
                    for k_, v_ in k.items():
                        result_[k_] = v_
                else:
                    result_[k] = v

        # Need to grab some information from the dimreduc arg file
        dimreduc_path = '/'.join(args['task_args']['dimreduc_file'].split('/')[:-1])
        dimreduc_no = args['task_args']['dimreduc_file'].split('_')[-1].split('.dat')[0]
        dimreduc_argfile = '%s/arg%s.dat' % (dimreduc_path, dimreduc_no)

        with open(dimreduc_argfile, 'rb') as f:
            dr_args = pickle.load(f)

        for result_ in result:
            for k, v in dr_args.items():
                if type(k) == dict:
                    for k_, v_ in k.items():
                        result_[k_] = v_
                else:
                    result_[k] = v

        result_list.extend(result)

    # For ease of use, use data file names that do not involve the directory path
    for r in result_list:
        r['data_file'] = r['data_file'].split('/')[-1]

    # Save the result list, directory path, a

    with open(save_path, 'wb') as f:
        f.write(pickle.dumps(result_list))
        f.write(pickle.dumps(src_path))


def consolidate_dimreduc():
    pass