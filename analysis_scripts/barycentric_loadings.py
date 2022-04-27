import pdb
import numpy as np
import matplotlib.pyplot as plt
import scipy 
import time
import glob
import pickle
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LinearRegression

import sys

sys.path.append('/home/akumar/nse/neural_control')
from utils import apply_df_filters, calc_loadings
from loaders import load_sabes

def calc_loadings(df):

    # Try the raw leverage scores instead
    loadings_l = []
    data_files = np.unique(df['data_file'].values)

    for i, data_file in tqdm(enumerate(data_files)):
        # Assemble loadings from dims 2-10
        for d in range(2, 11):
            loadings = []
            loadings_unnorm = []
            angles = []
            for dimreduc_method in ['DCA', 'KCA', 'LQGCA', 'PCA']:
                loadings_fold = []
                loadings_unnorm_fold = []
                angles_fold = []
                for fold_idx in range(5):            
                    df_ = apply_df_filters(df, data_file=data_file, fold_idx=fold_idx, dim=d, dimreduc_method=dimreduc_method)
                    if dimreduc_method == 'LQGCA':
                        df_ = apply_df_filters(df_, dimreduc_args={'T': 3, 'loss_type': 'trace', 'n_init': 5})
                    V = df_.iloc[0]['coef']
                    if dimreduc_method == 'PCA':
                        V = V[:, 0:2]        

                    loadings_fold.append(calc_loadings(V))
                    loadings_unnorm_fold.append(np.linalg.norm(V, axis=1))
                    angles_fold.append(np.arccos(np.linalg.norm(V, axis=1)))

                # Average loadings across folds
                loadings.append(np.mean(np.array(loadings_fold), axis=0))
                loadings_unnorm.append(np.mean(np.array(loadings_unnorm_fold), axis=0))
                angles.append(np.mean(np.array(angles_fold), axis=0))

            for j in range(loadings[0].size):
                d_ = {}
                d_['data_file'] = data_file
                d_['DCA_loadings'] = loadings[0][j]
                d_['KCA_loadings'] = loadings[1][j]
                d_['FCCA_loadings'] = loadings[2][j]
                d_['PCA_loadings'] = loadings[3][j]

                d_['DCA_lnorms'] = loadings_unnorm[0][j]            
                d_['KCA_lnorms'] = loadings_unnorm[1][j]            
                d_['FCCA_lnorms'] = loadings_unnorm[2][j]            
                d_['PCA_lnorms'] = loadings_unnorm[3][j]            

                d_['DCA_angles'] = angles[0][j]
                d_['KCA_angles'] = angles[1][j]
                d_['FCCA_angles'] = angles[2][j]
                d_['PCA_angles'] = angles[3][j]

                d_['nidx'] = j
                d_['dim'] = d
                loadings_l.append(d_)           
           

    loadings_df = pd.DataFrame(loadings_l)
    return loadings_df


def calc_su_statistics():
    pass


if __name__ == '__main__':

    dpath = '/mnt/sdb1/nc_data/sabes'

    with open('/mnt/sdb1/nc_data/sabes_decoding_df.dat', 'rb') as f:
        df = pickle.load(f)

    # Calculate 