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

def get_scalar(df_, stat, neu_idx):

    if stat == 'decoding_weights':
        decoding_win = df_.iloc[0]['decoder_params']['decoding_window']
        c = calc_loadings(df_.iloc[0]['decoding_weights'].T, d=decoding_win)[neu_idx]
    elif stat == 'encoding_weights':
        decoding_win = df_.iloc[0]['decoder_params']['decoding_window']
        c =  calc_loadings(df_.iloc[0]['encoding_weights'], d=decoding_win)[neu_idx]        
    elif stat in ['su_r2_pos', 'su_r2_vel', 'su_var', 'su_mmse', 'su_pi', 'su_fcca']:
        c = df_.iloc[0][stat][neu_idx]
    return c

def get_su_scalars(loadings_df, su_df):
    # Collect the desired single unit statistics into an array with the same ordering as those present in the loadings df
    stats = ['decoding_weights', 'encoding_weights', 'su_r2_pos', 'su_r2_vel', 
            'su_var', 'su_mmse', 'su_pi', 'su_fcca']

    carray = np.zeros((loadings_df.shape[0], len(stats)))

    for i in range(loadings_df.shape[0]):
        for j, stat in enumerate(stats):
            # Grab the unique identifiers needed
            data_file = loadings_df.iloc[i]['data_file']
            nidx = loadings_df.iloc[i]['nidx']
            df_ = apply_df_filters(su_df, data_file=data_file)
            carray[i, j] = get_scalar(df_, stat, nidx)


    # Need to treat the encoding/decoding weights post projection as a special case
    # dims hard-coded (9) 
    carray2 = np.zeros((loadings_df.shape[0], 3, 4, 9))
    for i in range(loadings_df.shape[0]):
        data_file = loadings_df.iloc[i]['data_file']
        nidx = loadings_df.iloc[i]['nidx']
        df_ = apply_df_filters(su_df, data_file=data_file)

        carray2[i, 0, ...] = df_.iloc[0]['proj_dw_pos'][..., nidx]
        carray2[i, 1, ...] = df_.iloc[0]['proj_dw_vel'][..., nidx]
        carray2[i, 2, ...] = df_.iloc[0]['proj_ew'][..., nidx]


    return carray, carray2


def barycenter_plot1():


def barycentric_plot2():


def proj_dw_scatter():


def loaddings_correlation_summary():


if __name__ == '__main__':

    dpath = '/mnt/sdb1/nc_data/sabes'

    with open('/mnt/sdb1/nc_data/sabes_decoding_df.dat', 'rb') as f:
        df = pickle.load(f)

    # Calculate 