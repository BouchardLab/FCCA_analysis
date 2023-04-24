import pdb
import numpy as np
import matplotlib.pyplot as plt
import scipy 
import time
import torch
import glob
import pickle
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

import sys

sys.path.append('/home/akumar/nse/neural_control')
from utils import apply_df_filters, calc_loadings, calc_cascaded_loadings
from loaders import load_sabes, load_peanut, load_cv
from decoders import lr_decoder, lr_encoder
from subspaces import SubspaceIdentification, IteratedStableEstimator, estimate_autocorrelation

from dca.cov_util import calc_cross_cov_mats_from_data, calc_pi_from_cross_cov_mats
from dca_research.kca import calc_mmse_from_cross_cov_mats
from dca_research.lqg import build_loss as build_lqg_loss

good_loco_files = ['loco_20170210_03.mat',
                   'loco_20170213_02.mat',
                   'loco_20170215_02.mat',
                   'loco_20170227_04.mat',
                   'loco_20170228_02.mat',
                   'loco_20170301_05.mat',
                   'loco_20170302_02.mat']

with open('/mnt/Secondary/data/postprocessed/loco_decoding_df.dat', 'rb') as f:
    result_list = pickle.load(f)
with open('/mnt/Secondary/data/postprocessed/indy_S1_df.dat', 'rb') as f:
    rl2 = pickle.load(f)


sabes_df = pd.DataFrame(result_list)
indy_df = pd.DataFrame(rl2)
sabes_df = pd.concat([sabes_df, indy_df])

good_loco_files.append(indy_df.iloc[0]['data_file'])
loader_arg = {'bin_width':50, 'filter_fn':'none', 'filter_kwargs':{}, 'boxcox':0.5, 'spike_threshold':100, 'region':'S1'}
decoder_arg = sabes_df.iloc[0]['decoder_args']

s1df = apply_df_filters(sabes_df, decoder_args=decoder_arg, loader_args=loader_arg)
s1df = apply_df_filters(s1df, data_file=good_loco_files)

sabes_su_l = []
decoder_params = {'trainlag': 4, 'testlag': 4, 'decoding_window': 5}
#data_path = '/mnt/sdb1/nc_data/sabes'
data_path = '/mnt/Secondary/data/sabes'
data_files = np.unique(s1df['data_file'].values)
print(len(data_files))
pdb.set_trace()
for i, data_file in tqdm(enumerate(data_files)):    
    dat = load_sabes('%s/%s' % (data_path, data_file), bin_width=loader_arg['bin_width'],
                     filter_fn=loader_arg['filter_fn'], filter_kwargs=loader_arg['filter_kwargs'],
                     boxcox=loader_arg['boxcox'], spike_threshold=loader_arg['spike_threshold'], region='S1')
    
    X = np.squeeze(dat['spike_rates'])
    Z = dat['behavior']

    kfold = KFold(n_splits=5, shuffle=False)

    # Average results across folds
    decoding_weights = []
    encoding_weights = []
    su_decoding_weights = []
    su_encoding_weights = []
    su_r2_pos = []
    su_r2_vel = []
    su_r2_enc = []

    # Single unit statistics
    su_var = np.zeros((5, X.shape[-1]))
    su_act = np.zeros((5, X.shape[-1]))

    for fold_idx, (train_idxs, test_idxs) in enumerate(kfold.split(X)):

        r = {}

        ztrain = Z[train_idxs, :]
        ztest = Z[test_idxs, :]

        # Population level decoding/encoding - use the coefficient in the linear fit
        # Record both the weights in the coefficient but also the loadings onto the SVD

        xtrain = X[train_idxs, :]
        xtest = X[test_idxs, :]

        ccm = calc_cross_cov_mats_from_data(xtrain, T=20)
        ccm = torch.tensor(ccm)

        _, _, _, decodingregressor = lr_decoder(xtest, xtrain, ztest, ztrain, **decoder_params)
        _, encodingregressor = lr_encoder(xtest, xtrain, ztest, ztrain, **decoder_params)

        decoding_weights.append(decodingregressor.coef_)
        encoding_weights.append(encodingregressor.coef_)                
        
        r2_pos_decoding, r2_vel_decoding, r2_encoding = [], [], []
        
        su_dw = []
        su_ew = []            
        sur2pos = []
        sur2vel = []
        sur2enc = []
        for neu_idx in range(X.shape[-1]):           #Fit all neurons one by one
            
            xtrain = X[train_idxs, neu_idx][:, np.newaxis]
            xtest = X[test_idxs, neu_idx][:, np.newaxis]

            # Decoding
            r2_pos, r2_vel, _, dr = lr_decoder(xtest, xtrain, ztest, ztrain, **decoder_params)
            r2_pos_decoding.append(r2_pos)
            r2_vel_decoding.append(r2_vel)
            su_dw.append(dr.coef_)
            sur2pos.append(r2_pos)
            sur2vel.append(r2_vel)

            # Encoding
            r2_encoding_, er = lr_encoder(xtest, xtrain, ztest, ztrain, **decoder_params)
            r2_encoding.append(r2_encoding_)
            su_ew.append(er.coef_)        
            sur2enc.append(r2_encoding_)


        su_decoding_weights.append(np.array(su_dw))
        su_encoding_weights.append(np.array(su_ew))
        
        su_r2_pos.append(np.array(sur2pos))
        su_r2_vel.append(np.array(sur2vel))
        su_r2_enc.append(np.array(sur2enc))
        
        for neu_idx in range(X.shape[-1]):

            xtrain = X[train_idxs, neu_idx][:, np.newaxis]
            xtest = X[test_idxs, neu_idx][:, np.newaxis]

            su_var[fold_idx, neu_idx] = np.var(xtrain)
            
            ccm_j = ccm[:, neu_idx, neu_idx].numpy()
            ccm_j /= ccm_j[0]

            thr = 1e-1
            acov_crossing = np.where(ccm_j < thr)
            if len(acov_crossing[0]) > 0:
                su_act[fold_idx, neu_idx] = np.where(ccm_j < thr)[0][0]
            else:
                su_act[fold_idx, neu_idx] = len(ccm)


        # Calculate decoding weights based on projection of the data first

    # Average results across folds
    decoding_weights = np.mean(np.array(decoding_weights), axis=0)
    encoding_weights = np.mean(np.array(encoding_weights), axis=0)
    su_decoding_weights = np.mean(np.array(su_decoding_weights), axis=0)
    su_encoding_weights = np.mean(np.array(su_encoding_weights), axis=0)
    
    su_r2_pos = np.mean(np.array(su_r2_pos), axis=0)
    su_r2_vel = np.mean(np.array(su_r2_vel), axis=0)
    su_r2_enc = np.mean(np.array(su_r2_enc), axis=0)

    su_var = np.mean(su_var, axis=0)
    su_act = np.mean(su_act, axis=0)

    result = {}
    for variable in ('data_file', 'decoding_weights', 'encoding_weights', 'su_decoding_weights', 'su_encoding_weights', 'su_r2_pos',
                     'su_r2_vel', 'su_r2_enc', 'su_var', 'su_act', 'decoder_params'):
        result[variable] = eval(variable)

    sabes_su_l.append(result)

with open('/mnt/Secondary/data/postprocessed/sabes_su_calcsS1.dat', 'wb') as f:
    f.write(pickle.dumps(sabes_su_l))
