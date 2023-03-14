import pdb
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy 
import time
import glob
import pickle
import pandas as pd
from tqdm import tqdm
import itertools

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold


sys.path.append('/home/akumar/nse/neural_control')
from utils import apply_df_filters, calc_loadings

from loaders import load_peanut
from decoders import lr_decoder

if __name__ == '__main__':

    # Where to save?
    if len(sys.argv) > 1:
        figpath = sys.argv[1]
    else:
        figpath = '/home/akumar/nse/neural_control/figs/final'

    # # Sequentially do indy, peanut decoding
    with open('/home/akumar/nse/neural_control/data/indy_decoding_marginal.dat', 'rb') as f:
        rl = pickle.load(f)
    sabes_df = pd.DataFrame(rl)
    sabes_df = apply_df_filters(sabes_df, dimreduc_method='LQGCA')

    # Grab PCA results
    with open('/home/akumar/nse/neural_control/data/sabes_kca_decodign_df.dat', 'rb') as f:
        pca_decoding_df = pickle.load(f)

    data_files = np.unique(sabes_df['data_file'].values)
    dims = np.unique(sabes_df['dim'].values)
    r2fc = np.zeros((len(data_files), dims.size, 5, 3))

    for i, data_file in tqdm(enumerate(data_files)):
        for j, dim in enumerate(dims):               
            for f in range(5):
                dim_fold_df = apply_df_filters(sabes_df, data_file=data_file, dim=dim, fold_idx=f)
                # Trace loss
                try:
                    assert(dim_fold_df.shape[0] == 1)
                except:
                    pdb.set_trace()
                r2fc[i, j, f, :] = dim_fold_df.iloc[0]['r2']

    dims = np.unique(sabes_df['dim'].values)
    sr2_vel_pca = np.zeros((28, 30, 5))
    for i, data_file in enumerate(data_files):
        for j, dim in enumerate(dims):
            data_file = data_file.split('/')[-1]
            pca_df = apply_df_filters(pca_decoding_df, dim=dim, data_file=data_file, dr_method='PCA')        
            for k in range(pca_df.shape[0]):
                sr2_vel_pca[i, j, k] = pca_df.iloc[k]['r2'][1]

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))

    # Average across folds and plot
    # REINSERT OLS(5) IN HERE IF NEEDED

    colors = ['black', 'red', '#781820', '#5563fa']
    dim_vals =dims

    # # DCA averaged over folds
    # dca_r2 = np.mean(r2[:, :, 1, :, 1], axis=2)
    # # KCA averaged over folds
    # kca_r2 = np.mean(r2[:, :, 2, :, 1], axis=2)

    # FCCA averaged over folds
    fca_r2 = np.mean(r2fc[:, :, :, 1], axis=2)
    # PCA
    pca_r2 = np.mean(sr2_vel_pca, axis=-1)
    # ax.fill_between(dim_vals, np.mean(dca_r2, axis=0) + np.std(dca_r2, axis=0)/np.sqrt(28),
    #                 np.mean(dca_r2, axis=0) - np.std(dca_r2, axis=0)/np.sqrt(28), color=colors[0], alpha=0.25)
    # ax.plot(dim_vals, np.mean(dca_r2, axis=0), color=colors[0])
    # ax.fill_between(dim_vals, np.mean(kca_r2, axis=0) + np.std(kca_r2, axis=0)/np.sqrt(28),
    #                 np.mean(kca_r2, axis=0) - np.std(kca_r2, axis=0)/np.sqrt(28), color=colors[1], alpha=0.25)
    # ax.plot(dim_vals, np.mean(kca_r2, axis=0), color=colors[1])
    ax.fill_between(dim_vals, np.mean(fca_r2, axis=0) + np.std(fca_r2, axis=0)/np.sqrt(28),
                    np.mean(fca_r2, axis=0) - np.std(fca_r2, axis=0)/np.sqrt(28), color=colors[1], alpha=0.25)
    ax.plot(dim_vals, np.mean(fca_r2, axis=0), color=colors[1])

    ax.fill_between(dim_vals, np.mean(pca_r2, axis=0) + np.std(pca_r2, axis=0)/np.sqrt(28),
                    np.mean(pca_r2, axis=0) - np.std(pca_r2, axis=0)/np.sqrt(28), color=colors[0], alpha=0.25)
    ax.plot(dim_vals, np.mean(pca_r2, axis=0), color=colors[0])

    ax.set_xlabel('Dimension', fontsize=14)
    ax.set_ylabel('Velocity Decoding ' + r'$r^2$', fontsize=14)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)

    ax.legend(['FCCA', 'PCA'], loc='lower right', fontsize=14)
    ax.set_title('Macaque M1', fontsize=16)
    fig.tight_layout()
    fig.savefig('%s/indy_vel_decoding.pdf' % figpath, bbox_inches='tight', pad_inches=0)

    with open('/home/akumar/nse/neural_control/data/peanut_decoding_df.dat', 'rb') as f:
        peanut_decoding_df = pickle.load(f)

    peanut_decoding_df = pd.DataFrame(peanut_decoding_df)
    pdf_fca = apply_df_filters(peanut_decoding_df, dimreduc_method='LQGCA', dimreduc_args={'T':3, 'loss_type':'trace', 'n_init':5})

    #fig, ax = plt.subplots(4, 2, figsize=(10, 12))
    epochs = np.unique(peanut_decoding_df['epoch'].values)
    folds = np.unique(peanut_decoding_df['fold_idx'].values)
    dimvals = np.unique(peanut_decoding_df['dim'].values)
    decoder_args = [{'trainlag': 0, 'testlag': 0, 'decoding_window': 6}, {'trainlag': 3, 'testlag': 3, 'decoding_window': 6}, {'trainlag': 6, 'testlag': 6, 'decoding_window': 6}]

    r2 = np.zeros((epochs.size, len(decoder_args), folds.size, dimvals.size))

    for i, epoch in enumerate(epochs):
        for k, da in enumerate(decoder_args):
            for f, fold in enumerate(folds):
                for d, dimval in enumerate(dimvals):            
                    df_ = apply_df_filters(pdf_fca, epoch=epoch, fold_idx=fold, dim=dimval, decoder_args=da)
                    try:
                        assert(df_.shape[0] == 1)
                    except:
                        pdb.set_trace()
                    r2[i, k, f, d] = df_.iloc[0]['r2'][0]

    
    # Something went wrong with PCA results, just run them here real quick:
    pca_r2 = np.zeros((epochs.size, len(decoder_args), 5, dimvals.size))
    for i, epoch in tqdm(enumerate(epochs)):
        dat = load_peanut('/mnt/Secondary/data/peanut/data_dict_peanut_day14.obj', epoch=epoch, spike_threshold=200)
        X = dat['spike_rates']
        Y = dat['behavior']
        train_test = list(KFold(n_splits=5).split(X))
        for k, da in enumerate([decoder_args[0]]):
            for f, fold in enumerate(folds):
                for d, dimval in enumerate(dimvals):            
 
                    df_ = apply_df_filters(peanut_decoding_df, epoch=epoch, dimreduc_method='PCA', fold_idx=fold, dim=dimval, decoder_args=da)
                    try:
                        coef = df_.iloc[0]['coef'][:, 0:dimval]         
                    except:
                        pdb.set_trace()

                    train_idxs = train_test[f][0]
                    test_idxs = train_test[f][1]

                    Ytrain = Y[train_idxs]
                    Ytest = Y[test_idxs]

                    Xtrain = X[train_idxs] @ coef
                    Xtest = X[test_idxs] @ coef


                    try:
                        r2_pos, r2_vel, r2_acc, decoder_obj = lr_decoder(Xtest, Xtrain, Ytest, Ytrain, **da)
                    except:
                        pdb.set_trace()
                    pca_r2[i, k, f, d] = r2_pos

    #fig, ax = plt.subplots(4, 2, figsize=(8, 16))
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    
    fca_mean = np.mean(np.mean(r2[:, 0, :, :], axis=1), axis=0)

    # Move the fold indices up and then reshape to calc std
    fca_std = np.std(r2[:, 0, :, :].reshape((-1, 30)), axis=0)/np.sqrt(40)

    pca_mean = np.mean(np.mean(pca_r2[:, 0, :, :], axis=1), axis=0)
    pca_std = np.std(pca_r2[:, 0, :, :].reshape((-1, 30)), axis=0)/np.sqrt(40)

    ax.fill_between(np.arange(1, 31), fca_mean - fca_std, fca_mean + fca_std, color='r', alpha=0.25)
    ax.fill_between(np.arange(1, 31), pca_mean - pca_std, pca_mean + pca_std, color='k', alpha=0.25)

    ax.plot(np.arange(1, 31), fca_mean, color='r')
    ax.plot(np.arange(1, 31), pca_mean, color='k')

    ax.legend(['FCCA', 'PCA'], loc='lower right', fontsize=14)
    ax.set_title('Rat Hippocampus', fontsize=14)
    ax.set_xlabel('Dimension', fontsize=14)
    ax.set_ylabel('Position Decoding ' + r'$r^2$', fontsize=14)    
    ax.tick_params(axis='both', labelsize=12)

    fig.tight_layout()
    fig.savefig('%s/peanut_decoding.pdf' % figpath, bbox_inches='tight', pad_inches=0)
    # fig.savefig('peanut_decoding.pdf', bbox_inches='tight', pad_inches=0)

    ########## S1 ####################################################################
    with open('/mnt/Secondary/data/postprocessed/loco_decoding_df.dat', 'rb') as f:
        result_list = pickle.load(f)
    with open('/mnt/Secondary/data/postprocessed/indy_S1_df.dat', 'rb') as f:
        rl2 = pickle.load(f)

    loco_df = pd.DataFrame(result_list)
    indy_df = pd.DataFrame(rl2)

    sabes_df = pd.concat([loco_df, indy_df])

    
