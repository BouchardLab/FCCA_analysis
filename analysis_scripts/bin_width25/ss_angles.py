import numpy as np
import scipy
import sys
import pdb
import matplotlib.pyplot as plt
from glob import glob
import pickle
from pyuoi.linear_model.var import VAR
from tqdm import tqdm
import pandas as pd
from neurosim.models.var import VAR as VARss
from neurosim.models.var import form_companion
from copy import deepcopy
from sklearn.model_selection import KFold

from pseudopy import Normal

sys.path.append('/home/akumar/nse/neural_control')
from loaders import load_sabes, load_peanut, load_cv
from subspaces import estimate_autocorrelation
from utils import apply_df_filters
from dstableFGM import dstable_descent

if __name__ == '__main__':

    # Where to save?
    if len(sys.argv) > 1:
        figpath = sys.argv[1]
    else:
        figpath = '/home/akumar/nse/neural_control/figs/final'

    dframe = '/home/akumar/nse/neural_control/data/indy_decoding_marginal.dat'
    print('Using dataframe %s' % dframe)
    with open(dframe, 'rb') as f:
        sabes_df = pickle.load(f)
    sabes_df = pd.DataFrame(sabes_df)

    dim = 10
    ss_angles = np.zeros((28, 5, dim))
    data_files = np.unique(sabes_df['data_file'].values)
    folds = np.arange(5)
    dimreduc_methods = dimreduc_methods = ['PCA', 'LQGCA']
    LQGCA_dimreduc_args = [{'T':3, 'loss_type':'trace', 'n_init':10}]
    dimvals = np.unique(sabes_df['dim'].values)

    # Pick one
    decoder_arg = sabes_df.iloc[0]['decoder_args']
    df = apply_df_filters(sabes_df, decoder_args=decoder_arg)

    for i, data_file in enumerate(data_files):
        for f, fold in enumerate(folds):

            dfpca = apply_df_filters(df, data_file=data_file, dimreduc_method='PCA', fold_idx=fold, dim=dim)
            dffcca = apply_df_filters(df, data_file=data_file, dimreduc_method='LQGCA', dimreduc_args=LQGCA_dimreduc_args[0], fold_idx=fold, dim=dim)

            try:
                assert(dfpca.shape[0] == 1)
            except:
                pdb.set_trace()

            assert(dffcca.shape[0] == 1)
            
            ss_angles[i, f, :] = scipy.linalg.subspace_angles(dfpca.iloc[0]['coef'][:, 0:dim], dffcca.iloc[0]['coef'])

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.hist(np.mean(ss_angles, axis=-1).ravel(), color='r', alpha=0.5, linewidth=1, edgecolor='k')
    ax.set_xlim([0, np.pi/2])
    ax.tick_params(axis='both', labelsize=12)
    ax.set_ylabel('Count', fontsize=14)
    ax.set_xlabel('FCCA/PCA avg. subspace angle (rads)', fontsize=14)
    ax.set_xticks([0, np.pi/8, np.pi/4, 3 * np.pi/8, np.pi/2])
    ax.set_xticklabels(['0', r'$\pi/8$', r'$\pi/4$', r'$3\pi/8$', r'$\pi/2$'])
    fig.savefig('%s/m1_ssangles.pdf' % figpath, bbox_inches='tight', pad_inches=0)


    # with open('/home/akumar/nse/neural_control/data/peanut_decoding_df.dat', 'rb') as f:
    #     peanut_df = pickle.load(f)
    # peanut_df = pd.DataFrame(peanut_df)

    # dim = 10

    # ss_angles = np.zeros((8, 5, dim))

    # epochs = np.unique(peanut_df['epoch'].values)
    # folds = np.arange(5)
    # dimreduc_methods = ['PCA', 'LQGCA']
    # LQGCA_dimreduc_args = [{'T':3, 'loss_type':'trace', 'n_init':5}]

    # # Pick one
    # decoder_arg = peanut_df.iloc[0]['decoder_args']
    # df = apply_df_filters(peanut_df, decoder_args=decoder_arg)

    # for i, epoch in enumerate(epochs):
    #     for f, fold in enumerate(folds):
    #         dfpca = apply_df_filters(df, epoch=epoch, dimreduc_method='PCA', fold_idx=fold, dim=dim)
    #         dffcca = apply_df_filters(df, epoch=epoch, dimreduc_method='LQGCA', dimreduc_args=LQGCA_dimreduc_args[0], fold_idx=fold, dim=dim)

    #         assert(dfpca.shape[0] == 1)
    #         assert(dffcca.shape[0] == 1)
            
    #         ss_angles[i, f, :] = scipy.linalg.subspace_angles(dfpca.iloc[0]['coef'][:, 0:dim], dffcca.iloc[0]['coef'])

    # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    # ax.hist(np.mean(ss_angles, axis=-1).ravel(), color='r', alpha=0.5, linewidth=1, edgecolor='k')
    # ax.set_xlim([0, np.pi/2])
    # ax.tick_params(axis='both', labelsize=12)
    # ax.set_ylabel('Count', fontsize=14)
    # ax.set_xlabel('FCCA/PCA avg. subspace angle (rads)', fontsize=14)
    # ax.set_xticks([0, np.pi/8, np.pi/4, 3 * np.pi/8, np.pi/2])
    # ax.set_xticklabels(['0', r'$\pi/8$', r'$\pi/4$', r'$3\pi/8$', r'$\pi/2$'])
    # fig.savefig('/home/akumar/nse/neural_control/figs/final/hpc_ssangles.pdf', bbox_inches='tight', pad_inches=0)