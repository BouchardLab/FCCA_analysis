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
from utils import apply_df_filters, calc_loadings
from dstableFGM import dstable_descent

if __name__ == '__main__':

    # Where to save?
    if len(sys.argv) > 1:
        figpath = sys.argv[1]
    else:
        figpath = '/home/akumar/nse/neural_control/figs/cosyne23'

    # Open indy, loco, peanut, and cv dataframes 
    with open('/mnt/Secondary/data/postprocessed/indy_decoding_df2.dat', 'rb') as f:
        rl = pickle.load(f)
    indy_df = pd.DataFrame(rl)

    with open('/mnt/Secondary/data/postprocessed/loco_decoding_df.dat', 'rb') as f:
        loco_df = pickle.load(f)
    loco_df = pd.DataFrame(loco_df)
    loco_df = apply_df_filters(loco_df,
                            loader_args={'bin_width': 50, 'filter_fn': 'none', 'filter_kwargs': {}, 'boxcox': 0.5, 'spike_threshold': 100, 'region': 'M1'},
                            decoder_args={'trainlag': 4, 'testlag': 4, 'decoding_window': 5})
    good_loco_files = ['loco_20170210_03.mat',
        'loco_20170213_02.mat',
        'loco_20170215_02.mat',
        'loco_20170227_04.mat',
        'loco_20170228_02.mat',
        'loco_20170301_05.mat',
        'loco_20170302_02.mat']

    loco_df = apply_df_filters(loco_df, data_file=good_loco_files)        
    sabes_df = pd.concat([indy_df, loco_df])
    with open('/mnt/Secondary/data/postprocessed/peanut_decoding_df.dat', 'rb') as f:
        peanut_df = pickle.load(f)
    peanut_df = pd.DataFrame(peanut_df)

    # Choose one of the decoding args
    peanut_df = apply_df_filters(peanut_df, decoder_args=peanut_df.iloc[0]['decoder_args'])

    with open('/mnt/Secondary/data/postprocessed/cv_dimreduc_df.dat', 'rb') as f:
        cv_df = pickle.load(f)
    cv_df = pd.DataFrame(cv_df)

    dim = 6
    data_files = np.unique(sabes_df['data_file'].values)
    folds = np.arange(5)
    sabes_ss_angles = np.zeros((len(data_files), folds.size, dim))
    sabes_lc = np.zeros((len(data_files), folds.size))

    for i, data_file in enumerate(data_files):
        for f, fold in enumerate(folds):

            dfpca = apply_df_filters(sabes_df, data_file=data_file, dimreduc_method='PCA', fold_idx=fold, dim=dim)
            dffcca = apply_df_filters(sabes_df, data_file=data_file, dimreduc_method='LQGCA', fold_idx=fold, dim=dim)

            try:
                assert(dfpca.shape[0] == 1)
            except:
                pdb.set_trace()

            try:
                assert(dffcca.shape[0] == 1)
            except:
                pdb.set_trace()

            sabes_ss_angles[i, f, :] = scipy.linalg.subspace_angles(dfpca.iloc[0]['coef'][:, 0:dim], dffcca.iloc[0]['coef'])
            sabes_lc[i, f] = scipy.stats.spearmanr(calc_loadings(dfpca.iloc[0]['coef'][:, 0:dim]), calc_loadings(dffcca.iloc[0]['coef']))[0]

    peanut_ss_angles = np.zeros((8, 5, dim))
    peanut_lc = np.zeros((8, 5))
    epochs = np.unique(peanut_df['epoch'].values)
    folds = np.arange(5)

    for i, epoch in enumerate(epochs):
        for f, fold in enumerate(folds):

            dfpca = apply_df_filters(peanut_df, epoch=epoch, dimreduc_method='PCA', fold_idx=fold, dim=dim)
            dffcca = apply_df_filters(peanut_df, epoch=epoch, dimreduc_method='LQGCA', fold_idx=fold, dim=dim)

            try:
                assert(dfpca.shape[0] == 1)
            except:
                pdb.set_trace()

            try:
                assert(dffcca.shape[0] == 1)
            except:
                pdb.set_trace()

            peanut_ss_angles[i, f, :] = scipy.linalg.subspace_angles(dfpca.iloc[0]['coef'][:, 0:dim], dffcca.iloc[0]['coef'])
            peanut_lc[i, f] = scipy.stats.spearmanr(calc_loadings(dfpca.iloc[0]['coef'][:, 0:dim]), calc_loadings(dffcca.iloc[0]['coef']))[0]

    data_files = np.unique(cv_df['data_file'].values)
    cv_ss_angles = np.zeros((len(data_files), 5, dim))
    cv_lc = np.zeros((len(data_files), 5))

    for i, data_file in enumerate(data_files):
        for f, fold in enumerate(folds):
            dfpca = apply_df_filters(cv_df, dimreduc_method='PCA', fold_idx=fold, dim=dim, data_file=data_file)
            dffcca = apply_df_filters(cv_df, dimreduc_method='LQGCA', dimreduc_args={'T':40, 'loss_type':'trace', 'n_init':5}, fold_idx=fold, dim=dim, data_file=data_file)

            try:
                assert(dfpca.shape[0] == 1)
            except:
                pdb.set_trace()

            try:
                assert(dffcca.shape[0] == 1)
            except:
                pdb.set_trace()

            cv_ss_angles[i, f, :] = scipy.linalg.subspace_angles(dfpca.iloc[0]['coef'][:, 0:dim], dffcca.iloc[0]['coef'])
            cv_lc[i, f] = scipy.stats.spearmanr(calc_loadings(dfpca.iloc[0]['coef'][:, 0:dim]), calc_loadings(dffcca.iloc[0]['coef']))[0]


    fig, ax = plt.subplots(1, 2, figsize=(4, 4))
    bplot1 = ax[0].boxplot([np.mean(sabes_ss_angles, axis=-1).ravel(), np.mean(peanut_ss_angles, axis=-1).ravel(), np.mean(cv_ss_angles, axis=-1).ravel()], 
                    showfliers=False, patch_artist=True, notch=True)
    bplot2 = ax[1].boxplot([sabes_lc.ravel(), peanut_lc.ravel(), cv_lc.ravel()], showfliers=False, patch_artist=True, notch=True)

    colors = ['blue', '#e85b15']
    for i, bplot in enumerate((bplot1, bplot2)):
        for patch in bplot['boxes']:
            patch.set_facecolor(colors[i])
            patch.set_alpha(0.5)
        for median in bplot['medians']:
            median.set_color('black')

    # ax.hist(np.mean(ss_angles, axis=-1).ravel(), color='r', alpha=0.5, linewidth=1, edgecolor='k')
    ax[0].set_ylim([0, np.pi/2])
    # ax.tick_params(axis='both', labelsize=12)
    ax[0].set_ylabel('Mean Subspace Angle (rads)', fontsize=14)
    ax[0].set_xticklabels(['M1', 'HPC', 'vSMC'], fontsize=11)
    ax[1].set_xticklabels(['M1', 'HPC', 'vSMC'], fontsize=11)

    ax[1].set_ylabel('Leverage Score Spearman-r', fontsize=14)
    ax[1].yaxis.set_label_position("right")
    ax[1].yaxis.tick_right()
    print((np.mean(sabes_lc.ravel()), np.mean(peanut_lc.ravel()), np.mean(cv_lc.ravel())))
    fig.subplots_adjust(wspace=0, hspace=0)
    #fig.savefig('%s/sslc.pdf' % figpath, bbox_inches='tight', pad_inches=0)
