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

    M1 = False
    S1 = False
    HPC = True 

    # Where to save?
    if len(sys.argv) > 1:
        figpath = sys.argv[1]
    else:
        figpath = '/home/akumar/nse/neural_control/figs/loco_indy_merge'

    if M1:
        with open('/mnt/Secondary/data/postprocessed/indy_decoding_df2.dat', 'rb') as f:
            rl = pickle.load(f)
        indy_df = pd.DataFrame(rl)

        # PCA results for indy should be taken from an older dataframe because indy_decoding_df2 
        # erroneously did not restrict the PCA dimension prior to doing decoding
        # Grab PCA results
        with open('/mnt/Secondary/data/postprocessed/sabes_kca_decodign_df.dat', 'rb') as f:
            pca_decoding_df = pickle.load(f)

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

        indy_data_files = np.unique(indy_df['data_file'].values)
        loco_data_files = np.unique(loco_df['data_file'].values)

        dims = np.unique(indy_df['dim'].values)
        r2fc = np.zeros((len(indy_data_files) + len(loco_data_files), dims.size, 5))
        sr2_vel_pca = np.zeros((len(indy_data_files) + len(loco_data_files), dims.size, 5))

        for i, data_file in tqdm(enumerate(indy_data_files)):
            for j, dim in enumerate(dims):               
                for f in range(5):
                    dim_fold_df = apply_df_filters(indy_df, data_file=data_file, dim=dim, fold_idx=f, dimreduc_method='LQGCA')
                    assert(dim_fold_df.shape[0] == 1)
                    r2fc[i, j, f] = dim_fold_df.iloc[0]['r2'][1]

                    pca_df = apply_df_filters(pca_decoding_df, data_file=data_file, dim=dim, fold_idx=f, dr_method='PCA')
                    assert(pca_df.shape[0] == 1)
                    sr2_vel_pca[i, j, f] = pca_df.iloc[0]['r2'][1]

        for i, data_file in tqdm(enumerate(loco_data_files)):
            for j, dim in enumerate(dims):               
                for f in range(5):
                    dim_fold_df = apply_df_filters(loco_df, data_file=data_file, dim=dim, fold_idx=f, dimreduc_method='LQGCA')
                    assert(dim_fold_df.shape[0] == 1)
                    r2fc[i + len(indy_data_files), j, f] = dim_fold_df.iloc[0]['r2'][1]

                    pca_df = apply_df_filters(loco_df, data_file=data_file, dim=dim, fold_idx=f, dimreduc_method='PCA')
                    assert(pca_df.shape[0] == 1)
                    sr2_vel_pca[i + len(indy_data_files), j, f] = pca_df.iloc[0]['r2'][1]

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))

        # Average across folds and plot
        # REINSERT OLS(5) IN HERE IF NEEDED

        colors = ['black', 'red', '#781820', '#5563fa']
        dim_vals = dims

        # # DCA averaged over folds
        # dca_r2 = np.mean(r2[:, :, 1, :, 1], axis=2)
        # # KCA averaged over folds
        # kca_r2 = np.mean(r2[:, :, 2, :, 1], axis=2)

        # FCCA averaged over folds
        fca_r2 = np.mean(r2fc[:, :, :], axis=2)
        # PCA
        pca_r2 = np.mean(sr2_vel_pca, axis=2)
        # ax.fill_between(dim_vals, np.mean(dca_r2, axis=0) + np.std(dca_r2, axis=0)/np.sqrt(35),
        #                 np.mean(dca_r2, axis=0) - np.std(dca_r2, axis=0)/np.sqrt(35), color=colors[0], alpha=0.25)
        # ax.plot(dim_vals, np.mean(dca_r2, axis=0), color=colors[0])
        # ax.fill_between(dim_vals, np.mean(kca_r2, axis=0) + np.std(kca_r2, axis=0)/np.sqrt(35),
        #                 np.mean(kca_r2, axis=0) - np.std(kca_r2, axis=0)/np.sqrt(35), color=colors[1], alpha=0.25)
        # ax.plot(dim_vals, np.mean(kca_r2, axis=0), color=colors[1])
        ax.fill_between(dim_vals, np.mean(fca_r2, axis=0) + np.std(fca_r2, axis=0)/np.sqrt(35),
                        np.mean(fca_r2, axis=0) - np.std(fca_r2, axis=0)/np.sqrt(35), color=colors[1], alpha=0.25)
        ax.plot(dim_vals, np.mean(fca_r2, axis=0), color=colors[1])

        ax.fill_between(dim_vals, np.mean(pca_r2, axis=0) + np.std(pca_r2, axis=0)/np.sqrt(35),
                        np.mean(pca_r2, axis=0) - np.std(pca_r2, axis=0)/np.sqrt(35), color=colors[0], alpha=0.25)
        ax.plot(dim_vals, np.mean(pca_r2, axis=0), color=colors[0])

        # Plot the paired differences

        # ax.plot(dim_vals, )
        # ax.fill_between(dim_vals, np.mean(fca_r2 - pca_r2, axis=0) + np.std(fca_r2 - pca_r2, axis=0)/np.sqrt(35),
        #                 np.mean(fca_r2 - pca_r2, axis=0) - np.std(fca_r2 - pca_r2, axis=0)/np.sqrt(35), color=colors[0], alpha=0.25)

        ax.set_xlabel('Dimension', fontsize=18)
        ax.set_ylabel('Velocity Prediction ' + r'$r^2$', fontsize=18)
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
        ax.set_xlim([1, 30])
        ax.set_xticks([1, 15, 30])
        ax.set_yticks([0., 0.2, 0.4])

        #ax.legend(['FCCA', 'PCA'], fontsize=10, bbox_to_anchor=(0.32, 1.01), frameon=False)
        ax.legend(['FBC', 'FFC'], fontsize=10, loc='upper left', frameon=False)

        #ax.legend(['FCCA', 'PCA'], fontsize=14, loc='lower right', frameon=False)
        
        #ax.set_title('Macaque M1', fontsize=18)

        # Inset that shows the paired differences
        #axin = ax.inset_axes([0.125, -0.924, 0.75, 0.75])
        axin = ax.inset_axes([0.6, 0.1, 0.35, 0.35])

        pca_auc = np.sum(pca_r2, axis=1)
        fca_auc = np.sum(fca_r2, axis=1)

        # Run a signed rank test
        _, p = scipy.stats.wilcoxon(pca_auc, fca_auc, alternative='less')
        print(p)

        axin.scatter(np.zeros(35), pca_auc, color='k', alpha=0.75, s=3)
        axin.scatter(np.ones(35), fca_auc, color='r', alpha=0.75, s=3)
        axin.plot(np.array([(0, 1) for _ in range(pca_r2.shape[0])]).T, np.array([(y1, y2) for y1, y2 in zip(np.sum(pca_r2, axis=1), np.sum(fca_r2, axis=1))]).T, color='k', alpha=0.5)
        axin.set_yticks([])
        axin.set_ylabel('Decoding AUC', fontsize=10)
        axin.set_xlim([-0.5, 1.5])
        axin.set_xticks([0, 1])
        axin.set_xticklabels(['FFC', 'FBC'], fontsize=10)
        axin.set_title('****')
        #fig.tight_layout()
        fig.savefig('%s/indy_vel_decoding.pdf' % figpath, bbox_inches='tight', pad_inches=0)

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        # Plot of the differences across dimensions
        ax.fill_between(dim_vals, np.mean(fca_r2 - pca_r2, axis=0) + np.std(fca_r2 - pca_r2, axis=0)/np.sqrt(35),
                        np.mean(fca_r2 - pca_r2, axis=0) - np.std(fca_r2 - pca_r2, axis=0)/np.sqrt(35), color='blue', alpha=0.25)
        ax.plot(dim_vals, np.mean(fca_r2 - pca_r2, axis=0), color='blue')

        max_delta = np.max(np.mean(fca_r2 - pca_r2, axis=0))
        fractional_delta = max_delta/np.mean(pca_r2, axis=0)[np.argmax(np.mean(fca_r2 - pca_r2, axis=0))]

        print('M1 peak fractional improvement:%f' % fractional_delta)

        ax.set_xlabel('Dimension', fontsize=18)
        ax.set_ylabel(r'$\Delta$' + ' Velocity Prediction ' + r'$r^2$', fontsize=18)
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)

        ax.vlines(6, 0, np.mean(fca_r2 - pca_r2, axis=0)[5], linestyles='dashed', color='blue')
        ax.hlines(np.mean(fca_r2 - pca_r2, axis=0)[5], 0, 6, linestyles='dashed', color='blue')
        ax.set_xlim([1, 30])
        ax.set_xticks([1, 6, 15, 30])
        ax.set_yticks([0., 0.12])
        ax.set_ylim([0, 0.125])

        fig.savefig('%s/indy_vel_decoding_delta.pdf' % figpath, bbox_inches='tight', pad_inches=0)


    if S1:
        with open('/mnt/Secondary/data/postprocessed/loco_lag1_decodingdf.dat', 'rb') as f:
            result_list = pickle.load(f)
        with open('/mnt/Secondary/data/postprocessed/indy_S1_decodingdf.dat', 'rb') as f:
            rl2 = pickle.load(f)

        loco_df = pd.DataFrame(result_list)
        # filter by good files
        good_loco_files = ['loco_20170210_03.mat',
        'loco_20170213_02.mat',
        'loco_20170215_02.mat',
        'loco_20170227_04.mat',
        'loco_20170228_02.mat',
        'loco_20170301_05.mat',
        'loco_20170302_02.mat']

        loco_df = apply_df_filters(loco_df, data_file=good_loco_files, 
                                   loader_args={'bin_width': 50, 'filter_fn': 'none', 'filter_kwargs': {}, 
                                   'boxcox': 0.5, 'spike_threshold': 100, 'region': 'S1'})

        indy_df = pd.DataFrame(rl2)        

        sabes_df = pd.concat([loco_df, indy_df])

        data_files = np.unique(sabes_df['data_file'].values)
        dims = np.unique(sabes_df['dim'].values)
        r2fc = np.zeros((len(data_files), dims.size, 5))
        r2pca = np.zeros((len(data_files), dims.size, 5))

        for i, data_file in tqdm(enumerate(data_files)):
            for j, dim in enumerate(dims):               
                for f in range(5):
                    dim_fold_df = apply_df_filters(sabes_df, data_file=data_file, dim=dim, fold_idx=f, dimreduc_method='LQGCA')
                    # Trace loss
                    try:
                        assert(dim_fold_df.shape[0] == 1)
                    except:
                        pdb.set_trace()
                    r2fc[i, j, f] = dim_fold_df.iloc[0]['r2'][1]

                    dim_fold_df = apply_df_filters(sabes_df, data_file=data_file, dim=dim, fold_idx=f, dimreduc_method='PCA')
                    # Trace loss
                    try:
                        assert(dim_fold_df.shape[0] == 1)
                    except:
                        pdb.set_trace()
                    r2pca[i, j, f] = dim_fold_df.iloc[0]['r2'][1]        

        # FCCA averaged over folds
        fca_r2 = np.mean(r2fc, axis=2)
        # PCA
        pca_r2 = np.mean(r2pca, axis=2)
        # ax.fill_between(dim_vals, np.mean(dca_r2, axis=0) + np.std(dca_r2, axis=0)/np.sqrt(35),
        #                 np.mean(dca_r2, axis=0) - np.std(dca_r2, axis=0)/np.sqrt(35), color=colors[0], alpha=0.25)
        # ax.plot(dim_vals, np.mean(dca_r2, axis=0), color=colors[0])
        # ax.fill_between(dim_vals, np.mean(kca_r2, axis=0) + np.std(kca_r2, axis=0)/np.sqrt(35),
        #                 np.mean(kca_r2, axis=0) - np.std(kca_r2, axis=0)/np.sqrt(35), color=colors[1], alpha=0.25)
        # ax.plot(dim_vals, np.mean(kca_r2, axis=0), color=colors[1])
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        colors = ['black', 'red', '#781820', '#5563fa']
        dim_vals = dims


        ax.fill_between(dim_vals, np.mean(fca_r2, axis=0) + np.std(fca_r2, axis=0)/np.sqrt(8),
                        np.mean(fca_r2, axis=0) - np.std(fca_r2, axis=0)/np.sqrt(8), color=colors[1], alpha=0.25)
        ax.plot(dim_vals, np.mean(fca_r2, axis=0), color=colors[1])

        ax.fill_between(dim_vals, np.mean(pca_r2, axis=0) + np.std(pca_r2, axis=0)/np.sqrt(8),
                        np.mean(pca_r2, axis=0) - np.std(pca_r2, axis=0)/np.sqrt(8), color=colors[0], alpha=0.25)
        ax.plot(dim_vals, np.mean(pca_r2, axis=0), color=colors[0])

        # Plot the paired differences

        # ax.plot(dim_vals, )
        # ax.fill_between(dim_vals, np.mean(fca_r2 - pca_r2, axis=0) + np.std(fca_r2 - pca_r2, axis=0)/np.sqrt(35),
        #                 np.mean(fca_r2 - pca_r2, axis=0) - np.std(fca_r2 - pca_r2, axis=0)/np.sqrt(35), color=colors[0], alpha=0.25)

        ax.set_xlabel('Dimension', fontsize=18)
        ax.set_ylabel('Velocity Prediction ' + r'$r^2$', fontsize=18)
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)

        ax.legend(['FBC', 'FFC'], fontsize=10, bbox_to_anchor=(0.32, 1.01), frameon=False)
        ax.set_yticks([0., 0.25])
        ax.set_xticks([1., 15, 30])
        #ax.legend(['FCCA', 'PCA'], fontsize=14, loc='lower right', frameon=False)
        
        #ax.set_title('Macaque S1', fontsize=18)

        # Inset that shows the paired differences
        #axin = ax.inset_axes([0.125, -0.924, 0.75, 0.75])
        axin = ax.inset_axes([0.8, 0.1, 0.35, 0.35])

        pca_auc = np.sum(pca_r2, axis=1)
        fca_auc = np.sum(fca_r2, axis=1)

        # Run a signed rank test
        _, p = scipy.stats.wilcoxon(pca_auc, fca_auc, alternative='less')
        print(p)

        axin.scatter(np.zeros(8), pca_auc, color='k', alpha=0.75, s=3)
        axin.scatter(np.ones(8), fca_auc, color='r', alpha=0.75, s=3)
        axin.plot(np.array([(0, 1) for _ in range(pca_r2.shape[0])]).T, np.array([(y1, y2) for y1, y2 in zip(np.sum(pca_r2, axis=1), np.sum(fca_r2, axis=1))]).T, color='k', alpha=0.5)
        axin.set_yticks([])
        axin.set_ylabel('Decoding AUC', fontsize=10)
        axin.set_xlim([-0.5, 1.5])
        axin.set_xticks([0, 1])
        axin.set_xticklabels(['FFC', 'FBC'], fontsize=10)
        axin.set_title('***')
        #fig.tight_layout()
        fig.savefig('%s/S1_vel_decoding.pdf' % figpath, bbox_inches='tight', pad_inches=0)

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        # Plot of the differences across dimensions
        ax.fill_between(dim_vals, np.mean(fca_r2 - pca_r2, axis=0) + np.std(fca_r2 - pca_r2, axis=0)/np.sqrt(8),
                        np.mean(fca_r2 - pca_r2, axis=0) - np.std(fca_r2 - pca_r2, axis=0)/np.sqrt(8), color='blue', alpha=0.25)
        ax.plot(dim_vals, np.mean(fca_r2 - pca_r2, axis=0), color='blue')

        max_delta = np.max(np.mean(fca_r2 - pca_r2, axis=0))
        fractional_delta = max_delta/np.mean(pca_r2, axis=0)[np.argmax(np.mean(fca_r2 - pca_r2, axis=0))]

        print('S1 peak fractional improvement:%f' % fractional_delta)

        ax.set_xlabel('Dimension', fontsize=18)
        ax.set_ylabel(r'$\Delta$' + ' Velocity Prediction ' + r'$r^2$', fontsize=18)
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)

        ax.set_yticks([0., 0.1])
        ax.set_ylim([0., 0.11])
        ax.set_xticks([1, 6, 15, 30])

        ax.vlines(6, 0, np.mean(fca_r2 - pca_r2, axis=0)[5], linestyles='dashed', color='blue')
        ax.hlines(np.mean(fca_r2 - pca_r2, axis=0)[5], 0, 6, linestyles='dashed', color='blue')
        ax.set_xlim([0, 30])
        fig.savefig('%s/S1_vel_decoding_delta.pdf' % figpath, bbox_inches='tight', pad_inches=0)

    # fig.savefig('/home/akumar/pCloudDrive/Documents/tex/Cosyne23/fig2.pdf')

    ################################## Old code for peanut ############################################################

    if HPC:
        with open('/mnt/Secondary/data/postprocessed/peanut_decoding_df.dat', 'rb') as f:
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

                        r2_pos, r2_vel, r2_acc, decoder_obj, _, _, _ = lr_decoder(Xtest, Xtrain, Ytest, Ytrain, **da)

                        pca_r2[i, k, f, d] = r2_pos

        #fig, ax = plt.subplots(4, 2, figsize=(8, 16))
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))

        # Average over folds
        fca_r2 = np.mean(r2[:, 0, :, :], axis=1)
        pca_r2 = np.mean(pca_r2[:, 0, :, :], axis=1)

        fca_mean = np.mean(fca_r2, axis=0)

        # Move the fold indices up and then reshape to calc std
        fca_std = np.std(fca_r2, axis=0)/np.sqrt(8)

        pca_mean = np.mean(pca_r2, axis=0)
        pca_std = np.std(pca_r2, axis=0)/np.sqrt(8)

        ax.fill_between(np.arange(1, 31), fca_mean - fca_std, fca_mean + fca_std, color='r', alpha=0.25)
        ax.fill_between(np.arange(1, 31), pca_mean - pca_std, pca_mean + pca_std, color='k', alpha=0.25)

        ax.plot(np.arange(1, 31), fca_mean, color='r')
        ax.plot(np.arange(1, 31), pca_mean, color='k')

        ax.legend(['FCCA', 'PCA'], fontsize=10, bbox_to_anchor=(0.32, 1.01), frameon=False)

        # # Inset that shows the paired differences
        axin = ax.inset_axes([0.6, 0.1, 0.35, 0.35])
        #axin = ax.inset_axes([0.125, -0.924, 0.75, 0.75])

        pca_auc = np.sum(pca_r2, axis=1)
        fca_auc = np.sum(fca_r2, axis=1)

        # Run a signed rank test
        _, p = scipy.stats.wilcoxon(pca_auc, fca_auc, alternative='less')
        print(p)

        axin.scatter(np.zeros(8), pca_auc, color='k', alpha=0.75, s=3)
        axin.scatter(np.ones(8), fca_auc, color='r', alpha=0.75, s=3)
        axin.plot(np.array([(0, 1) for _ in range(pca_r2.shape[0])]).T, np.array([(y1, y2) for y1, y2 in zip(np.sum(pca_r2, axis=1), np.sum(fca_r2, axis=1))]).T, color='k', alpha=0.5)
        axin.set_yticks([])
        axin.set_ylabel('Decoding AUC', fontsize=10)
        axin.set_xlim([-0.5, 1.5])
        axin.set_xticks([0, 1])
        axin.set_xticklabels(['PCA', 'FCCA'], fontsize=10)
        axin.set_title('**')
        ax.set_title('Rat Hippocampus', fontsize=18)
        ax.set_xlabel('Dimension', fontsize=18)
        ax.set_ylabel('Position Prediction ' + r'$r^2$', fontsize=18)    
        ax.tick_params(axis='both', labelsize=16)

        #fig.tight_layout()
        fig.savefig('%s/peanut_decoding.pdf' % figpath, bbox_inches='tight', pad_inches=0)

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        # Plot of the differences across dimensions
        dim_vals = np.arange(1, 31)
        ax.fill_between(dim_vals, np.mean(fca_r2 - pca_r2, axis=0) + np.std(fca_r2 - pca_r2, axis=0)/np.sqrt(35),
                        np.mean(fca_r2 - pca_r2, axis=0) - np.std(fca_r2 - pca_r2, axis=0)/np.sqrt(35), color='blue', alpha=0.25)
        ax.plot(dim_vals, np.mean(fca_r2 - pca_r2, axis=0), color='blue')

        max_delta = np.max(np.mean(fca_r2 - pca_r2, axis=0))
        fractional_delta = max_delta/np.mean(pca_r2, axis=0)[np.argmax(np.mean(fca_r2 - pca_r2, axis=0))]

        print('HPC peak fractional improvement:%f' % fractional_delta)

        import matplotlib.ticker as tick

        d = np.mean(fca_r2 - pca_r2, axis=0)
        ax.set_xlabel('Dimension', fontsize=18)
        ax.set_ylabel(r'$\Delta$' + ' Position Prediction ' + r'$r^2$', fontsize=18)
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
        ax.vlines(dim_vals[np.argmax(np.mean(fca_r2 - pca_r2, axis=0))], 0, np.max(np.mean(fca_r2 - pca_r2, axis=0)), linestyles='dashed', color='blue')
        ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
        fig.savefig('%s/peanut_decoding_delta.pdf' % figpath, bbox_inches='tight', pad_inches=0)

