import pdb
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import scipy 
import time
import glob
import pickle
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

from scipy.ndimage import gaussian_filter1d
from mpl_toolkits.axisartist.axislines import AxesZero

from dca.methods_comparison import JPCA
from pyuoi.linear_model.var  import VAR
from neurosim.models.var import form_companion

sys.path.append('/home/akumar/nse/neural_control')
from utils import apply_df_filters, calc_loadings, calc_cascaded_loadings
from loaders import load_sabes
from decoders import lr_decoder
from segmentation import reach_segment_sabes, measure_straight_dev

start_times = {'indy_20160426_01': 0,
               'indy_20160622_01':1700,
               'indy_20160624_03': 500,
               'indy_20160627_01': 0,
               'indy_20160630_01': 0,
               'indy_20160915_01': 0,
               'indy_20160921_01': 0,
               'indy_20160930_02': 0,
               'indy_20160930_05': 300,
               'indy_20161005_06': 0,
               'indy_20161006_02': 350,
               'indy_20161007_02': 950,
               'indy_20161011_03': 0,
               'indy_20161013_03': 0,
               'indy_20161014_04': 0,
               'indy_20161017_02': 0,
               'indy_20161024_03': 0,
               'indy_20161025_04': 0,
               'indy_20161026_03': 0,
               'indy_20161027_03': 500,
               'indy_20161206_02': 5500,
               'indy_20161207_02': 0,
               'indy_20161212_02': 0,
               'indy_20161220_02': 0,
               'indy_20170123_02': 0,
               'indy_20170124_01': 0,
               'indy_20170127_03': 0,
               'indy_20170131_02': 0,
               }

if __name__ == '__main__':

    # Where to save?
    if len(sys.argv) > 1:
        figpath = sys.argv[1]
    else:
        figpath = '/home/akumar/nse/neural_control/figs/marginal'

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))

    ################################ Decoding comparison #####################################
    # co-plot marginal decoding with the usual decoding

    # # Sequentially do indy, peanut decoding
    with open('/home/akumar/nse/neural_control/data/indy_decoding_marginal.dat', 'rb') as f:
        rl = pickle.load(f)
    marginal_df = pd.DataFrame(rl)
    
    with open('/home/akumar/nse/neural_control/data/indy_decoding_df2.dat', 'rb') as f:
        rl = pickle.load(f)
    sabes_df = pd.DataFrame(rl)
    #sabes_df = apply_df_filters(sabes_df, dimreduc_method='LQGCA')

    # Grab PCA results
    with open('/home/akumar/nse/neural_control/data/sabes_kca_decodign_df.dat', 'rb') as f:
        pca_decoding_df = pickle.load(f)

    data_files = np.unique(sabes_df['data_file'].values)
    dims = np.unique(sabes_df['dim'].values)
    r2fc = np.zeros((len(data_files), dims.size, 5, 3))
    # marginal r^2
    r2_marginal = np.zeros((len(data_files), dims.size, 5, 2, 3))

    for i, data_file in tqdm(enumerate(data_files)):
        for j, dim in enumerate(dims):               
            for f in range(5):
                dim_fold_df = apply_df_filters(sabes_df, data_file=data_file, dim=dim, fold_idx=f, dimreduc_method='LQGCA')
                # Trace loss
                try:
                    assert(dim_fold_df.shape[0] == 1)
                except:
                    pdb.set_trace()
                r2fc[i, j, f, :] = dim_fold_df.iloc[0]['r2']

                dim_fold_marginal_df = apply_df_filters(marginal_df, data_file=data_file, dim=dim, fold_idx=f, dimreduc_method='LQGCA')
                try:
                    assert(dim_fold_marginal_df.shape[0] == 1)
                except:
                    pdb.set_trace()
                r2_marginal[i, j, f, 0, :] = dim_fold_marginal_df.iloc[0]['r2']

                dim_fold_marginal_df = apply_df_filters(marginal_df, data_file=data_file, dim=dim, fold_idx=f, dimreduc_method='PCA')
                try:
                    assert(dim_fold_marginal_df.shape[0] == 1)
                except:
                    pdb.set_trace()
                r2_marginal[i, j, f, 1, :] = dim_fold_marginal_df.iloc[0]['r2']


    dims = np.unique(sabes_df['dim'].values)
    sr2_vel_pca = np.zeros((28, 30, 5))
    for i, data_file in enumerate(data_files):
        for j, dim in enumerate(dims):
            data_file = data_file.split('/')[-1]
            pca_df = apply_df_filters(pca_decoding_df, dim=dim, data_file=data_file, dr_method='PCA')        
            for k in range(pca_df.shape[0]):
                sr2_vel_pca[i, j, k] = pca_df.iloc[k]['r2'][1]
    
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

    # FCA marginal r2 averaged over folds
    fca_marginal_r2 = np.mean(r2_marginal[:, :, :, 0, 1], axis=2)
    # PCA marginal
    pca_marginal_r2 = np.mean(r2_marginal[:, :, :, 1, 1], axis=2)

    # Panel 2: Paired differences
    ax[0].fill_between(dim_vals, np.mean(fca_r2 - fca_marginal_r2, axis=0) + np.std(fca_r2 - fca_marginal_r2, axis=0)/np.sqrt(28),
                    np.mean(fca_r2 - fca_marginal_r2, axis=0) - np.std(fca_r2 - fca_marginal_r2, axis=0)/np.sqrt(28), color=colors[1], alpha=0.25)

    ax[0].plot(dim_vals, np.mean(fca_r2 - fca_marginal_r2, axis=0), color=colors[1])

    ax[0].fill_between(dim_vals, np.mean(pca_r2 - pca_marginal_r2, axis=0) + np.std(pca_r2 - pca_marginal_r2, axis=0)/np.sqrt(28),
                    np.mean(pca_r2 - pca_marginal_r2, axis=0) - np.std(pca_r2 - pca_marginal_r2, axis=0)/np.sqrt(28), color=colors[0], alpha=0.25)

    ax[0].plot(dim_vals, np.mean(pca_r2 - pca_marginal_r2, axis=0), color=colors[0])
    ax[0].set_xlabel('Dimension', fontsize=14)
    ax[0].set_ylabel(r'$\Delta$' + ' Velocity Decoding ' + r'$r^2$', fontsize=14)
    ax[0].tick_params(axis='x', labelsize=12)
    ax[0].tick_params(axis='y', labelsize=12)

    ax[0].legend(['FCCA/FCCAm', 'PCA/PCAm'], loc='lower right', fontsize=12)
    ax[0].set_title('Paired differences in decoding', fontsize=16)

    #fig.savefig('%s/decoding_differences.pdf' % figpath, bbox_inches='tight', pad_inches=0)

    ################################################ Subspace angles ###########################################################

    dim = 10

    ss_angles = np.zeros((28, 5, 4, dim))
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

            dfpca_marginal = apply_df_filters(marginal_df, data_file=data_file, dimreduc_method='PCA', fold_idx=fold, dim=dim)
            dffca_marginal = apply_df_filters(marginal_df, data_file=data_file, dimreduc_method='LQGCA', fold_idx=fold, dim=dim)


            try:
                assert(dfpca.shape[0] == 1)
            except:
                pdb.set_trace()

            assert(dffcca.shape[0] == 1)
            assert(dfpca_marginal.shape[0] == 1)
            assert(dffca_marginal.shape[0] == 1)

            ss_angles[i, f, 0, :] = scipy.linalg.subspace_angles(dfpca.iloc[0]['coef'][:, 0:dim], dffcca.iloc[0]['coef'])
            ss_angles[i, f, 1, :] = scipy.linalg.subspace_angles(dfpca.iloc[0]['coef'][:, 0:dim], dfpca_marginal.iloc[0]['coef'][:, 0:dim])
            ss_angles[i, f, 2, :] = scipy.linalg.subspace_angles(dffcca.iloc[0]['coef'], dffca_marginal.iloc[0]['coef'])
            ss_angles[i, f, 3, :] = scipy.linalg.subspace_angles(dffca_marginal.iloc[0]['coef'], dfpca_marginal.iloc[0]['coef'][:, 0:dim])

    medianprops = dict(linewidth=0)
    bplot = ax[1].boxplot([np.mean(ss_angles[:, :, 0, :], axis=-1).ravel(), np.mean(ss_angles[:, :, 1, :], axis=-1).ravel(), np.mean(ss_angles[:, :, 2, :], axis=-1).ravel(), np.mean(ss_angles[:, :, 3, :], axis=-1).ravel()], 
                  patch_artist=True, medianprops=medianprops, notch=True, vert=False, showfliers=False)
    ax[1].set_yticklabels(['FCCA/PCA', 'PCA/PCAm', 'FCCA/FCCAm', 'FCCAm/PCAm'])
    ax[1].set_xlim([0, np.pi/2])
    ax[1].set_xlabel('Subspace angles (rads)')

    colors = ['blue', 'black', 'red', 'green']
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    #ax[2].set_xlim([0, np.pi/2])
    #ax[2].tick_params(axis='both', labelsize=12)
    #ax[2].set_ylabel('Count', fontsize=14)
    #ax[2].set_xlabel('Subspace angle (rads)', fontsize=14)
    #ax[2].legend(['FCCA/PCA', 'PCA/PCA m.', 'FCCA/FCCA m.', 'FCCA m./PCA m.'])
    #ax[2].set_xticks([0, np.pi/8, np.pi/4, 3 * np.pi/8, np.pi/2])
    #ax[2].set_xticklabels(['0', r'$\pi/8$', r'$\pi/4$', r'$3\pi/8$', r'$\pi/2$'])

    fig.tight_layout()
    fig.savefig('%s/marginal_summary.pdf' % figpath, bbox_inches='tight', pad_inches=0)
