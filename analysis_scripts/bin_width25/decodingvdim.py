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
        figpath = '/home/akumar/nse/neural_control/figs/bin_width25'

    # # Sequentially do indy, peanut decoding
    with open('/mnt/Secondary/data/postprocessed/loco_decoding25.dat', 'rb') as f:
        rl = pickle.load(f)

    master_df = pd.DataFrame(rl)
    sabes_df = apply_df_filters(master_df, dimreduc_method='LQGCA', dimreduc_args={'T':3, 'loss_type':'trace', 'n_init':10}, 
                                decoder_args={'trainlag': 4, 'testlag': 4, 'decoding_window':5})
    pca_decoding_df = apply_df_filters(master_df, dimreduc_method='PCA', decoder_args={'trainlag': 4, 'testlag': 4, 'decoding_window':5})
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
            pca_df = apply_df_filters(pca_decoding_df, dim=dim, data_file=data_file)        
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
    fig.savefig('%s/loco_vel_decoding.pdf' % figpath, bbox_inches='tight', pad_inches=0)