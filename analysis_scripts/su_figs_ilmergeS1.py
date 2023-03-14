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

def get_scalar(df_, stat, neu_idx):

    if stat == 'decoding_weights':
        decoding_win = df_.iloc[0]['decoder_params']['decoding_window']
        # Restrict to velocity decoding
        c = calc_loadings(df_.iloc[0]['decoding_weights'][2:4].T, d=decoding_win)[neu_idx]
    elif stat == 'encoding_weights':
        decoding_win = df_.iloc[0]['decoder_params']['decoding_window']
        c =  calc_loadings(df_.iloc[0]['encoding_weights'], d=decoding_win)[neu_idx]        
    elif stat in ['su_r2_pos', 'su_r2_vel', 'su_r2_enc', 'su_var', 'su_act']:
        c = df_.iloc[0][stat][neu_idx]  
    elif stat == 'orientation_tuning':
        c = np.zeros(8)
        for j in range(8):
            c[j] = df_.loc[df_['bin_idx'] == j].iloc[0]['tuning_r2'][j, 2, neu_idx]
        c = np.mean(c)
        # c = odf_.iloc[0]

    return c

if __name__ == '__main__':

    # # Which plots should we make and save?

    # Where to save?
    if len(sys.argv) > 1:
        figpath = sys.argv[1]
    else:
        figpath = '/home/akumar/nse/neural_control/figs/loco_indy_merge'

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
    print(indy_df.iloc[0]['data_file'])
    
    # Still need to fold the indy results into calc_single unit statistics S1
    #good_loco_files.append(indy_df.iloc[0]['data_file'])

    loader_arg = {'bin_width':50, 'filter_fn':'none', 'filter_kwargs':{}, 'boxcox':0.5, 'spike_threshold':100, 'region':'S1'}
    decoder_arg = sabes_df.iloc[0]['decoder_args']

    s1df = apply_df_filters(sabes_df, decoder_args=decoder_arg, loader_args=loader_arg)
    s1df = apply_df_filters(s1df, data_file=good_loco_files)

    DIM = 6

    # Try the raw leverage scores instead
    loadings_l = []
    data_files = np.unique(s1df['data_file'].values)
    for i, data_file in tqdm(enumerate(data_files)):

        loadings = []
        for dimreduc_method in ['LQGCA', 'PCA']:
            loadings_fold = []
            for fold_idx in range(5):  
                df_ = apply_df_filters(s1df, data_file=data_file, fold_idx=fold_idx, dim=DIM, dimreduc_method=dimreduc_method)
                if dimreduc_method == 'LQGCA':
                    df_ = apply_df_filters(df_, dimreduc_args={'T': 3, 'loss_type': 'trace', 'n_init': 10})
                try:
                    assert(df_.shape[0] == 1)
                except:
                    pdb.set_trace()

                V = df_.iloc[0]['coef']
                if dimreduc_method == 'PCA':
                    V = V[:, 0:DIM]        
                loadings_fold.append(calc_loadings(V))


            # Average loadings across folds
            loadings.append(np.mean(np.array(loadings_fold), axis=0))

        for j in range(loadings[0].size):
            d_ = {}
            d_['data_file'] = data_file
            d_['FCCA_loadings'] = loadings[0][j]
            d_['PCA_loadings'] = loadings[1][j]
            # d_['DCA_loadings'] = loadings[2][j]
            d_['nidx'] = j
            loadings_l.append(d_)                

    loadings_df = pd.DataFrame(loadings_l)
    ########################## Histogram ########################################
    #############################################################################

    with open('/mnt/Secondary/data/postprocessed/sabes_su_calcsS1.dat', 'rb') as f:
        sabes_su_l = pickle.load(f)

    sabes_su_df = pd.DataFrame(sabes_su_l)

    # Dimensionality selection
    itrim_df = loadings_df
    data_files = np.unique(itrim_df['data_file'].values)
    # Collect the desired single unit statistics into an array with the same ordering as those present in the loadings df
    stats = ['su_var', 'su_act', 'decoding_weights', 'su_r2_enc']
    carray = []
    for i, data_file in enumerate(data_files):
        df = apply_df_filters(itrim_df, data_file=data_file)
        carray_ = np.zeros((df.shape[0], len(stats)))
        for j in range(df.shape[0]):                    # Find the correlation between 
            for k, stat in enumerate(stats):
                # Grab the unique identifiers needed
                nidx = df.iloc[j]['nidx']
                df_ = apply_df_filters(sabes_su_df, data_file=data_file)
                try:
                    carray_[j, k] = get_scalar(df_, stat, nidx)
                except:
                    pdb.set_trace()
        carray.append(carray_)

    su_r = np.zeros((len(carray), 2, carray[0].shape[1]))
    keys = ['FCCA_loadings', 'PCA_loadings']

    X = []
    Yf = []
    Yp = []
    for i in range(len(carray)):
        for j in range(2):

            df = apply_df_filters(itrim_df, data_file=data_files[i])
            x1 = df[keys[j]].values

            if j == 0:
                Yf.extend(x1)
            else:
                Yp.extend(x1)

            xx = []

            for k in range(carray[0].shape[1]):
                x2 = carray[i][:, k]
                xx.append(x2)
                su_r[i, j, k] = scipy.stats.spearmanr(x1, x2)[0]
            xx = np.array(xx).T            
        X.append(xx)

    X = np.vstack(X)
    Yf = np.array(Yf)[:, np.newaxis]
    Yp = np.array(Yp)[:, np.newaxis]
    assert(X.shape[0] == Yf.shape[0])
    assert(X.shape[0] == Yp.shape[0])

    # Train a linear model to predict loadings from the single unit statistics and then assess the 
    # spearman correlation between predicted and actual loadings

    linmodel1 = LinearRegression().fit(X, Yp)
    linmodel2 = LinearRegression().fit(X, np.log10(Yp))

    Yp_pred1 = linmodel1.predict(X)
    Yp_pred2 = linmodel2.predict(X)

    r1p = scipy.stats.spearmanr(Yp_pred1.squeeze(), Yp.squeeze())[0]
    r2p = scipy.stats.spearmanr(Yp_pred2.squeeze(), Yp.squeeze())[0]

    linmodel1 = LinearRegression().fit(X, Yf)
    linmodel2 = LinearRegression().fit(X, np.log10(Yf))

    Yf_pred1 = linmodel1.predict(X)
    Yf_pred2 = linmodel2.predict(X)

    r1f = scipy.stats.spearmanr(Yf_pred1.squeeze(), Yf.squeeze())[0]
    r2f = scipy.stats.spearmanr(Yf_pred2.squeeze(), Yf.squeeze())[0]

    print(r1p)
    print(r1f)

    fig, ax = plt.subplots(figsize=(5, 5),)

    # Prior to averaging, run tests. 

    # Updated for multiple comparisons adjustment. 
    _, p1 = scipy.stats.wilcoxon(su_r[:, 0, 0], su_r[:, 1, 0], alternative='less')
    _, p2 = scipy.stats.wilcoxon(su_r[:, 0, 1], su_r[:, 1, 1], alternative='less')
    _, p3 = scipy.stats.wilcoxon(su_r[:, 0, 2], su_r[:, 1, 2], alternative='less')
    _, p4 = scipy.stats.wilcoxon(su_r[:, 0, 3], su_r[:, 1, 3], alternative='less')

    # sort
    pvec = np.sort([p1, p2, p3, p4])
    # Sequentially test and determine the minimum significance level
    a1 = pvec[0] * 4
    a2 = pvec[1] * 3
    a3 = pvec[2] * 2
    a4 = pvec[3]
    print(max([a1, a2, a3, a4]))   

    std_err = np.std(su_r, axis=0).ravel()/np.sqrt(35)
    su_r = np.mean(su_r, axis=0).ravel()

    # Permute so that each statistic is next to each other
    su_r = su_r[[0, 4, 1, 5, 2, 6, 3, 7]]
    std_err = std_err[[0, 4, 1, 5, 2, 6, 3, 7]]

    bars = ax.bar([0, 1, 3, 4, 6, 7, 9, 10],
                    su_r,
                    color=['r', 'k', 'r', 'k', 'r', 'k', 'r', 'k'], alpha=0.65,
                    yerr=std_err, capsize=5)

    # Place numerical values above the bars
    # for rect in bars: 
    #     if rect.get_height() > 0:
    #         ax.text(rect.get_x() + rect.get_width()/2, np.sign(rect.get_height()) * (np.abs(rect.get_height()) + 0.075), '%.2f' % rect.get_height(),
    #                 ha='center', va='bottom', fontsize=10)
    #     else:
    #         ax.text(rect.get_x() + rect.get_width()/2, np.sign(rect.get_height()) * (np.abs(rect.get_height()) + 0.11), '%.2f' % rect.get_height(),
    #                 ha='center', va='bottom', fontsize=10)

    # Add significance tests
    ax.text(0.5, 1.0, '**', ha='center')
    ax.text(3.5, 0.7, '**', ha='center')
    ax.text(6.5, 0.6, '**', ha='center')
    ax.text(9.5, 0.76, '**', ha='center')


    ax.set_ylim([-0.5, 1.1])
    ax.set_xticks([0.5, 3.5, 6.5, 9.5])

    ax.set_xticklabels(['S.U. Variance', 'Autocorr. time', 'Decoding Weights', 'S.U. Enc. ' + r'$r^2$'], rotation=30, fontsize=12, ha='right')

    ax.tick_params(axis='y', labelsize=12)

    # Manual creation of legend
    colors = ['r', 'k']
    handles = [plt.Rectangle((0,0),1,1, color=c, alpha=0.65) for c in colors]
    labels = ['FCCA', 'PCA']
    ax.legend(handles, labels, loc='lower right')

    ax.set_ylabel('Spearman Correlation ' + r'$\rho$', fontsize=14)
    ax.set_yticks([-0.5, 0, 0.5, 1.])

    # Horizontal line at 0
    ax.hlines(0, -0.5, 10.5, color='k')

    #fig.savefig('%s/su_spearman_d%dS1.pdf' % (figpath, DIM), bbox_inches='tight', pad_inches=0)