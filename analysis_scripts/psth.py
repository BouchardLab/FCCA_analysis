import pdb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import scipy 
import time
import glob
import pickle
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
import sys

from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler

from npeet.entropy_estimators import entropy as knn_entropy

sys.path.append('..')

from utils import apply_df_filters, calc_loadings
from loaders import load_sabes
from segmentation import reach_segment_sabes

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

def get_top_neurons(dimreduc_df, method1='FCCA', method2='PCA', n=10, pairwise_exclude=True):

    data_path = globals()['data_path']
    T = globals()['T']
    n = globals()['n']
    bin_width = globals()['bin_width']

    # Load dimreduc_df and calculate loadings
    data_files = np.unique(dimreduc_df['data_file'].values)
    # Try the raw leverage scores instead
    loadings_l = []

    for i, data_file in tqdm(enumerate(data_files)):
            loadings = []
            for dimreduc_method in ['DCA', 'KCA', 'LQGCA', 'PCA']:
                loadings_fold = []
                for fold_idx in range(5):            
                    df_ = apply_df_filters(dimreduc_df, data_file=data_file, fold_idx=fold_idx, dim=6, dimreduc_method=dimreduc_method)
                    if dimreduc_method == 'LQGCA':
                        df_ = apply_df_filters(df_, dimreduc_args={'T': 3, 'loss_type': 'trace', 'n_init': 10})
                    V = df_.iloc[0]['coef']
                    if dimreduc_method == 'PCA':
                        V = V[:, 0:2]        
                    loadings_fold.append(calc_loadings(V))

                # Average loadings across folds
                loadings.append(np.mean(np.array(loadings_fold), axis=0))

            for j in range(loadings[0].size):
                d_ = {}
                d_['data_file'] = data_file
                d_['DCA_loadings'] = loadings[0][j]
                d_['KCA_loadings'] = loadings[1][j]
                d_['FCCA_loadings'] = loadings[2][j]
                d_['PCA_loadings'] = loadings[3][j]
                d_['nidx'] = j
                loadings_l.append(d_)                

    loadings_df = pd.DataFrame(loadings_l)

    # For each data file, find the top 5 neurons that are high in one method but low in all others
    top_neurons_l = []
    n = 10
    for i, data_file in tqdm(enumerate(data_files)):
        df_ = apply_df_filters(loadings_df, data_file=data_file)
        DCA_ordering = np.argsort(df_['DCA_loadings'].values)
        KCA_ordering = np.argsort(df_['KCA_loadings'].values)
        FCCA_ordering = np.argsort(df_['FCCA_loadings'].values)
        PCA_ordering = np.argsort(df_['PCA_loadings'].values)
        
        rank_diffs = np.zeros((DCA_ordering.size, 6))
        for j in range(df_.shape[0]):
            rank_diffs[j, 0] = list(DCA_ordering).index(j) - list(KCA_ordering).index(j)
            rank_diffs[j, 1] = list(DCA_ordering).index(j) - list(FCCA_ordering).index(j)
            rank_diffs[j, 2] = list(DCA_ordering).index(j) - list(PCA_ordering).index(j)
            
            rank_diffs[j, 3] = list(KCA_ordering).index(j) - list(FCCA_ordering).index(j)
            rank_diffs[j, 4] = list(KCA_ordering).index(j) - list(PCA_ordering).index(j)
            
            rank_diffs[j, 5] = list(FCCA_ordering).index(j) - list(PCA_ordering).index(j)

        # Find the top 5 neurons according to all pairwise high/low orderings
        top_neurons = np.zeros((2, n)).astype(int)


        # User selects which pairwise comparison is desired
        method_dict = {'PCA': PCA_ordering, 'DCA': DCA_ordering, 'KCA':KCA_ordering, 'FCCA':FCCA_ordering}

        top1 = []
        top2 = []

        idx = 0
        while not np.all([len(x) >= n for x in [top1, top2]]):
            idx += 1
            # Take neurons from the top ordering of each method. Disregard neurons that 
            # show up in all methods
            # top_DCA = DCA_ordering[-idx]
            top1_ = method_dict[method1][-idx]
            top2_ = method_dict[method2][-idx]

            if top1_ != top2_:
                if top1_ not in top2:
                    top1.append(top1_)
                if top2_ not in top1:
                    top2.append(top2_)
            else:
                continue

        top_neurons[0, :] = top1[0:n]
        top_neurons[1, :] = top2[0:n]

        top_neurons_l.append({'data_file':data_file, 'rank_diffs':rank_diffs, 'top_neurons': top_neurons}) 
    top_neurons_df = pd.DataFrame(top_neurons_l)
    
    return top_neurons_df

def PSTH_plot(top_neurons_df, path):
    data_path = globals()['data_path']
    T = globals()['T']
    n = globals()['n']
    bin_width = globals()['bin_width']

    ndf = 7
    fig, ax = plt.subplots(2 * 4, ndf, figsize=(4*ndf, 32))

    data_files = np.unique(top_neurons_df['data_file'].values)

    for h, data_file in enumerate(data_files):

        df_ = apply_df_filters(top_neurons_df, data_file=data_file)
        dat = load_sabes('%s/%s' % (data_path, data_file), boxcox=None, high_pass=False)
        dat_segment = reach_segment_sabes(dat, start_time=start_times[data_file.split('.mat')[0]])
        
        T = 30
        t = np.array([t_[1] - t_[0] for t_ in dat_segment['transition_times']])
        valid_transitions = np.arange(t.size)[t >= T]

        # (Bin size 50 ms)
        time = 40 * np.arange(T)

        for i in range(2):
            for j in range(n):
                tn = df_.iloc[0]['top_neurons'][i, j]    
                try:
                    x_ = np.array([dat['spike_rates'][0, dat_segment['transition_times'][idx][0]:dat_segment['transition_times'][idx][0] + T, tn] 
                                for idx in valid_transitions])
                except:
                    pdb.set_trace()
                
                # Mean subtract
    #            x_ -= np.mean(x_, axis=1, keepdims=True)
                try:
                    x_ = StandardScaler().fit_transform(x_.T).T
                    x_ = gaussian_filter1d(x_, sigma=2)
                    x_ = np.mean(x_, axis=0)

                    if i == 0:
                        ax[2 * (h//7) + i, h % 7].plot(time, x_, 'k', alpha=0.5)
                        ax[2 * (h//7) + i, h % 7].set_title(data_file)

                    if i == 1:
                        ax[2 * (h//7) + i, h % 7].plot(time, x_, 'r', alpha=0.5)
                        ax[2 * (h//7) + i, h % 7].set_title(data_file)
                except:
                    continue

    for i in range(ndf):
        #ax[0, i].set_title('Top FCCA neurons', fontsize=14)
        #ax[1, i].set_title('Top PCA neurons', fontsize=14)
        ax[1, i].set_xlabel('Time (ms)')
        ax[0, i].set_ylabel('Z-scored trial averaged firing rate')

    fig.savefig('%s/PSTH.pdf' % path, bbox_inches='tight', pad_inches=0)

def cross_cov_calc(top_neurons_df):

    data_path = globals()['data_path']
    T = globals()['T']
    n = globals()['n']
    bin_width = globals()['bin_width']

    # (Bin size 50 ms)
    time = 50 * np.arange(T)

    data_files = np.unique(top_neurons_df['data_file'].values)

    cross_covs = np.zeros((len(data_files), 2, n, n, time.size))
    cross_covs_01 = np.zeros((len(data_files), 2, n, n, time.size))

    for h, data_file in enumerate(data_files):
        df_ = apply_df_filters(top_neurons_df, data_file=data_file)
        dat = load_sabes('%s/%s' % (data_path, data_file), boxcox=None, high_pass=False)
        dat_segment = reach_segment_sabes(dat, start_time=start_times[data_file.split('.mat')[0]])
        
        t = np.array([t_[1] - t_[0] for t_ in dat_segment['transition_times']])
        valid_transitions = np.arange(t.size)[t >= T]


        for i in range(2):
            
            # Store trajectories for subsequent pairwise analysis
            x = np.zeros((n, time.size))

            for j in range(n):
                tn = df_.iloc[0]['top_neurons'][i, j]    
                x_ = np.array([dat['spike_rates'][0, dat_segment['transition_times'][idx][0]:dat_segment['transition_times'][idx][0] + T, tn] 
                            for idx in valid_transitions])
                
                x_ = StandardScaler().fit_transform(x_.T).T
                x_ = gaussian_filter1d(x_, sigma=2)
                x_ = np.mean(x_, axis=0)

                # Put on a 0-1 scale
                x[j, :] = x_

            for j in range(n):
                for k in range(n):
                    cross_covs[h, i, j, k] = np.correlate(x[j], x[k], mode='same')/30
                    cross_covs_01[h, i, j, k] = np.correlate(x[j]/np.max(x[j]), x[k]/np.max(x[k]), mode='same')/30

    return cross_covs, cross_covs_01

def cross_covs_statistics(cross_covs, cross_covs_01):
    data_path = globals()['data_path']
    T = globals()['T']
    n = globals()['n']
    bin_width = globals()['bin_width']

    # Significance tests
    # Reshape into a sequence so we can sort along the axis of pairwise cross-correlation 
    cc_mag = np.zeros((cross_covs.shape[0], cross_covs.shape[1], cross_covs.shape[2] * cross_covs.shape[3] - cross_covs.shape[2], cross_covs.shape[-1]))
    cc_tau = np.zeros((cross_covs.shape[0], cross_covs.shape[1], cross_covs.shape[2] * cross_covs.shape[3] - cross_covs.shape[2], cross_covs.shape[-1]))

    idx = 0
    for i in range(10):
        for j in range(10):
            if i == j:
                continue
            cc_mag[:, :, idx, :] = cross_covs01[:, :, i, j, :]
            cc_tau[:, :, idx, :] = cross_covs_01[:, :, i, j, :]
            idx += 1

    tau_max = np.zeros((cc_tau.shape[0], cc_tau.shape[1], cc_tau.shape[2]))
    mag = np.zeros((cc_mag.shape[0], cc_tau.shape[1], cc_tau.shape[2]))

    # references for sharing axes
    for h in range(cc_tau.shape[0]):
        for i in range(2):
            # Sort by max cross-cov. Impart the correct units
            tau_max[h, i] = bin_width * (np.sort(np.argmax(cc_tau[h, i, :, :], axis=-1)) - T//2)
    
    mag = np.max(cc_mag, axis=-1)

    # Paired difference tests pooling all samples together
    wstat1, p1 = scipy.stats.wilcoxon(tau_max[:, 0, :].ravel(), tau_max[:, 1, :].ravel())
    wstat2, p2 = scipy.stats.wilcoxon(mag[:, 0, :].ravel(), mag[:, 1, :].ravel())

    # Calculation of entropy using knn 
    tau_h1 = [knn_entropy(tau_max[idx, 0, :][:, np.newaxis]) for idx in range(tau_max.shape[0])]
    tau_h2 = [knn_entropy(tau_max[idx, 1, :][:, np.newaxis]) for idx in range(tau_max.shape[0])]

    avg_mag1 = np.mean(mag[:, 0, :], axis=-1)
    avg_mag2 = np.mean(mag[:, 1, :], axis=-1)

    # Paired difference tests
    wstat3, p3 = scipy.stats.wilcoxon(tau_h1, tau_h2)
    wstat4, p4 = scipy.stats.wilcoxon(avg_mag1, avg_mag2)

    return tau_max, mag, (wstat1, wstat2, wstat3, wstat4), (p1, p2, p3, p4)

def cross_cov_plots(method1, method2, tau_max, mag, stats, p, path):

    tau_h1 = [knn_entropy(tau_max[idx, 0, :][:, np.newaxis], k=5) for idx in range(tau_max.shape[0])]
    tau_h2 = [knn_entropy(tau_max[idx, 1, :][:, np.newaxis], k=5) for idx in range(tau_max.shape[0])]

    data_path = globals()['data_path']
    T = globals()['T']  
    n = globals()['n']
    bin_width = globals()['bin_width']

    # Plot the histograms
    fig, ax = plt.subplots(6, 5, figsize=(25, 30))
    for i in range(28):
        a = ax[np.unravel_index(i, (6, 5))]
        a.hist(tau_max[i, 0], alpha=0.5, bins=np.linspace(-1500, 1500, 25))
        a.hist(tau_max[i, 1], alpha=0.5)    

        # Title with entropy
        a.set_title('FCCA h: %.3f, PCA h:%.3f' % (tau_h1[i], tau_h2[i]))
        a.legend(['FCCA', 'PCA'])

    fig.suptitle('Peak cross-cov time, %s vs. %s' % (method1, method2))
    fig.savefig('%s/tau_hist.pdf' % path, bbox_inches='tight', pad_inches=0)

    # Plot the histograms
    fig, ax = plt.subplots(6, 5, figsize=(25, 30))
    for i in range(28):
        a = ax[np.unravel_index(i, (6, 5))]
        a.hist(mag[i, 0], alpha=0.5)
        a.hist(mag[i, 1], alpha=0.5)
        a.set_title('FCCA h: %.3f, PCA h:%.3f' % (np.mean(mag[i, 0]), np.mean(mag[i, 1])))
        a.legend(['FCCA', 'PCA'])

    fig.suptitle('Peak cross-cov magnitude, %s vs. %s' % (method1, method2))
    fig.savefig('%s/mag_hist.pdf' % path, bbox_inches='tight', pad_inches=0)

    avg_mag1 = np.mean(mag[:, 0, :], axis=-1)
    avg_mag2 = np.mean(mag[:, 1, :], axis=-1)

    # Boxplots
    fig, ax = plt.subplots(1, 2, figsize=(4, 4))

    medianprops = dict(linewidth=0)
    bplot1 = ax[0].boxplot([tau_h1, tau_h2], patch_artist=True, medianprops=medianprops, notch=True, showfliers=False)
    bplot2 = ax[1].boxplot([avg_mag1, avg_mag2], patch_artist=True, medianprops=medianprops, notch=True, showfliers=False)

    ax[0].set_xticklabels([method1, method2])
    ax[0].set_yticks([0, -15, -30])
    ax[1].set_xticklabels([method1, method2])
    # ax[1].set_yticks([15, 30, 45])
    #ax[0].set_title(r'$\tau$' + '-entropy, p=%f' % p[2], fontsize=10)
    #ax[1].set_title('Avg. magnitude, stat: p=%f' % p[3], fontsize=10)
    ax[0].set_title('p=%f' % p[2], fontsize=10)
    ax[1].set_title('p=%f' % p[3], fontsize=10)


    ax[0].set_ylabel('Entropy of peak cross-corr. times', fontsize=12)
    ax[1].set_ylabel('Average peak cross-corr.', fontsize=12)


    # fill with colors
    colors = ['red', 'black']
    for bplot in (bplot1, bplot2):
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

    fig.tight_layout()
    fig.savefig('%s/boxplots.pdf' % path, bbox_inches='tight', pad_inches=0)
    

if __name__ == '__main__':

    # Add data_path, T, n, bin_width to globals
    # Set these
    data_path = '/mnt/Secondary/data/sabes'
    T = 30
    n = 10
    bin_width = 50
    
    globals()['data_path'] = data_path
    globals()['T'] =  T
    globals()['n'] = n
    globals()['bin_width'] = bin_width

    # Load dimreduc_df
    with open('/home/akumar/nse/neural_control/data/indy_decoding_df2.dat', 'rb') as f:
        dimreduc_df = pd.DataFrame(pickle.load(f))

    method1 = 'FCCA'
    method2 = 'PCA'

    # Get top neurons
    top_neurons_df = get_top_neurons(dimreduc_df, method1=method1, method2=method2, n=10, pairwise_exclude=True)
    
    # Plot PSTH
    PSTH_plot(top_neurons_df, '/home/akumar/nse/neural_control/figs')

    # Cross-covariance stuff
    cross_covs, cross_covs01 = cross_cov_calc(top_neurons_df)
    tau_max, mag, stats, p = cross_covs_statistics(cross_covs, cross_covs01)
    cross_cov_plots(method1, method2, tau_max, mag, stats, p, '/home/akumar/nse/neural_control/figs')
    
