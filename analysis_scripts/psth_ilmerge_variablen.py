import pdb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from mpl_toolkits.mplot3d import Axes3D
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
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from npeet.entropy_estimators import entropy as knn_entropy
from dca.cov_util import calc_cross_cov_mats_from_data, calc_pi_from_cross_cov_mats

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

def get_top_neurons_alt(dimreduc_df, n=10, data_path=None, T=None, bin_width=None):

    if data_path is None:
        data_path = globals()['data_path']
    if T is None:
        T = globals()['T']
    if bin_width is None:
        bin_width = globals()['bin_width']

    # Load dimreduc_df and calculate loadings
    data_files = np.unique(dimreduc_df['data_file'].values)
    # Try the raw leverage scores instead
    loadings_pca = []
    idxs_pca = []
    loadings_fca = []
    idxs_fca = []

    for i, data_file in tqdm(enumerate(data_files)):
        loadings = []
        for dimreduc_method in ['LQGCA', 'PCA']:
            loadings_fold = []
            for fold_idx in range(5):            
                df_ = apply_df_filters(dimreduc_df, data_file=data_file, fold_idx=fold_idx, dim=6, dimreduc_method=dimreduc_method)
                if dimreduc_method == 'LQGCA':
                    df_ = apply_df_filters(df_, dimreduc_args={'T': 3, 'loss_type': 'trace', 'n_init': 10})

                assert(df_.shape[0] == 1)
                V = df_.iloc[0]['coef']
                if dimreduc_method == 'PCA':
                    V = V[:, 0:6]        
                loadings_fold.append(calc_loadings(V))

            if dimreduc_method == 'LQGCA':
                loadings_fca.extend(np.mean(loadings_fold, axis=0))
                idxs_fca.extend([(i, j) for j in np.arange(loadings_fold[0].size)])
            elif dimreduc_method == 'PCA':
                loadings_pca.extend(np.mean(loadings_fold, axis=0))
                idxs_pca.extend([(i, j) for j in np.arange(loadings_fold[0].size)])
   
    # Assemble the disjoint sets across all recording sessions.
    # Want to end up with n * n_data_files neurons, but pooled across sessions now.
    top_neurons_pca = []
    top_neurons_fca = []
    
    N = n * len(data_files)

    pca_ordering = np.argsort(loadings_pca)[::-1]
    fca_ordering = np.argsort(loadings_fca)[::-1]

    def empty_min(x):
        if len(x) == 0:
            return np.inf
        else:
            return np.min(x)

    idx = 0
    while not np.all([len(x) >= N for x in [top_neurons_pca, top_neurons_fca]]):
        top_pca_candidate = idxs_pca[pca_ordering[idx]]

        # Accept the candidate only if it is not contained in the top N FCCA neurons
        candidate_fca_idx = idxs_fca.index(top_pca_candidate)

        if not candidate_fca_idx in fca_ordering[:N]:
            top_neurons_pca.append(pca_ordering[idx])

        top_fca_candidate = idxs_fca[fca_ordering[idx]]

        candidate_pca_idx = idxs_pca.index(top_fca_candidate)

        if not candidate_pca_idx in pca_ordering[:N]:
            top_neurons_fca.append(fca_ordering[idx])
    
        idx += 1

    return top_neurons_pca, top_neurons_fca, loadings_pca, loadings_fca

def get_top_neurons(dimreduc_df, fraction_cutoff=0.9, method1='FCCA', method2='PCA', n=10, 
                    pairwise_exclude=True, data_path=None, T=None, bin_width=None):

    if data_path is None:
        data_path = globals()['data_path']
    if T is None:
        T = globals()['T']
    if bin_width is None:
        bin_width = globals()['bin_width']

    # Load dimreduc_df and calculate loadings
    data_files = np.unique(dimreduc_df['data_file'].values)
    # Try the raw leverage scores instead
    loadings_l = []

    for i, data_file in tqdm(enumerate(data_files)):
        loadings = []
        for dimreduc_method in ['LQGCA', 'PCA']:
            loadings_fold = []
            for fold_idx in range(5):            
                df_ = apply_df_filters(dimreduc_df, data_file=data_file, fold_idx=fold_idx, dim=6, dimreduc_method=dimreduc_method)
                if dimreduc_method == 'LQGCA':
                    df_ = apply_df_filters(df_, dimreduc_args={'T': 3, 'loss_type': 'trace', 'n_init': 10})

                assert(df_.shape[0] == 1)
                V = df_.iloc[0]['coef']
                if dimreduc_method == 'PCA':
                    V = V[:, 0:6]        
                loadings_fold.append(calc_loadings(V))

            # Average loadings across folds
            loadings.append(np.mean(np.array(loadings_fold), axis=0))

        for j in range(loadings[0].size):
            d_ = {}
            d_['data_file'] = data_file
            d_['FCCA_loadings'] = loadings[0][j]
            d_['PCA_loadings'] = loadings[1][j]
            d_['nidx'] = j
            loadings_l.append(d_)                

    loadings_df = pd.DataFrame(loadings_l)

    # For each data file, find the top 5 neurons that are high in one method but low in all others
    top_neurons_l = []
    for i, data_file in tqdm(enumerate(data_files)):
        df_ = apply_df_filters(loadings_df, data_file=data_file)

        # Order neurons according to linear fractional FBC score
        fbc_fraction = np.divide(df_['FCCA_loadings'].values, df_['FCCA_loadings'].values + df_['PCA_loadings'].values)
        ffc_fraction = np.divide(df_['PCA_loadings'].values, df_['FCCA_loadings'].values + df_['PCA_loadings'].values)
        ordering = np.argsort(fbc_fraction)[::-1]
        
        # FCCA_ordering = np.argsort(df_['FCCA_loadings'].values)
        # PCA_ordering = np.argsort(df_['PCA_loadings'].values)
        
        # rank_diffs = np.zeros((PCA_ordering.size,))
        # for j in range(df_.shape[0]):            
        #     rank_diffs[j] = list(FCCA_ordering).index(j) - list(PCA_ordering).index(j)

        # Find the top 5 neurons according to all pairwise high/low orderings
        # cutoff_1 = np.quantile(fbc_fraction, fraction_cutoff)
        # cutoff_2 = np.quantile(ffc_fraction, fraction_cutoff)

        fbc_neuron_indices = np.arange(df_.shape[0])[fbc_fraction > fraction_cutoff]
        ffc_neuron_indices = np.arange(df_.shape[0])[ffc_fraction > fraction_cutoff]
        pdb.set_trace()
        # Possible sets are off by a few..shave to the smallest length
        # min_size = min(fbc_neuron_indices.size, ffc_neuron_indices.size)
        # fbc_neuron_indices = fbc_neuron_indices[:min_size]
        # ffc_neuron_indices = ffc_neuron_indices[:min_size]

        # Ensure there is no overlap
        assert(len(np.intersect1d(fbc_neuron_indices, ffc_neuron_indices)) == 0)

        top_neurons = [fbc_neuron_indices.astype(int), ffc_neuron_indices.astype(int)]
        top_neurons_l.append({'data_file':data_file, 'top_neurons': top_neurons}) 


    top_neurons_df = pd.DataFrame(top_neurons_l)
    
    return top_neurons_df, loadings_df

def heatmap_plot(top_neurons_df, path):

    data_path = globals()['data_path']
    T = globals()['T']
    n = globals()['n']
    bin_width = globals()['bin_width']

    # Plot the pairwise cross-correlations as a sequence of 2D curves
    n_df = 7
    #fig, ax = plt.subplots(2 * 4, ndf, figsize=(4*ndf, 32))
    fig = plt.figure(figsize=(5*7, 5*4))
    gs = GridSpec(4, 7, hspace=0.2, wspace=0.1)

    data_files = np.unique(top_neurons_df['data_file'].values)

    # references for sharing axes
    ax_obj = np.zeros(len(data_files), dtype='object')

    for h, data_file in enumerate(data_files):

        df_ = apply_df_filters(top_neurons_df, data_file=data_file)
        dat = load_sabes('%s/%s' % (data_path, data_file), boxcox=None, high_pass=False)
        dat_segment = reach_segment_sabes(dat, data_file=data_file.split('.mat')[0])
        
        T = 30
        t = np.array([t_[1] - t_[0] for t_ in dat_segment['transition_times']])
        valid_transitions = np.arange(t.size)[t >= T]

        # (Bin size 50 ms)
        time = 40 * np.arange(T)

        cols = ['k', 'r']
        titles = ['FCCA', 'PCA']
        sgs = gs[np.unravel_index(h, (4, 7))].subgridspec(1, 2, wspace=0.05, hspace=0.0)
        for i in range(2):
            x = np.zeros((T, n))
            for j in range(n):
                tn = df_.iloc[0]['top_neurons'][i, j]    
                try:
                    x_ = np.array([dat['spike_rates'][0, dat_segment['transition_times'][idx][0]:dat_segment['transition_times'][idx][0] + T, tn] 
                                for idx in valid_transitions])
                except:
                    pdb.set_trace()
                
                # x_ = MinMaxScaler().fit_transform(x_.T).T
                x_ = gaussian_filter1d(x_, sigma=2)
                x_ = np.mean(x_, axis=0)
                x[:, j]  = x_

            # Second round of mixmaxscaling
            x = MinMaxScaler().fit_transform(x)

            # Sort by time of peak response
            x = x[:, np.argsort((np.argmax(x, axis=0)))]

            if i == 0:
                ax = fig.add_subplot(sgs[i])
                ax_obj[h] = ax
            else:
                ax = fig.add_subplot(sgs[i], sharey=ax_obj[h])
                ax.set_yticks([])

            ax.pcolor(x)
            # ax.set_title(titles[i])

            # x-axis label, y-axis label, colormap...        

            # ax.set_aspect('equal')
            # for idx in range(cc_offdiag.shape[2]):
            #     ax.plot(idx * np.ones(30), np.arange(30), cc_offdiag[0, i, ordering[idx], :], cols[i], alpha=0.5)
        gs.update(left=0.55, right=0.98, hspace=0.05)

    fig.tight_layout()
    fig.savefig('%s/heatmap.pdf' % path, bbox_inches='tight', pad_inches=0)

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
        dat_segment = reach_segment_sabes(dat, data_file=data_file.split('.mat')[0])
        
        T = 30
        t = np.array([t_[1] - t_[0] for t_ in dat_segment['transition_times']])
        valid_transitions = np.arange(t.size)[t >= T]

        # (Bin size 50 ms)
        time = 50 * np.arange(T)

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
    # n = globals()['n']
    n = np.array([[top_neurons_df.iloc[j]['top_neurons'][k].size for k in range(2)]
        for j in range(top_neurons_df.shape[0])])
    bin_width = globals()['bin_width']
    # (Bin size 50 ms)
    time = 50 * np.arange(T)

    data_files = np.unique(top_neurons_df['data_file'].values)

    cross_covs = np.zeros((len(data_files), 2), dtype=np.object)
    cross_covs_01 = np.zeros((len(data_files), 2), dtype=np.object)
    dyn_range = np.zeros((len(data_files), 2), dtype=np.object)

    for h, data_file in enumerate(data_files):
        df_ = apply_df_filters(top_neurons_df, data_file=data_file)
        dat = load_sabes('%s/%s' % (data_path, data_file), boxcox=None, high_pass=False)
        dat_segment = reach_segment_sabes(dat, data_file=data_file.split('.mat')[0])
        
        t = np.array([t_[1] - t_[0] for t_ in dat_segment['transition_times']])
        valid_transitions = np.arange(t.size)[t >= T]

        for i in range(2):
            
            # Store trajectories for subsequent pairwise analysis
            x = np.zeros((n[h, i], time.size))

            for j in range(n[h, i]):
                tn = df_.iloc[0]['top_neurons'][i][j]
                x_ = np.array([dat['spike_rates'][0, dat_segment['transition_times'][idx][0]:dat_segment['transition_times'][idx][0] + T, tn] 
                            for idx in valid_transitions])
                
                x_ = StandardScaler().fit_transform(x_.T).T
                x_ = gaussian_filter1d(x_, sigma=2)
                x_ = np.mean(x_, axis=0)

                # Put on a 0-1 scale
                x[j, :] = x_

            cross_covs_ = np.zeros((n[h, i], n[h, i], time.size))
            cross_covs_01_ = np.zeros((n[h, i], n[h, i], time.size))
            dyn_range_ = np.zeros((n[h, i],))
            for j in range(n[h, i]):
                for k in range(n[h, i]):
                    cross_covs_[j, k] = np.correlate(x[j], x[k], mode='same')/T
                    cross_covs_01_[j, k] = np.correlate(x[j]/np.max(x[j]), x[k]/np.max(x[k]), mode='same')/T
                dyn_range_[j] = np.max(x[j])

            cross_covs[h, i] = cross_covs_
            cross_covs_01[h, i] = cross_covs_01_
            dyn_range[h, i] = dyn_range_

    return cross_covs, cross_covs_01, dyn_range

def statistics(cross_covs, cross_covs_01, dyn_range):
    data_path = globals()['data_path']
    T = globals()['T']
    bin_width = globals()['bin_width']

    # Significance tests
    # Reshape into a sequence so we can sort along the axis of pairwise cross-correlation 
    # (data_files, 2) --> object storing n^2 - n traces of length time.size
    cc_flat = np.zeros((cross_covs.shape[0], cross_covs.shape[1]), dtype=np.object)

    for i in range(cc_flat.shape[0]):
        for j in range(cc_flat.shape[1]):
            cc_flat[i, j] = cross_covs[i, j].reshape((-1, cross_covs[i, j].shape[-1]))

    tau_max = np.zeros((cc_flat.shape[0], cc_flat.shape[1]), dtype=np.object)

    # references for sharing axes
    for h in range(cc_flat.shape[0]):
        for i in range(cc_flat.shape[1]):
            # Sort by max cross-cov. Impart the correct units
            tau_max[h, i] = bin_width * (np.sort(np.argmax(cc_flat[h, i], axis=-1)) - T//2)

    mag = [[np.max(cc_flat[h, i], axis=-1) for i in range(cc_flat.shape[1])] for h in range(cc_flat.shape[0])]

    # Paired difference tests pooling all samples together
    #wstat1, p1 = scipy.stats.wilcoxon(tau_max[:, 0, :].ravel(), tau_max[:, 1, :].ravel())
    #wstat2, p2 = scipy.stats.wilcoxon(mag[:, 0, :].ravel(), mag[:, 1, :].ravel())
    wstat1, p1 = (np.nan, np.nan)
    wstat2, p2 = (np.nan, np.nan)

    # Calculation of entropy using knn 
    pdb.set_trace()
    tau_h1 = [knn_entropy(tau_max[idx, 0][:, np.newaxis]) for idx in range(tau_max.shape[0])]
    tau_h2 = [knn_entropy(tau_max[idx, 1][:, np.newaxis]) for idx in range(tau_max.shape[0])]

    avg_mag1 = [np.mean(mag[h][0], axis=-1) for h in range(len(mag))]
    avg_mag2 = [np.mean(mag[h][1], axis=-1) for h in range(len(mag))]
    pdb.set_trace()
    avg_dyn_range1 = np.mean(dyn_range[:, 0, :], axis=-1)
    avg_dyn_range2 = np.mean(dyn_range[:, 1, :], axis=-1)

    # Should be (27,6)
    # print((np.argmax(tau_h1), np.argmax(avg_dyn_range2)))

    # Paired difference tests
    # > , < , <
    wstat3, p3 = scipy.stats.mannwhitneyu(tau_h1, tau_h2, alternative='greater')
    wstat4, p4 = scipy.stats.mannwhitneyu(avg_mag1, avg_mag2, alternative='less')
    wstat5, p5 = scipy.stats.mannwhitneyu(avg_dyn_range1, avg_dyn_range2, alternative='less')
    
    # wstat3, p3 = scipy.stats.wilcoxon(tau_h1, tau_h2, alternative='greater')
    # wstat4, p4 = scipy.stats.wilcoxon(avg_mag1, avg_mag2, alternative='less')
    # wstat5, p5 = scipy.stats.wilcoxon(avg_dyn_range1, avg_dyn_range2, alternative='less')
    return tau_max, mag, (tau_h1, tau_h2), (avg_mag1, avg_mag2), (wstat1, wstat2, wstat3, wstat4, wstat5), (p1, p2, p3, p4, p5)

def box_plots(method1, method2, tau_max, mag, dyn_range, tau_h, avg_mag, stats, p, path):

    tau_h1 = tau_h[0]
    tau_h2 = tau_h[1]

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
    # fig.savefig('%s/tau_hist.pdf' % path, bbox_inches='tight', pad_inches=0)

    # Plot the histograms
    fig, ax = plt.subplots(6, 5, figsize=(25, 30))
    for i in range(28):
        a = ax[np.unravel_index(i, (6, 5))]
        a.hist(mag[i, 0], alpha=0.5)
        a.hist(mag[i, 1], alpha=0.5)
        a.set_title('FCCA h: %.3f, PCA h:%.3f' % (np.mean(mag[i, 0]), np.mean(mag[i, 1])))
        a.legend(['FCCA', 'PCA'])

    fig.suptitle('Peak cross-cov magnitude, %s vs. %s' % (method1, method2))
    # fig.savefig('%s/mag_hist.pdf' % path, bbox_inches='tight', pad_inches=0)

    # avg_mag1 = np.mean(mag[:, 0, :], axis=-1)
    # avg_mag2 = np.mean(mag[:, 1, :], axis=-1)
    avg_mag1 = avg_mag[0]
    avg_mag2 = avg_mag[1]

    avg_dyn_range1 = np.mean(dyn_range[:, 0, :], axis=-1)
    avg_dyn_range2 = np.mean(dyn_range[:, 1, :], axis=-1)

    # Boxplots
    fig, ax = plt.subplots(1, 2, figsize=(3.67, 4))

    medianprops = dict(linewidth=1, color='b')
    whiskerprops = dict(linewidth=0)
    # Plot dynamic arange, averag magnitude, and then entropy in order

    # Instead of PI, make boxplots of the dynamimc range (Z-scored)
    # bplot1 = ax[0].boxplot([avg_dyn_range1, avg_dyn_range2], patch_artist=True, medianprops=medianprops, 
    #                        notch=False, showfliers=False, vert=True, whiskerprops=whiskerprops, showcaps=False)
    bplot2 = ax[0].boxplot([avg_mag1, avg_mag2], patch_artist=True, medianprops=medianprops, notch=False, showfliers=False, 
                           vert=True, whiskerprops=whiskerprops, showcaps=False)
    bplot3 = ax[1].boxplot([tau_h1, tau_h2], patch_artist=True, medianprops=medianprops, notch=False, 
                           showfliers=False, vert=True, whiskerprops=whiskerprops, showcaps=False)

    method1 = 'FBC'
    method2 = 'FFC'
    # ax[0].set_xticklabels([method1, method2], rotation=45)
    # ax[1].set_xticklabels([method1, method2], rotation=45)
    # ax[2].set_xticklabels([method1, method2], rotation=45)
    ax[1].set_ylim([-30, 0])
    ax[1].set_yticks([-30, -15, 0])
    ax[0].set_ylim([0.10, 0.3])
    ax[0].set_yticks([0.10, 0.2, 0.3])
    # ax[1].set_xticks([15, 30, 45])
    #ax[0].set_title(r'$\tau$' + '-entropy, p=%f' % p[2], fontsize=10)
    #ax[1].set_title('Avg. magnitude, stat: p=%f' % p[3], fontsize=10)
    
    # Get the minimum significance level consistent with a multiple comparisons test
    pvec = np.sort(p[2:4])
    a1 = pvec[0] * 2
    a2 = pvec[1]
    print([a1, a2])
    # ax[0].set_title('*****', fontsize=10)
    # ax[1].set_title('*****', fontsize=10)
    # ax[2].set_title('*****', fontsize=10)
    
    # ax[0].set_ylabel('Avg. Dynamic Range', fontsize=16)
    ax[0].set_ylabel('Average Peak C.C.', fontsize=16)
    ax[1].set_ylabel('Peak C.C. Time Entropy', fontsize=16)
    for a in ax:
        a.tick_params(axis='both', labelsize=14)


    # fill with colors
    colors = ['red', 'black']
    for bplot in (bplot2, bplot3):
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight', pad_inches=0.1)
    
    # Additional plot for the trant - take the  paired differences of the average peak cross-correlation
    # fig, ax = plt.subplots(1, 1, figsize=(4, 1))
    # avg_mag1 = np.mean(mag[:, 0, :], axis=-1)
    # avg_mag2 = np.mean(mag[:, 1, :], axis=-1)
    # diff = avg_mag2 - avg_mag1
    # bplot = ax.boxplot(diff, patch_artist=True, medianprops=medianprops, notch=True, showfliers=False, vert=False)
    # for patch in bplot['boxes']:
    #     patch.set_facecolor('blue')
    #     patch.set_alpha(0.75)

    # ax.set_xlim([-0.05, 0.2])
    # ax.set_xticks([0.0, 0.2])
    # ax.set_yticks([])
    # fig.tight_layout()
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # #ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    # fig.savefig('%s/boxplot_diff_for_grant.pdf' % path, bbox_inches='tight', pad_inches=0.25)

if __name__ == '__main__':

    # Where to save?
    if len(sys.argv) > 1:
        figpath = sys.argv[1]
    else:
        figpath = '/home/akumar/nse/neural_control/figs/loco_indy_merge'

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

    # # Load dimreduc_dfs
    with open('/mnt/Secondary/data/postprocessed/indy_decoding_df2.dat', 'rb') as f:
        dimreduc_df = pd.DataFrame(pickle.load(f))

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

    dimreduc_df = pd.concat([dimreduc_df, loco_df])

    method1 = 'FCCA'
    method2 = 'PCA'

    # # Get top neurons as a function of FBC fraction threshold, and keep track of the test statistic and p-value as a function
    # # of this threshold, and magnitude of the effect
    fbc_fraction = np.linspace(0.5, 0.95, 25)[::-1]
    corrected_pvals = np.zeros((fbc_fraction.size, 2))
    test_stats = np.zeros((fbc_fraction.size, 2))
    effect = np.zeros((fbc_fraction.size, 2))
    for i, fbcf in enumerate(fbc_fraction):
        top_neurons_df, _ = get_top_neurons(dimreduc_df, fraction_cutoff=fbcf, method1=method1, method2=method2, n=10, pairwise_exclude=True)
        # Cross-covariance stuff
        cross_covs, cross_covs01, dyn_range = cross_cov_calc(top_neurons_df)
        tau_max, mag, tau_h, avg_mag, stats, p = statistics(cross_covs, cross_covs01, dyn_range)

        pvec = np.sort(p[2:4])
        order_ = np.argsort(p[2:4])
        a1 = pvec[0] * 2
        a2 = pvec[1]
        corrected_pvals[i, order_[0]] = a1
        corrected_pvals[i, order_[1]] = a2
        test_stats[i, 0] = stats[2]
        test_stats[i, 1] = stats[3]

        effect[i, 0] = np.median(np.array(tau_h[0]) - np.array(tau_h[1]))
        effect[i, 1] = np.median(np.array(avg_mag[0]) - np.array(avg_mag[1]))

    with open('%s/psth_diagnostic.dat' % figpath, 'wb') as f:
        pickle.dump({'fbc_fraction': fbc_fraction, 'corrected_pvals': corrected_pvals, 'test_stats': test_stats, 'effect': effect}, f)

    with open('%s/psth_diagnostic.dat' % figpath, 'rb') as f:
        results = pickle.load(f)

    fbc_fraction = results['fbc_fraction']
    corrected_pvals = results['corrected_pvals']
    test_stats = results['test_stats']
    effect = results['effect']

    # Make a plot of these quantities=
    fig, ax = plt.subplots(2, 3, figsize=(12, 8))
    ax[0, 0].plot(fbc_fraction, np.log10(corrected_pvals[:, 0]), 'b')
    ax[1, 0].plot(fbc_fraction, np.log10(corrected_pvals[:, 1]), 'b')
    for j in range(2):
        # Add horizontal levels at the [1e-5, 1e-4, 1e-3, 5e-2] levels:
        ax[j, 0].hlines(np.log10([1e-5, 1e-4, 1e-3, 5e-2]), fbc_fraction[0], fbc_fraction[-1], colors='k', linestyles='dashed', alpha=0.5)
        ax[j, 0].set_ylabel('Corrected p-value', fontsize=14)
        ax[j, 0].set_xlabel('FBC fraction threshold', fontsize=14)
    
    ax[0, 0].set_title('Peak c.c. Entropy', fontsize=16)
    ax[1, 0].set_title('Average Peak C.C.', fontsize=16)

    ax[0, 1].plot(fbc_fraction, test_stats[:, 0], 'b')
    ax[1, 1].plot(fbc_fraction, test_stats[:, 1], 'b')
    for j in range(2):
        ax[j, 1].set_ylabel('WSRT Test Statistic', fontsize=14)
        ax[j, 1].set_xlabel('FBC fraction threshold', fontsize=14)

    ax[0, 1].set_title('Peak c.c. Entropy', fontsize=16)
    ax[1, 1].set_title('Average Peak C.C.', fontsize=16)

    ax[0, 2].plot(fbc_fraction, effect[:, 0], 'b')
    ax[1, 2].plot(fbc_fraction, effect[:, 1], 'b')
    for j in range(2):
        ax[j, 2].set_title('Effect Size', fontsize=16)
        ax[j, 2].set_xlabel('FBC fraction threshold', fontsize=14)

    ax[0, 2].set_ylabel('Median Peak c.c. Entropy Diff.', fontsize=14)
    ax[1, 2].set_ylabel('Median Avg. Peak C.C. Diff', fontsize=14)

    fig.tight_layout()
    fig.savefig('%s/psth_diagnostic.pdf' % figpath, bbox_inches='tight', pad_inches=0)

    # heatmap_plot(top_neurons_df, figpath)
    # Plot PSTH
    # PSTH_plot(top_neurons_df, figpath)

    # data_files = np.unique(dimreduc_df['data_file'].values)
    # Print out data files corresponding to tuple ((np.argmax(tau_h1), np.argmax(avg_mag2))) for use in su_figs
    # print((data_files[27], data_files[6 ]))

    # For plots, we take cutoff fraction 0.5 and 0.9
    # top_neurons_df, _ = get_top_neurons(dimreduc_df, fraction_cutoff=0.5, method1=method1, method2=method2, n=10, pairwise_exclude=True)
    # # Cross-covariance stuff
    # cross_covs, cross_covs01, dyn_range = cross_cov_calc(top_neurons_df)
    # tau_max, mag, tau_h, avg_mag, stats, p = statistics(cross_covs, cross_covs01, dyn_range)
    # box_plots(method1, method2, tau_max, mag, dyn_range, tau_h, avg_mag, stats, p, '%s/psth_boxplots_q50.pdf' % figpath)
    
    # top_neurons_df, _ = get_top_neurons(dimreduc_df, fraction_cutoff=0.5, method1=method1, method2=method2, n=10, pairwise_exclude=True)
    # # Cross-covariance stuff
    # cross_covs, cross_covs01, dyn_range = cross_cov_calc(top_neurons_df)
    # tau_max, mag, tau_h, avg_mag, stats, p = statistics(cross_covs, cross_covs01, dyn_range)
    # box_plots(method1, method2, tau_max, mag, dyn_range, tau_h, avg_mag, stats, p, '%s/psth_boxplots_q90.pdf' % figpath)
