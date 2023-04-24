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

def get_top_neurons(dimreduc_df, method1='FCCA', method2='PCA', n=10, pairwise_exclude=True, data_path=None, T=None, bin_width=None):

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
                    V = V[:, 0:2]        
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
    n = 10
    for i, data_file in tqdm(enumerate(data_files)):
        df_ = apply_df_filters(loadings_df, data_file=data_file)
        FCCA_ordering = np.argsort(df_['FCCA_loadings'].values)
        PCA_ordering = np.argsort(df_['PCA_loadings'].values)
        
        rank_diffs = np.zeros((PCA_ordering.size,))
        for j in range(df_.shape[0]):            
            rank_diffs[j] = list(FCCA_ordering).index(j) - list(PCA_ordering).index(j)

        # Find the top 5 neurons according to all pairwise high/low orderings
        top_neurons = np.zeros((2, n)).astype(int)

        # User selects which pairwise comparison is desired
        method_dict = {'PCA': PCA_ordering, 'FCCA':FCCA_ordering}

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

            if pairwise_exclude:
                if top1_ != top2_:
                    if top1_ not in top2:
                        top1.append(top1_)
                    if top2_ not in top1:
                        top2.append(top2_)
                else:
                    continue
            else:
                top1.append(top1_)
                top2.append(top2_)

        top_neurons[0, :] = top1[0:n]
        top_neurons[1, :] = top2[0:n]

        top_neurons_l.append({'data_file':data_file, 'rank_diffs':rank_diffs, 'top_neurons': top_neurons}) 
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
        dat = load_sabes('%s/%s' % (data_path, data_file), boxcox=None, high_pass=False, region='S1')
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

    ndf = 2
    fig, ax = plt.subplots(2 * 4, ndf, figsize=(4*ndf, 16))

    data_files = np.unique(top_neurons_df['data_file'].values)

    for h, data_file in enumerate(data_files):

        df_ = apply_df_filters(top_neurons_df, data_file=data_file)
        dat = load_sabes('%s/%s' % (data_path, data_file), boxcox=None, high_pass=False, region='S1')
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
                        ax[2 * (h//2) + i, h % 2].plot(time, x_, 'k', alpha=0.5)
                        ax[2 * (h//2) + i, h % 2].set_title(data_file)

                    if i == 1:
                        ax[2 * (h//2) + i, h % 2].plot(time, x_, 'r', alpha=0.5)
                        ax[2 * (h//2) + i, h % 2].set_title(data_file)
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

    dyn_range = np.zeros((len(data_files), 2, n))

    for h, data_file in enumerate(data_files):
        df_ = apply_df_filters(top_neurons_df, data_file=data_file)
        dat = load_sabes('%s/%s' % (data_path, data_file), boxcox=None, high_pass=False, region='S1')
        dat_segment = reach_segment_sabes(dat, data_file=data_file.split('.mat')[0])
        
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
                    cross_covs[h, i, j, k] = np.correlate(x[j], x[k], mode='same')/T
                    cross_covs_01[h, i, j, k] = np.correlate(x[j]/np.max(x[j]), x[k]/np.max(x[k]), mode='same')/T
                dyn_range[h, i, j] = np.max(x[j])

    return cross_covs, cross_covs_01, dyn_range

def statistics(cross_covs, cross_covs_01, dyn_range):
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

    avg_dyn_range1 = np.mean(dyn_range[:, 0, :], axis=-1)
    avg_dyn_range2 = np.mean(dyn_range[:, 1, :], axis=-1)

    # Should be (27,6)
    print((np.argmax(tau_h1), np.argmax(avg_dyn_range2)))

    # Paired difference tests
    # > , < , <
    wstat3, p3 = scipy.stats.wilcoxon(tau_h1, tau_h2, alternative='greater')
    wstat4, p4 = scipy.stats.wilcoxon(avg_mag1, avg_mag2, alternative='less')
    wstat5, p5 = scipy.stats.wilcoxon(avg_dyn_range1, avg_dyn_range2, alternative='less')

    return tau_max, mag, (wstat1, wstat2, wstat3, wstat4, wstat5), (p1, p2, p3, p4, p5)

def box_plots(method1, method2, tau_max, mag, dyn_range, stats, p, path):

    tau_h1 = [knn_entropy(tau_max[idx, 0, :][:, np.newaxis], k=5) for idx in range(tau_max.shape[0])]
    tau_h2 = [knn_entropy(tau_max[idx, 1, :][:, np.newaxis], k=5) for idx in range(tau_max.shape[0])]

    data_path = globals()['data_path']
    T = globals()['T']  
    n = globals()['n']
    bin_width = globals()['bin_width']

    # Plot the histograms
    fig, ax = plt.subplots(4, 2, figsize=(8, 16))
    for i in range(8):
        a = ax[np.unravel_index(i, (4, 2))]
        a.hist(tau_max[i, 0], alpha=0.5, bins=np.linspace(-1500, 1500, 25))
        a.hist(tau_max[i, 1], alpha=0.5)    

        # Title with entropy
        a.set_title('FCCA h: %.3f, PCA h:%.3f' % (tau_h1[i], tau_h2[i]))
        a.legend(['FCCA', 'PCA'])

    fig.suptitle('Peak cross-cov time, %s vs. %s' % (method1, method2))
    fig.savefig('%s/tau_hist.pdf' % path, bbox_inches='tight', pad_inches=0)

    # Plot the histograms
    fig, ax = plt.subplots(4, 2, figsize=(8, 16))
    for i in range(8):
        a = ax[np.unravel_index(i, (4, 2))]
        a.hist(mag[i, 0], alpha=0.5)
        a.hist(mag[i, 1], alpha=0.5)
        a.set_title('FCCA h: %.3f, PCA h:%.3f' % (np.mean(mag[i, 0]), np.mean(mag[i, 1])))
        a.legend(['FCCA', 'PCA'])

    fig.suptitle('Peak cross-cov magnitude, %s vs. %s' % (method1, method2))
    fig.savefig('%s/mag_hist.pdf' % path, bbox_inches='tight', pad_inches=0)

    avg_mag1 = np.mean(mag[:, 0, :], axis=-1)
    avg_mag2 = np.mean(mag[:, 1, :], axis=-1)

    avg_dyn_range1 = np.mean(dyn_range[:, 0, :], axis=-1)
    avg_dyn_range2 = np.mean(dyn_range[:, 1, :], axis=-1)

    # Boxplots
    fig, ax = plt.subplots(1, 3, figsize=(5, 4))

    medianprops = dict(linewidth=0)

    # Plot dynamic arange, averag magnitude, and then entropy in order

    # Instead of PI, make boxplots of the dynamimc range (Z-scored)
    bplot1 = ax[0].boxplot([avg_dyn_range1, avg_dyn_range2], patch_artist=True, medianprops=medianprops, notch=True, showfliers=False, vert=True)
    bplot2 = ax[1].boxplot([avg_mag1, avg_mag2], patch_artist=True, medianprops=medianprops, notch=True, showfliers=False, vert=True)
    bplot3 = ax[2].boxplot([tau_h1, tau_h2], patch_artist=True, medianprops=medianprops, notch=True, showfliers=False, vert=True)

    ax[0].set_xticklabels([method1, method2])
    ax[1].set_xticklabels([method1, method2])
    ax[2].set_xticklabels([method1, method2])

    ax[2].set_yticks([0, -15, -30])
    # ax[1].set_xticks([15, 30, 45])
    #ax[0].set_title(r'$\tau$' + '-entropy, p=%f' % p[2], fontsize=10)
    #ax[1].set_title('Avg. magnitude, stat: p=%f' % p[3], fontsize=10)
    
    # Get the minimum significance level consistent with a multiple comparisons test
    pvec = np.sort(p[2:])
    a1 = pvec[0] * 3
    a2 = pvec[1] * 2
    a3 = pvec[2]
    print(max([a1, a2, a3]))
    ax[0].set_title('*****', fontsize=10)
    ax[1].set_title('*****', fontsize=10)
    ax[2].set_title('*****', fontsize=10)
    
    ax[0].set_ylabel('Avg. Dynamic Range', fontsize=12)
    ax[1].set_ylabel('Average peak cross-corr.', fontsize=12)
    ax[2].set_ylabel('Entropy of peak cross-corr. times', fontsize=12)

    # fill with colors
    colors = ['red', 'black']
    for bplot in (bplot1, bplot2, bplot3):
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

    fig.tight_layout()
    fig.savefig('%s/boxplots_with_dynrange.pdf' % path, bbox_inches='tight', pad_inches=0)
    
if __name__ == '__main__':

    # Where to save?
    if len(sys.argv) > 1:
        figpath = sys.argv[1]
    else:
        figpath = '/home/akumar/nse/neural_control/figs/loco_indy_merge/psth_S1'

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

    # Load dimreduc_dfs
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

    loco_df = pd.DataFrame(result_list)
    indy_df = pd.DataFrame(rl2)
    loco_df = apply_df_filters(loco_df, data_file=good_loco_files, decoder_args=loco_df.iloc[0]['decoder_args'])
    indy_df = apply_df_filters(indy_df, decoder_args=indy_df.iloc[0]['decoder_args'])
    sabes_df = pd.concat([loco_df, indy_df])        
    loader_arg = {'bin_width':50, 'filter_fn':'none', 'filter_kwargs':{}, 'boxcox':0.5, 'spike_threshold':100, 'region':'S1'}
    sabes_df = apply_df_filters(sabes_df, loader_args=loader_arg)

    dimreduc_df = sabes_df

    method1 = 'FCCA'
    method2 = 'PCA'

    # Get top neurons
    top_neurons_df, _ = get_top_neurons(dimreduc_df, method1=method1, method2=method2, n=10, pairwise_exclude=True)
    # heatmap_plot(top_neurons_df, figpath)
    # Plot PSTH
    PSTH_plot(top_neurons_df, figpath)

    # Cross-covariance stuff
    cross_covs, cross_covs01, dyn_range = cross_cov_calc(top_neurons_df)
    tau_max, mag, stats, p = statistics(cross_covs, cross_covs01, dyn_range)


    data_files = np.unique(dimreduc_df['data_file'].values)
    # Print out data files corresponding to tuple ((np.argmax(tau_h1), np.argmax(avg_mag2))) for use in su_figs
    # print((data_files[27], data_files[6 ]))

    box_plots(method1, method2, tau_max, mag, dyn_range, stats, p, figpath)
    