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
    make_scatter = False
    make_psth = False
    make_hist = True

    # Where to save?
    if len(sys.argv) > 1:
        figpath = sys.argv[1]
    else:
        figpath = '/home/akumar/nse/neural_control/figs/loco_indy_merge'

    #dframe = '/home/akumar/nse/neural_control/data/indy_decoding_marginal.dat'
    dframe_indy = '/mnt/Secondary/data/postprocessed/indy_decoding_df2.dat'
    dframe_loco = '/mnt/Secondary/data/postprocessed/loco_decoding_df.dat'

    print('Using dframes %s, %s' % (dframe_indy, dframe_loco))

    with open(dframe_indy, 'rb') as f:
        rl = pickle.load(f)
    indy_df = pd.DataFrame(rl)

    with open(dframe_loco, 'rb') as f:
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

    DIM = 6

    # Try the raw leverage scores instead
    loadings_l = []
    indy_data_files = np.unique(indy_df['data_file'].values)
    for i, data_file in tqdm(enumerate(indy_data_files)):
        loadings = []
        for dimreduc_method in ['LQGCA', 'PCA']:
            loadings_fold = []
            for fold_idx in range(5):  
                df_ = apply_df_filters(indy_df, data_file=data_file, fold_idx=fold_idx, dim=DIM, dimreduc_method=dimreduc_method)
                if dimreduc_method == 'LQGCA':
                    df_ = apply_df_filters(df_, dimreduc_args={'T': 3, 'loss_type': 'trace', 'n_init': 10})
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

    loco_data_files = np.unique(loco_df['data_file'].values)
    for i, data_file in tqdm(enumerate(loco_data_files)):
        loadings = []
        for dimreduc_method in ['LQGCA', 'PCA']:
            loadings_fold = []
            for fold_idx in range(5):  
                df_ = apply_df_filters(loco_df, data_file=data_file, fold_idx=fold_idx, dim=DIM, dimreduc_method=dimreduc_method)
                if dimreduc_method == 'LQGCA':
                    df_ = apply_df_filters(df_, dimreduc_args={'T': 3, 'loss_type': 'trace', 'n_init': 10})
                assert(df_.shape[0] == 1)
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

    # Combine list of data files into 1
    data_files = np.concatenate([indy_data_files, loco_data_files])

    # For each data file, find the top 5 neurons that are high in one method but low in all others
    top_neurons_l = []
    n = 10
    for i, data_file in tqdm(enumerate(data_files)):
        df_ = apply_df_filters(loadings_df, data_file=data_file)
        # DCA_ordering = np.argsort(df_['DCA_loadings'].values)
        # KCA_ordering = np.argsort(df_['KCA_loadings'].values)
        FCCA_ordering = np.argsort(df_['FCCA_loadings'].values)
        PCA_ordering = np.argsort(df_['PCA_loadings'].values)
        
        rank_diffs = np.zeros((FCCA_ordering.size, 1))
        for j in range(df_.shape[0]):
            rank_diffs[j, 0] = list(FCCA_ordering).index(j) - list(PCA_ordering).index(j)

        # Find the top 5 neurons according to all pairwise high/low orderings
        top_neurons = np.zeros((2, n)).astype(int)

        # DCA_top = set([])
        # KCA_top = set([])
        FCCA_top = []
        PCA_top = []

        idx = 0
        while not np.all([len(x) >= n for x in [FCCA_top, PCA_top]]):
            idx += 1
            # Take neurons from the top ordering of each method. Disregard neurons that 
            # show up in all methods
            # top_DCA = DCA_ordering[-idx]
            top_FCCA = FCCA_ordering[-idx]
            top_PCA = PCA_ordering[-idx]

            if top_FCCA != top_PCA:
                if top_FCCA not in PCA_top:
                    FCCA_top.append(top_FCCA)
                if top_PCA not in FCCA_top:
                    PCA_top.append(top_PCA)
            else:
                continue

        top_neurons[0, :] = FCCA_top[0:n]
        top_neurons[1, :] = PCA_top[0:n] 

        top_neurons_l.append({'data_file':data_file, 'rank_diffs':rank_diffs, 'top_neurons': top_neurons}) 

    if make_scatter:
        # Re-scatter with the top neurons highlighted
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

        #df_ = apply_df_filters(loadings_df, dim=6)
        df_ = loadings_df

        x1 = df_['FCCA_loadings'].values
        x2 = df_['PCA_loadings'].values
        
        #x1idxs = np.arange(x1.size)[x1 > np.quantile(x1, 0.75)]
        q1_pca = np.quantile(x2, 0.75)
        q1_fca = np.quantile(x1, 0.75)

        # Plot vertical lines at the PCA quantile
        #ax.hlines(np.log10(q1_pca), -5, 0, color='k')
        #ax.vlines(np.log10(q1_fca), -5, 0, color='k')

        #x1 = x1[x1idxs]
        #x2 = x2[x1idxs]
        #x1 = x1[x1 > np.quantile(x1, 0.05)]
        #x2 = x2[x2 > np.quantile(x2, 0.05)]

        ax.scatter(np.log10(x1), np.log10(x2), alpha=0.25, color='#753530', s=15)

        for i in range(len(top_neurons_l)):
            idxs1 = top_neurons_l[i]['top_neurons'][0, :]
            idxs2 = top_neurons_l[i]['top_neurons'][1, :]
            x = []
            y = []
            for j in range(len(idxs1)):
                d = apply_df_filters(df_, data_file=top_neurons_l[i]['data_file'], nidx=idxs1[j])
                assert(d.shape[0] == 1)
                x.append(d.iloc[0]['FCCA_loadings'])
                y.append(d.iloc[0]['PCA_loadings'])
            ax.scatter(np.log10(x), np.log10(y), color=(1, 0, 0, 0.5), edgecolors=(0, 0, 0, 0.5), s=15)

            x = []
            y = []
            for j in range(len(idxs1)):
                d = apply_df_filters(df_, data_file=top_neurons_l[i]['data_file'], nidx=idxs2[j])
                assert(d.shape[0] == 1)
                x.append(d.iloc[0]['FCCA_loadings'])
                y.append(d.iloc[0]['PCA_loadings'])
            ax.scatter(np.log10(x), np.log10(y), color=(0, 0, 0, 0.2), edgecolors=(0, 0, 0, 0.5), s=15)

        ax.set_xlim([-5, 0.1])
        ax.set_ylim([-5, 0.1])
        ax.set_xlabel('Log FCCA Loading', fontsize=14)
        ax.set_ylabel('Log PCA Loading', fontsize=14)
        ax.tick_params(axis='both', labelsize=12)

        # Annotate with the spearman-r
        r = scipy.stats.spearmanr(x1, x2)[0]

        # What is the spearman correlation in the intersection of the upper quartiles?
        idxs1 = np.argwhere(x1 > q1_fca)[:, 0]
        idxs2 = np.argwhere(x2 > q1_pca)[:, 0]
        intersct = np.array(list(set(idxs1).intersection(set(idxs2))))

        r2 = scipy.stats.spearmanr(x1[intersct], x2[intersct])[0]
        ax.annotate('Spearman r=%.2f' % r, (-4.8, -4.5), fontsize=14)
        # ax.annotate('Upper-quartile r=%.2f' % r2, (-4.8, -0.5), fontsize=14)
        fig.savefig('%s/FCAPCAscatter.pdf' % figpath, bbox_inches='tight', pad_inches=0)

    # Next: single neuron traces

    top_neurons_df = pd.DataFrame(top_neurons_l)

    if make_psth:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        #data_files = np.unique(top_neurons_df['data_file'].values)
        #data_file = data_files[4]
        data_file='indy_20161006_02.mat'

        df_ = apply_df_filters(top_neurons_df, data_file=data_file)
        data_path = '/mnt/Secondary/data/sabes'

        dat = load_sabes('%s/%s' % (data_path, data_file), boxcox=None, high_pass=False)
        dat_segment = reach_segment_sabes(dat, start_time=start_times[data_file.split('.mat')[0]])

        T = 30
        t = np.array([t_[1] - t_[0] for t_ in dat_segment['transition_times']])
        valid_transitions = np.arange(t.size)[t >= T]

        # (Bin size 50 ms)
        time = 50 * np.arange(T)
        n = 10
        for j in range(10):
            tn = df_.iloc[0]['top_neurons'][1, j]    
            x_ = np.array([dat['spike_rates'][0, dat_segment['transition_times'][idx][0]:dat_segment['transition_times'][idx][0] + T, tn] 
                        for idx in valid_transitions])

            x_ = gaussian_filter1d(x_, sigma=2)
            x_ = StandardScaler().fit_transform(x_.T).T
            x_ = np.mean(x_, axis=0)

            h1 = ax.plot(time, x_, 'k', alpha=0.5)

        T = 30
        t = np.array([t_[1] - t_[0] for t_ in dat_segment['transition_times']])
        valid_transitions = np.arange(t.size)[t >= T]

        #ax.spines['left'].set_position('center')
        ax.spines['bottom'].set_position('center')

        # Eliminate upper and right axes
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')

        # Show ticks in the left and lower axes only
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        ax.set_xticks([0, 1500])
        ax.set_xticklabels([])
        ax.set_yticks([-1, 0, 1])
        ax.tick_params(axis='both', labelsize=12)
        #ax.set_xlabel('Time (s)', fontsize=12)
        ax.xaxis.set_label_coords(1.05, 0.56)
        #ax.set_ylabel('Z-scored Response', fontsize=12)
        ax.legend([h1], ['PCA'])
        #ax.set_title('Top PCA units', fontsize=14)

        fig.savefig('%s/topPCApsth.pdf' % figpath, bbox_inches='tight', pad_inches=0)

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        df_ = apply_df_filters(top_neurons_df, data_file=data_file)
        dat = load_sabes('%s/%s' % (data_path, data_file), boxcox=None, high_pass=False)
        dat_segment = reach_segment_sabes(dat, start_time=start_times[data_file.split('.mat')[0]])

        T = 30
        t = np.array([t_[1] - t_[0] for t_ in dat_segment['transition_times']])
        valid_transitions = np.arange(t.size)[t >= T]

        # (Bin size 50 ms)
        time = 40 * np.arange(T)

        for j in range(n):
            tn = df_.iloc[0]['top_neurons'][0, j]    
            x_ = np.array([dat['spike_rates'][0, dat_segment['transition_times'][idx][0]:dat_segment['transition_times'][idx][0] + T, tn] 
                        for idx in valid_transitions])
            
            x_ = gaussian_filter1d(x_, sigma=2)
            x_ = StandardScaler().fit_transform(x_.T).T
            x_ = np.mean(x_, axis=0)

            ax.plot(time, x_, 'r', alpha=0.5)

        #ax.spines['left'].set_position('center')
        ax.spines['bottom'].set_position('center')

        # Eliminate upper and right axes
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')

        # Show ticks in the left and lower axes only
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        ax.set_xticks([0, 1500])
        ax.set_yticks([-1, 0, 1])
        ax.set_xticklabels([])
        ax.tick_params(axis='both', labelsize=12)

        #ax.set_xlabel('Time (s)', fontsize=12)
        ax.xaxis.set_label_coords(1.05, 0.56)
        #ax.set_ylabel('Z-scored Response', fontsize=12)
        #ax.set_title('Top FCCA units', fontsize=14)

        fig.savefig('%s/topFCCApsth.pdf' % figpath, bbox_inches='tight', pad_inches=0)

    ########################## Histogram ########################################
    #############################################################################
    if make_hist:

        with open('/mnt/Secondary/data/postprocessed/sabes_su_calcs.dat', 'rb') as f:
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
                    if stat == 'orientation_tuning':
                        df_ = apply_df_filters(odf, file=data_file, tau=4)
                    else:
                        df_ = apply_df_filters(sabes_su_df, data_file=data_file)
                    carray_[j, k] = get_scalar(df_, stat, nidx)
            carray.append(carray_)

        su_r = np.zeros((len(carray), 2, carray[0].shape[1]))
        keys = ['FCCA_loadings', 'PCA_loadings']
        for i in range(len(carray)):
            for j in range(2):
                for k in range(carray[0].shape[1]):
                    df = apply_df_filters(itrim_df, data_file=data_files[i])

                    x1 = df[keys[j]].values
                    x2 = carray[i][:, k]
                    su_r[i, j, k] = scipy.stats.spearmanr(x1, x2)[0]

        fig, ax = plt.subplots(figsize=(5, 5),)

        # Prior to averaging, run tests
        _, p1 = scipy.stats.wilcoxon(su_r[:, 0, 0], su_r[:, 1, 0])
        _, p2 = scipy.stats.wilcoxon(su_r[:, 0, 1], su_r[:, 1, 1])
        _, p3 = scipy.stats.wilcoxon(su_r[:, 0, 2], su_r[:, 1, 2])
        _, p4 = scipy.stats.wilcoxon(su_r[:, 0, 3], su_r[:, 1, 3])

        print(p1)
        print(p2)
        print(p3)
        print(p4)

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
        for rect in bars: 
            if rect.get_height() > 0:
                ax.text(rect.get_x() + rect.get_width()/2, np.sign(rect.get_height()) * (np.abs(rect.get_height()) + 0.075), '%.2f' % rect.get_height(),
                        ha='center', va='bottom', fontsize=10)
            else:
                ax.text(rect.get_x() + rect.get_width()/2, np.sign(rect.get_height()) * (np.abs(rect.get_height()) + 0.11), '%.2f' % rect.get_height(),
                        ha='center', va='bottom', fontsize=10)

        # Add significance tests
        ax.text(0.5, -0.47, '****', ha='center')
        ax.text(3.5, 0.7, '**', ha='center')
        ax.text(6.5, 0.6, '***', ha='center')
        ax.text(9.5, 0.76, '****', ha='center')


        ax.set_ylim([-0.5, 1.1])
        ax.set_xticks([0.5, 3.5, 6.5, 9.5])

        ax.set_xticklabels(['S.U. Variance', 'Autocorr. time', 'Decoding Weights', 'S.U. Enc. ' + r'$r^2$'], rotation=30, fontsize=12, ha='right')

        ax.tick_params(axis='y', labelsize=12)

        # Manual creation of legend
        colors = ['r', 'k']
        handles = [plt.Rectangle((0,0),1,1, color=c, alpha=0.65) for c in colors]
        labels = ['FCCA', 'PCA']
        ax.legend(handles, labels, loc='lower right')

        ax.set_ylabel('Spearman Correlation', fontsize=14)
        ax.set_yticks([-0.5, 0, 0.5, 1.])
        fig.savefig('%s/su_spearman_d%d.pdf' % (figpath, DIM), bbox_inches='tight', pad_inches=0)