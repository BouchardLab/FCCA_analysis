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

from psth_ilmerge import get_top_neurons

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
    make_scatter = True
    make_psth = False
    make_hist = False

    data_path = '/mnt/Secondary/data/sabes'
    T = 30
    n = 10
    bin_width = 50

    DIM=6

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
    dimreduc_df = pd.concat([indy_df, loco_df])

    top_neurons_df, loadings_df = get_top_neurons(dimreduc_df, method1='FCCA', method2='PCA', n=10, pairwise_exclude=False, data_path=data_path, T=T, bin_width=bin_width)

    if make_scatter:
        # Re-scatter with the top neurons highlighted
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

        #df_ = apply_df_filters(loadings_df, dim=6)
        df_ = loadings_df

        x1 = df_['FCCA_loadings'].values
        x2 = df_['PCA_loadings'].values
        
        ax.scatter(np.log10(x1), np.log10(x2), alpha=0.25, color='#a3a3a3', s=15)

        for i in range(top_neurons_df.shape[0]):
            idxs1 = top_neurons_df.iloc[i]['top_neurons'][0, :]
            idxs2 = top_neurons_df.iloc[i]['top_neurons'][1, :]

            # Force pairwise exclusion here
            idxs1_ = set(idxs1).difference(set(idxs2))
            idxs2_ = set(idxs2).difference(set(idxs1))
            idxs1 = np.array(list(idxs1_))
            idxs2 = np.array(list(idxs2_))

            x = []
            y = []

            for j in range(len(idxs1)):
                d = apply_df_filters(df_, data_file=top_neurons_df.iloc[i]['data_file'], nidx=idxs1[j])
                assert(d.shape[0] == 1)
                x.append(d.iloc[0]['FCCA_loadings'])
                y.append(d.iloc[0]['PCA_loadings'])
            ax.scatter(np.log10(x), np.log10(y), color=(1, 0, 0, 0.5), edgecolors=(0, 0, 0, 0.5), s=15)

            x = []
            y = []
            for j in range(len(idxs1)):
                d = apply_df_filters(df_, data_file=top_neurons_df.iloc[i]['data_file'], nidx=idxs2[j])
                assert(d.shape[0] == 1)
                x.append(d.iloc[0]['FCCA_loadings'])
                y.append(d.iloc[0]['PCA_loadings'])
            ax.scatter(np.log10(x), np.log10(y), color=(0, 0, 0, 0.2), edgecolors=(0, 0, 0, 0.5), s=15)

        ax.set_xlim([-5, 0.1])
        ax.set_ylim([-5, 0.1])
        ax.set_xlabel('Log FCCA Importance Score', fontsize=14)
        ax.set_ylabel('Log PCA Importance Score', fontsize=14)
        ax.tick_params(axis='both', labelsize=12)

        # Annotate with the spearman-r
        r = scipy.stats.spearmanr(x1, x2)[0]

        # What is the spearman correlation in the intersection of the upper quartiles?
        # idxs1 = np.argwhere(x1 > q1_fca)[:, 0]
        # idxs2 = np.argwhere(x2 > q1_pca)[:, 0]
        # intersct = np.array(list(set(idxs1).intersection(set(idxs2))))

        r2 = scipy.stats.spearmanr(x1, x2)[0]
        ax.annotate('Spearman ' + r'$\rho$' +'=%.2f' % r, (-4.8, -4.5), fontsize=14)
        # ax.annotate('Upper-quartile r=%.2f' % r2, (-4.8, -0.5), fontsize=14)
        fig.savefig('%s/FCAPCAscatter.pdf' % figpath, bbox_inches='tight', pad_inches=0)

    # Next: single neuron traces

    # Selectively plot the traces from respective recording sessions that best emphasize the point being made
    # These choices were made after manually inspecting statistics in psth_ilmerge.py    

    if make_psth:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        #data_files = np.unique(top_neurons_df['data_file'].values)
        #data_file = data_files[4]
        data_file_fca ='indy_20160624_03.mat'
        data_file_pca ='indy_20160930_02.mat'

        df_ = apply_df_filters(top_neurons_df, data_file=data_file_pca)
        data_path = '/mnt/Secondary/data/sabes'

        dat = load_sabes('%s/%s' % (data_path, data_file_pca), boxcox=None, high_pass=False)
        dat_segment = reach_segment_sabes(dat, start_time=start_times[data_file_pca.split('.mat')[0]])

        T = 30
        t = np.array([t_[1] - t_[0] for t_ in dat_segment['transition_times']])
        valid_transitions = np.arange(t.size)[t >= T]

        # (Bin size 50 ms)
        time = 50 * np.arange(T)
        n = 10
        for j in range(10):
            tn = df_.iloc[0]['top_neurons'][1, j]    
#            print('PCA tn': tn)
            x_ = np.array([dat['spike_rates'][0, dat_segment['transition_times'][idx][0]:dat_segment['transition_times'][idx][0] + T, tn] 
                        for idx in valid_transitions])

            x_ = StandardScaler().fit_transform(x_.T).T
            x_ = gaussian_filter1d(x_, sigma=2)
            x_ = np.mean(x_, axis=0)

            h1 = ax.plot(time, x_, 'k', alpha=0.5, linewidth=2.5)

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
        ax.set_yticks([-0.5, 0, 0.5])
        ax.tick_params(axis='both', labelsize=12)
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.xaxis.set_label_coords(1.05, 0.56)
        ax.set_ylabel('Z-scored Response', fontsize=12)
        #ax.legend([h1], ['PCA'])
        #ax.set_title('Top PCA units', fontsize=14)

        fig.savefig('%s/topPCApsth.pdf' % figpath, bbox_inches='tight', pad_inches=0)

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        df_ = apply_df_filters(top_neurons_df, data_file=data_file_fca)
        dat = load_sabes('%s/%s' % (data_path, data_file_fca), boxcox=None, high_pass=False)
        dat_segment = reach_segment_sabes(dat, start_time=start_times[data_file_fca.split('.mat')[0]])

        T = 30
        t = np.array([t_[1] - t_[0] for t_ in dat_segment['transition_times']])
        valid_transitions = np.arange(t.size)[t >= T]

        # (Bin size 50 ms)
        time = 50 * np.arange(T)

        for j in range(n):
            tn = df_.iloc[0]['top_neurons'][0, j]    
#            print('FCCA tn': tn)
            x_ = np.array([dat['spike_rates'][0, dat_segment['transition_times'][idx][0]:dat_segment['transition_times'][idx][0] + T, tn] 
                        for idx in valid_transitions])
            
            x_ = StandardScaler().fit_transform(x_.T).T
            x_ = gaussian_filter1d(x_, sigma=2)
            x_ = np.mean(x_, axis=0)
            ax.plot(time, x_, 'r', alpha=0.5, linewidth=2.5)

        #ax.spines['left'].set_position('center')
        ax.spines['bottom'].set_position('center')

        # Eliminate upper and right axes
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')

        # Show ticks in the left and lower axes only
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        ax.set_xticks([0, 1500])
        ax.set_yticks([-0.5, 0, 0.5])
        ax.set_xticklabels([])
        ax.tick_params(axis='both', labelsize=12)

        ax.set_xlabel('Time (s)', fontsize=12)
        ax.xaxis.set_label_coords(1.05, 0.56)
        ax.set_ylabel('Z-scored Response', fontsize=12)
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
        # linmodel = LinearRegression().fit(su_r[])

        # linmodel = LinearRegression().fit()


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
        #         ax.text(rect.get_x() + rect.get_width()/2, np.sign(rect.get_height()) * (np.abs(rect.    top_neurons_df = pd.DataFrame(top_neurons_l)
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

        fig.savefig('%s/su_spearman_d%d.pdf' % (figpath, DIM), bbox_inches='tight', pad_inches=0)