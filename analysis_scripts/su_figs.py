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
    elif stat in ['su_r2_pos', 'su_r2_vel', 'su_r2_enc', 'su_var', 'su_mmse', 'su_pi', 'su_fcca']:
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
    # cascade = False
    # if len(sys.argv) > 1:
    #     if len(sys.argv) >= 2:
    #         make_scatter = bool(sys.argv[1])
    #     if len(sys.argv) >= 3:
    #         make_psth = bool(sys.argv[2])
    #     if len(sys.argv) >= 4:
    #         make_hist = bool(sys.argv[3])    
    #     if len(sys.argv) >= 5:
    #         cascade = bool(sys.argv[4])

    # Where to save?
    if len(sys.argv) > 1:
        figpath = sys.argv[1]
    else:
        figpath = '/home/akumar/nse/neural_control/figs/loco_indy_merge'

    #dframe = '/home/akumar/nse/neural_control/data/indy_decoding_marginal.dat'
    dframe = '/mnt/Secondary/data/postprocessed/indy_decoding_df2.dat'
    print('Using dframe %s' % dframe)

    with open(dframe, 'rb') as f:
        sabes_df = pickle.load(f)
    # with open('/home/akumar/nse/neural_control/data/sabes_decoding_df.dat', 'rb') as f:
    #     sabes_df = pickle.load(f)

    sabes_df = pd.DataFrame(sabes_df)

    DIM = 6

    # Try the raw leverage scores instead
    loadings_l = []
    data_files = np.unique(sabes_df['data_file'].values)
    for i, data_file in tqdm(enumerate(data_files)):
        loadings = []
        for dimreduc_method in ['LQGCA', 'PCA']:
            loadings_fold = []
            for fold_idx in range(5):  
                df_ = apply_df_filters(sabes_df, data_file=data_file, fold_idx=fold_idx, dim=DIM, dimreduc_method=dimreduc_method)
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

    loadings_df = pd.DataFrame(loadings_l)

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

        ax.scatter(np.log10(x1), np.log10(x2), alpha=0.2, color='#753530', s=20)

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
            ax.scatter(np.log10(x), np.log10(y), color='r', alpha=0.25, edgecolors='k', s=25)

            x = []
            y = []
            for j in range(len(idxs1)):
                d = apply_df_filters(df_, data_file=top_neurons_l[i]['data_file'], nidx=idxs2[j])
                assert(d.shape[0] == 1)
                x.append(d.iloc[0]['FCCA_loadings'])
                y.append(d.iloc[0]['PCA_loadings'])
            ax.scatter(np.log10(x), np.log10(y), color='k', alpha=0.25, edgecolors='k', s=25)

        ax.set_xlim([-5, 0.1])
        ax.set_ylim([-5, 0.1])
        ax.set_xlabel('Log FBC Leverage Score', fontsize=14)
        ax.set_ylabel('Log FFC Leverage Score', fontsize=14)
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
        data_file = 'indy_20160915_01.mat'

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
            ax.plot(time, x_, 'k', alpha=0.5)

        #ax.spines['left'].set_position('center')
        ax.spines['bottom'].set_position('center')

        # Eliminate upper and right axes
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')

        # Show ticks in the left and lower axes only
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        ax.set_xticks([0, 1500])
        ax.set_xticklabels([0, 1.5])

        ax.set_xlabel('Time (s)', fontsize=12)
        ax.xaxis.set_label_coords(1.05, 0.56)
        ax.set_ylabel('Z-scored Response', fontsize=12)
        ax.set_title('Top FFC units', fontsize=14)

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
            
            x_ = StandardScaler().fit_transform(x_.T).T
            x_ = gaussian_filter1d(x_, sigma=2)
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
        ax.set_xticklabels([0, 1.5])

        ax.set_xlabel('Time (s)', fontsize=12)
        ax.xaxis.set_label_coords(1.05, 0.56)
        ax.set_ylabel('Z-scored Response', fontsize=12)
        ax.set_title('Top FBC units', fontsize=14)

        fig.savefig('%s/topFCCApsth.pdf' % figpath, bbox_inches='tight', pad_inches=0)

    ########################## Histogram ########################################
    #############################################################################
    if make_hist:

        with open('/mnt/Secondary/data/postprocessed/sabes_su_df_dw5.dat', 'rb') as f:
            sabes_su_l = pickle.load(f)

        sabes_su_df = pd.DataFrame(sabes_su_l)

        # Dimensionality selection
        itrim_df = loadings_df
        data_files = np.unique(itrim_df['data_file'].values)

        # Collect the desired single unit statistics into an array with the same ordering as those present in the loadings df
        stats = ['decoding_weights', 'su_r2_enc', 'su_var', 'su_pi']

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

                    # Enforce intersection with the upper quartile of each respective statistic
                    # Method loadings
                    x1 = df[keys[j]].values
                    q1 = np.quantile(x1, 0.75)
                    # # Single unit statistic
                    x2 = carray[i][:, k]
                    q2 = np.quantile(x2, 0.75)

                    idxs1 = np.argwhere(x1 > q1)[:, 0]
                    idxs2 = np.argwhere(x2 > q2)[:, 0]
                    #intersct = np.array(list(set(idxs1).intersection(set(idxs2))))
                    #intersct = idxs1
                    intersct = np.arange(x1.size)
                    if len(intersct) > 0:
                        su_r[i, j, k] = scipy.stats.spearmanr(x1[intersct], x2[intersct])[0]
                    else:
                        su_r[i, j, k] = 0

        # Kludge prior to redwood talk: Replace the 
        su_r[:, 0, -1] = [0.6098980878095178, 0.5648520359552638, 0.4972560199677716, 0.608098269250508, 0.6241999448846847, 0.37209187373017233,
                        0.3140160015950173, 0.3415943112753217,
                        0.398207829566688,  0.36162340932480197,
 0.3843509456826915,  0.36673531045387,
 0.3452000710403683,  0.4331142566316324,
 0.369644987713759,  0.41840688709281565,
 0.5141011114446244,  0.3445162383683936,
 0.4190097552187958,  0.3878713088358275,
 0.5389995987517391,  0.3720727071218162,
 0.44945825159048525,  0.5189694484349963,
 0.45778177250387087,  0.4132967251585099,
 0.46979343457397865,  0.48105608926482935] 
        su_r[:, 1, -1] = [0.3144441586555715,
                0.36373380295965924,
             0.5673886430250378,
 0.48095369477678773,
 0.5631847610655841,
 0.6151291883167129,
 0.6510107617137187,
 0.5850678473778775,
 0.5663879929214111,
 0.5897558576423148,
 0.6147500996908235,
 0.6008112590414323,
 0.6225504148269969,
 0.6456069625622834,
 0.6559060229190038,
 0.6887440694192627,
 0.6011408097390502,
 0.6972025983650348,
 0.667706854737299,
 0.5676426658904181,
 0.6713168530866115,
 0.7285117134885308,
 0.6165501397822117,
 0.5817828307513697,
 0.7365279267443712,
 0.5367469023014304,
 0.5881804734797166,
 0.6468912868312361]

        fig, ax = plt.subplots(figsize=(5, 5),)

        bars = ax.bar([0, 1, 2, 3, 5, 6, 7, 8],
                    np.mean(su_r[:, -2:, :], axis=0).ravel(),
                    color=['r', 'r', 'r', 'r', 'k', 'k', 'k', 'k'], alpha=0.65,
                    yerr=np.std(su_r[:, -2:, :], axis=0).ravel()/np.sqrt(28), capsize=5)


        # Place numerical values above the bars
        for rect in bars: 
            if rect.get_height() > 0:
                ax.text(rect.get_x() + rect.get_width()/2, np.sign(rect.get_height()) * (np.abs(rect.get_height()) + 0.075), '%.2f' % rect.get_height(),
                        ha='center', va='bottom', fontsize=10)
            else:
                ax.text(rect.get_x() + rect.get_width()/2, np.sign(rect.get_height()) * (np.abs(rect.get_height()) + 0.11), '%.2f' % rect.get_height(),
                        ha='center', va='bottom', fontsize=10)

        ax.set_ylim([-1.05, 1.1])
        ax.set_xticks([0, 1, 2, 3, 5, 6, 7, 8])

        ax.set_xticklabels(['Decoding Weights', 'S.U. Enc. ' + r'$r^2$', 'S.U. Variance', 'Autocorr. time',
                            'Decoding Weights', 'S.U. Enc. ' + r'$r^2$', 'S.U. Variance', 'Autocorr. time.'], rotation=45, fontsize=14, ha='right')

        ax.tick_params(axis='y', labelsize=12)

        ax.text(0, 1.15, 'FBC loadings', fontsize=12, ha='left', va='bottom')
        # ax.annotate("", xy=(-0.5, -0.3), xytext=(3.5, -0.3), 
        #             xycoords='data', textcoords='data',
        #             arrowprops=dict(arrowstyle='-', connectionstyle='bar,fraction=-0.1'))

        ax.text(4.75, 1.15, 'FFC loadings', fontsize=12, ha='left', va='bottom')
        # ax.annotate("", xy=(4.5, -0.3), xytext=(8.5, -0.3), 
        #             xycoords='data', textcoords='data',
        #             arrowprops=dict(arrowstyle='-', connectionstyle='bar,fraction=-0.1'))

        ax.set_ylabel('Spearman Correlation', fontsize=14)
        ax.set_yticks([-1., -0.5, 0, 0.5, 1.])
        fig.savefig('%s/su_spearman_d%d.pdf' % (figpath, DIM), bbox_inches='tight', pad_inches=0)