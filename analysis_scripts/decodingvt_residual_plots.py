import pdb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy 
import time
import os
import glob
import sys
import pickle
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from scipy.ndimage import gaussian_filter1d

import itertools
from sklearn.model_selection import KFold
sys.path.append('/home/akumar/nse/neural_control')
from utils import apply_df_filters, calc_loadings, decorrelate
from loaders import load_sabes
from segmentation import reach_segment_sabes
from decoders import lr_decode_windowed, expand_state_space, decorrelate as decorrelate2

# Return the acceleration and velocity for each data file,
# do the decorrelation on a trial segmented basis, and then calculate the magnitude for plotting purposes
def decorrelate_acceleration(dvt_df, sabes_df):
    if not os.path.exists('acc_vel_tmp_residual.pkl'):
        data_idxs = np.unique(dvt_df['didx'].values)
        vel_all = []
        acc_all = []
        vel_all_decorr = []
        acc_all_decorr = []

        for i, data_idx in tqdm(enumerate(data_idxs)):

            data_file = np.unique(sabes_df['data_file'].values)[data_idx]
            dat = load_sabes('/mnt/Secondary/data/sabes/%s' % data_file)
            dat = reach_segment_sabes(dat, data_file=data_file.split('/')[-1].split('.mat')[0])

            # Intersection

            X = dat['spike_rates'].squeeze()
            Z = dat['behavior'] 

            Z, X = expand_state_space([Z], [X], True, True)
            X = X[0]
            Z = Z[0]

            d = apply_df_filters(dvt_df, didx=data_idx)
            error_filter = d.iloc[0]['error_filter']
            reach_filter = d.iloc[0]['reach_filter']
            window_filter = d.iloc[0]['window_filter']    

            transition_times = np.array(dat['transition_times'])[error_filter][reach_filter][window_filter]
            transition_times = np.array([(t[0] - 2, t[1] - 2) for t in transition_times])

            # Shift all transition times by 2


            # Decorrelate
            # To ensure consistency with the usual way we visualize acc/vel traces, we take the transition times to be 
            # tt[0] - t + tt[0] + 30
            transition_times_uniform = [(tt[0], tt[0] + 30) for tt in transition_times]
            vel_residuals, acc_residuals = decorrelate2(Z, 'trialized', embed=False, transition_times=transition_times_uniform)

            # Calculate norms
            vel_all.append([])
            acc_all.append([])
            vel_all_decorr.append([])
            acc_all_decorr.append([])
            for j in range(len(vel_residuals)):
                tt = transition_times[j]
                vel = np.linalg.norm(Z[tt[0] - 5:tt[0] + 30, 2:4], axis=1)/(1e3 * 0.05)
                acc = np.linalg.norm(Z[tt[0] - 5:tt[0] + 30, 4:6], axis=1)/(1e3 * 0.05 * 0.05)
                vel_all[i].append(vel)
                acc_all[i].append(acc)
                
                vel = np.linalg.norm(vel_residuals[j], axis=1)/(1e3 * 0.05)
                acc = np.linalg.norm(acc_residuals[j], axis=1)/(1e3 * 0.05 * 0.05)
                vel_all_decorr[i].append(vel)
                acc_all_decorr[i].append(acc)

        # Save away
        with open('acc_vel_tmp_residual.pkl', 'wb') as f:
            f.write(pickle.dumps(vel_all))
            f.write(pickle.dumps(acc_all))
            f.write(pickle.dumps(vel_all_decorr))
            f.write(pickle.dumps(acc_all_decorr))

    else:
        with open('acc_vel_tmp_residual.pkl', 'rb') as f:
            vel_all = pickle.load(f)
            acc_all = pickle.load(f)
            vel_all_decorr = pickle.load(f)
            acc_all_decorr = pickle.load(f)
    return vel_all, acc_all, vel_all_decorr, acc_all_decorr

def load_data_M1(decorrelation='trialized'):

    data_path = '/home/akumar/nse/neural_control/data/decodingvt_residual_%s' % decorrelation
    #data_path = '/home/akumar/nse/neural_control/data/decodingvt_cv_ttshift_lag2w30'
    # Cross-validated using the dimreduc fit on each fold
    with open('/mnt/Secondary/data/postprocessed/indy_decoding_df2.dat', 'rb') as f:
        rl = pickle.load(f)
    indy_df = pd.DataFrame(rl)

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

    sabes_df = pd.concat([indy_df, loco_df])

    data_idx = 0
    data_file = np.unique(sabes_df['data_file'].values)[data_idx]

    dat = load_sabes('/mnt/Secondary/data/sabes/%s' % data_file)
    dat = reach_segment_sabes(dat, data_file=data_file.split('/')[-1].split('.mat')[0])

    fls = glob.glob('%s/*.dat' % data_path)
    rl = []
    for fl in fls:

        didx = int(fl.split('didx')[1].split('_')[0])
        # dim = int(fl.split('dim')[1].split('_')[0])
        s = fl.split('_')[5:]
        
        filter_params = {}
        # filter_params['error_thresh'] = float(s[7])/100
        # filter_params['error_op'] = s[-2]
        # filter_params['filter_type'] = int(s[1])
        # filter_params['op'] = s[3]
        # filter_params['q'] = float(s[5])/100

        measure_from_end = bool(int(s[-1].split('.dat')[0]))

        with open(fl, 'rb') as f:
            wr2 = pickle.load(f) 
            error_filter = pickle.load(f)
            reach_filter = pickle.load(f)
            window_filter = pickle.load(f)
            windows = pickle.load(f)
            _ = pickle.load(f)
            # MSEtr = pickle.load(f)
            # MSEte = pickle.load(f)
            # full_reaches_train = pickle.load(f)
            # full_reaches_test = pickle.load(f)
            # behavioral_array = pickle.load(f)

        result = {}
        result['r2'] = wr2
        result['error_filter'] = error_filter
        result['window_filter'] = window_filter
        result['window'] = np.squeeze(windows)
        result['reach_filter'] = reach_filter
        result['didx'] = didx
        result['dim'] = 6
        result['measure_from_end'] = measure_from_end

        # result['MSEtr'] = MSEtr
        # result['MSEte'] = MSEte
        # result['full_reaches_train'] = full_reaches_train
        # result['full_reaches_test'] = full_reaches_test
        # result['behavioral_array'] = behavioral_array

        for k, v in filter_params.items():
            result[k] = v
        
        rl.append(result)
    dvt_df = pd.DataFrame(rl)

    # with open('/home/akumar/nse/neural_control/notebooks/intermediate/acc_vel_tmp_ttdiff_w30.pkl', 'rb') as f:
    #     vel_all = pickle.load(f)
    #     acc_all = pickle.load(f)

    vel_all, acc_all, vel_all_decorr, acc_all_decorr = decorrelate_acceleration(dvt_df, sabes_df)

    return dvt_df, vel_all, acc_all, vel_all_decorr, acc_all_decorr

def load_data_S1():

    fls = glob.glob('/home/akumar/nse/neural_control/data/decodingvt_cv_ttshift_S1/*.dat')
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

    # Still need to fold the indy results into calc_single unit statistics S1

    loader_arg = {'bin_width':50, 'filter_fn':'none', 'filter_kwargs':{}, 'boxcox':0.5, 'spike_threshold':100, 'region':'S1'}
    sabes_df = apply_df_filters(sabes_df, loader_args=loader_arg)

    data_files = np.unique(sabes_df['data_file'].values)

    rl = []

    # Get windows according to rank
    window_centers = np.arange(35)
    windows = np.array_split(window_centers, 8)

    for fl in fls:

        didx = int(fl.split('didx')[1].split('_')[0])
        rank = int(fl.split('rank')[1].split('_')[0])

        with open(fl, 'rb') as f:
            wr2 = pickle.load(f) 
            error_filter = pickle.load(f)
            reach_filter = pickle.load(f)
            window_filter = pickle.load(f)
            windows = pickle.load(f)
            filter_params = pickle.load(f)
            MSEtr = pickle.load(f)
            MSEte = pickle.load(f)
            full_reaches_train = pickle.load(f)
            full_reaches_test = pickle.load(f)
            behavioral_metrics = pickle.load(f)


        result = {}
        result['r2'] = wr2
        result['error_filter'] = error_filter
        result['reach_filter'] = reach_filter
        result['window_filter'] = window_filter
        result['window'] = windows
        result['didx'] = didx
        result['dim'] = 6

        result['full_straight_train'] = full_reaches_train
        result['full_straight_test'] = full_reaches_test
        result['MSEtr'] = MSEtr
        result['MSEte'] = MSEte
        result['behavioral_metrics'] = behavioral_metrics
        
        rl.append(result)
    dvt_df = pd.DataFrame(rl)

    with open('/home/akumar/nse/neural_control/notebooks/intermediate/velacc_s1.dat', 'rb') as f:
        vel_all, acc_all, vel_pk = pickle.load(f)

    return dvt_df, vel_all, acc_all

# Provide axes in pairs. The top will plot the desired behavioral metric. The bottom will plot the delta of the metric against acceleration
# For position, do not plot a behavioral metric
def plot_(dvt_df, vel_all, acc_all, vel_all_decorr, acc_all_decorr, metrics, ax, region):

    window_centers = np.arange(30)
    df_ = dvt_df

    # Make sure we have been given enough axes to plot each fo the desired metrics
    # assert(len(metrics) == len(ax) - 1)

    atwins = []

    for i, metric in enumerate(metrics):
        if metric == 'position':
            atwins.append((None, None))
            continue
        elif metric == 'velocity':
            atwins.append((ax[i][0].twinx(), ax[i][1].twinx()))
            atwins[i][0].set_ylabel('Cursor velocity (m/s)', fontsize=14)            
        else:
            atwins.append((ax[i][0].twinx(), ax[i][1].twinx()))
            atwins[i][0].set_ylabel('Cursor acceleration ' r'$(m/s^2)$', fontsize=14)
        atwins[i][1].set_ylabel('Cursor acceleration' r'$(m/s^2)$', fontsize=14)

    # Intersection
    ntr = np.array([len(v) for v in vel_all])
    velall = np.array([elem for v in vel_all for elem in v if len(elem) == 35])
    accall = np.array([elem for v in acc_all for elem in v if len(elem) == 35])
    velalld = np.array([elem for v in vel_all_decorr for elem in v if len(elem) == 30])
    accalld = np.array([elem for v in acc_all_decorr for elem in v if len(elem) == 30])

    for i, metric in enumerate(metrics):
        if metric == 'position':
            continue
        elif metric == 'velocity':
            ax[i][0].plot(np.nan, linestyle='--', color='teal', label='Avg. Vel.')
            atwins[i][0].plot(50 * np.arange(-5, 30), np.mean(velall, axis=0), linestyle='--', color='teal', linewidth=2, alpha=1.)
            atwins[i][0].plot(50 * np.arange(0, 30), np.mean(velalld, axis=0), linestyle='--', color='brown', linewidth=2, alpha=1.)

        else:
            ax[i][0].plot(np.nan, linestyle='--', color='teal', label='Avg. Acc.')
            atwins[i][0].plot(50 * np.arange(-5, 30), np.mean(accall, axis=0), linestyle='--', color='teal', linewidth=2, alpha=1.)
            atwins[i][0].plot(50 * np.arange(0, 30), np.mean(accalld, axis=0), linestyle='--', color='brown', linewidth=2, alpha=1.)

        avg_acc = []
        avg_vel = []

        for k in range(len(acc_all)):
            a = [a_ for a_ in acc_all[k] if a_.size == 35]
            v = [v_ for v_ in vel_all[k] if v_.size == 35]
            v = np.array(v)[:, 5:]
            a = np.array(a)[:, 5:]
            avg_acc.append(np.mean(a, axis=0))
            avg_vel.append(np.mean(v, axis=0))

        avg_acc = np.array(avg_acc).T
        avg_vel = np.array(avg_vel).T

        acc_d, pr22 = decorrelate(avg_vel.T, avg_acc.T)
        pdb.set_trace()
        ax[i][1].plot(np.nan, linestyle='--', color='teal', label='Avg. Acc.')
        atwins[i][1].plot(50 * np.arange(30), np.mean(acc_d, axis=0), linestyle='--', color='teal', linewidth=2, alpha=1.)
        #atwins[i][1].plot(50 * np.arange(0, 30), np.mean(accalld, axis=0), linestyle='--', color='brown', linewidth=2, alpha=1.)

    # Index depending on the metric
    metric_idx = {'position':3, 'velocity':4, 'acceleration':5}
    pos_color = '#4eabfc'
    vel_color = '#615ef7'
    acc_color = '#542991'
    colors = {'position':pos_color, 'velocity':vel_color, 'acceleration':acc_color}
    labels = {'position':'Pos.', 'velocity':'Vel.', 'acceleration':'Acc.'}

    # Cross-correlation
    acf_coef = np.zeros((2 * len(metrics), np.unique(df_['didx'].values).size))

    def plot_trace(rtrace, ax, color, label):
        ax.fill_between(50 * window_centers, np.array([np.sum(np.multiply(r, ntr))/np.sum(ntr) for r in rtrace]) - np.array([np.sqrt(np.var(r) * np.sum(ntr**2)/np.sum(ntr)**2) for r in rtrace]), 
                        np.array([np.sum(np.multiply(r, ntr))/np.sum(ntr) for r in rtrace]) + np.array([np.sqrt(np.var(r) * np.sum(ntr**2)/np.sum(ntr)**2) for r in rtrace]),
                        color=color, zorder=0, alpha=0.25)

        ax.plot(50 * window_centers, [np.sum(np.multiply(r, ntr))/np.sum(ntr) for r in rtrace], '-', color=color, zorder=0, label=label)

    def plot_delta_trace(r2f, r2p, ax, color, label):

        # Normalize the delta r2 relative to the maximum FBC score
        dr = r2f - r2p
        dr_d, pr2 = decorrelate(avg_vel.T, dr.T)
        dr_d = dr_d.T
        dr2 = np.array([np.sum(np.multiply(d, ntr))/np.sum(ntr) for d in dr_d])
        norm_ = np.max(np.array([np.sum(np.multiply(r, ntr))/np.sum(ntr) for r in r2p]))
        dr201 = dr2/norm_

        # How do we handle the std?
        dr2std = np.array([np.sqrt(np.var(d) * np.sum(ntr**2)/np.sum(ntr)**2) for d in dr_d])
        dr2std01 = np.divide(dr2std, norm_)

        ax.fill_between(50 * window_centers, dr201 - dr2std01, dr201 + dr2std01, color=color, alpha=0.25)
        ax.plot(50 * window_centers, dr201, '-', color=color, label=label)

    for i, metric in enumerate(metrics):

        r2f = []
        r2p = []
        for data_idx in np.unique(df_['didx'].values):
            d = apply_df_filters(df_, didx=data_idx)
            wc = []
            r2f_ = []
            r2p_ = []        
            for k in range(d.shape[0]):
                win = d.iloc[k]['window']
                if win.ndim == 1:
                    wc.extend([np.mean(win)])
                else:
                    wc.extend(np.mean(win, axis=1))

                r2f_.extend(np.nanmean(d.iloc[k]['r2'][:, :, 1, metric_idx[metric]], axis=1))
                r2p_.extend(np.nanmean(d.iloc[k]['r2'][:, :, 0, metric_idx[metric]], axis=1)) 

            win_order = np.argsort(wc)
            r2f_ = np.array(r2f_)[win_order]
            r2p_ = np.array(r2p_)[win_order]

            r2f.append(r2f_)
            r2p.append(r2p_)

        r2f = np.array(r2f).T
        r2p = np.array(r2p).T

        plot_trace(r2f, ax[i][0], 'r', 'FBC')
        plot_trace(r2p, ax[i][0], 'k', 'FFC')

        plot_delta_trace(r2f, r2p, ax[i][1], vel_color, r'$\Delta$' + labels[metric])
        #dr_d, pr2 = decorrelate(avg_vel.T, dr.T)
        #acc_d, pr22 = decorrelate(avg_acc.T, dr.T)

               


        # vertical lines
        r2fwa = np.array([np.sum(np.multiply(r, ntr))/np.sum(ntr) for r in r2f])
        r2pwa = np.array([np.sum(np.multiply(r, ntr))/np.sum(ntr) for r in r2p])


        ax[i][0].vlines(50 * window_centers[np.argmax(r2fwa)], 0, 0.6, linestyles=[(0,(3,6))], color='r', alpha=0.75)
        ax[i][0].vlines(50 * window_centers[np.argmax(r2pwa)], 0, 0.6, linestyles=[(4,(3,6))], color='k', alpha=0.75)


        # # Get the 80 % shoulders
        acc_threshold = 0.8 * np.max(np.mean(accall, axis=0))
        acc_idxs = np.argwhere(np.mean(accall, axis=0) > acc_threshold)[[0, -1], 0]

        if metric == 'position':
            pass
        elif metric == 'velocity':
            ax[i][1].vlines(50 * np.arange(-5, 30)[np.argmax(np.mean(velall, axis=0))], 0, 0.725, linestyles='dashed', color='teal', alpha=0.75)    
        else:
            ax[i][1].vlines(50 * np.arange(-5, 30)[acc_idxs[0]], 0, 2.1, linestyles=[(0, (3, 6))], color='teal', alpha=0.75)
            ax[i][1].vlines(50 * np.arange(-5, 30)[acc_idxs[1]], 0, 2.1, linestyles=[(0, (3, 6))], color='teal', alpha=0.75)

        if metric != 'position':
            atwins[i][0].set_xlim([0, 1500])
            atwins[i][1].set_xlim([0, 1500])
            atwins[i][0].set_xticks([0, 500, 1000, 1500])
            atwins[i][1].set_xticks([0, 500, 1000, 1500])
        ax[i][0].set_xlabel('Time from reach start (ms)', fontsize=14)
        ax[i][1].set_xlabel('Time from reach start (ms)', fontsize=14)
        ax[i][0].set_ylabel('%s Prediction ' % labels[metric] + r'$r^2$', fontsize=14)
        ax[i][1].set_ylabel('Normalized ' + r'$\Delta$' + ' ' r'$r^2$', fontsize=14)

        # In the last panel, plot the cross correlation statistics between delta decoding for each metric provided and velocity/acceleration

        avg_acc = []
        avg_vel = []

        for k in range(len(acc_all)):
            a = [a_ for a_ in acc_all[k] if a_.size == 35]
            v = [v_ for v_ in vel_all[k] if v_.size == 35]
            v = np.array(v)[:, 5:]
            a = np.array(a)[:, 5:]
            avg_acc.append(np.mean(a, axis=0))
            avg_vel.append(np.mean(v, axis=0))

        avg_acc = np.array(avg_acc).T
        avg_vel = np.array(avg_vel).T

        dr = r2f - r2p
        avg_vel_01 = avg_vel
        avg_acc_01 = avg_acc

        
        # residual_corr = np.zeros(dr_d.shape[0])
        # for k in range(dr_d.shape[0]):
        #     residual_corr[k] = dr_d[k, :] @ acc_d[k, :]/np.sqrt(dr_d[k, :] @ dr_d[k, :] * acc_d[k, :] @ acc_d[k, :])

        # fig, ax = plt.subplots()
        # plot_trace(dr, ax, 'k', 'dr')
        # plot_trace(dr_d, ax, 'r', 'dr_d')
        # fig.savefig('decorrelation.png')
        for k in range(avg_vel.shape[1]):        
            acf_coef[2*i, k] = dr[:, k] @ avg_vel_01[:, k]/np.sqrt(dr[:, k] @ dr[:, k] * avg_vel_01[:, k] @ avg_vel_01[:, k])
            acf_coef[2*i + 1, k] = dr[:, k] @ avg_acc_01[:, k]/np.sqrt(dr[:, k] @ dr[:, k] * avg_acc_01[:, k] @ avg_acc_01[:, k])


    #print(np.mean(acf_coef.T, axis=1))
    ax[-1].set_ylim([0, 1.05])
    ax[-1].set_ylabel(r'$\Delta r^2$' + '/Kinematic Cross-Corr.', fontsize=14)
    ax[-1].set_yticks([0, 0.5, 1.])

    # Paired difference plot isntead of boxplot
    ax[-1].scatter(np.zeros(acf_coef.shape[1]), acf_coef[0, :], color='b', alpha=0.75, s=3)
    ax[-1].scatter(np.ones(acf_coef.shape[1]), acf_coef[1, :], color='b', alpha=0.75, s=3)
    ax[-1].plot(np.array([0]))
    x = np.array([(0, 1) for _ in range(acf_coef.shape[1])]).T
    y = np.array([(y1, y2) for y1, y2 in zip(acf_coef[0, :], acf_coef[1, :])]).T
    ax[-1].plot(x, y, color='k', alpha=0.5)

    # bplot = ax[-1].boxplot(acf_coef.T, medianprops={'linewidth':0}, showfliers=False, patch_artist=True)

    ticklabels = []
    for metric in metrics:
        ticklabels.extend([r'$\Delta$' + ' %s' % labels[metric], 'Acc./' + r'$\Delta$' + ' %s' % labels[metric]])

    ax[-1].set_xticklabels(ticklabels, rotation=45, ha='right')
    cdict = {'position':pos_color, 'velocity':vel_color, 'acceleration':acc_color}

    colors = []
    for metric in metrics:
        colors.append(cdict[metric])
        colors.append(cdict[metric])

    # for i, patch in enumerate(bplot['boxes']):
    #     patch.set_facecolor(colors[i])
    #     patch.set_alpha(0.75)

    # Statistical tests
    for i in range(len(metrics)):
        print('Metric: %s, Region: %s' % (metrics[i], region))
        print(scipy.stats.wilcoxon(acf_coef[2*i, :], acf_coef[2*i + 1, :], alternative='less'))

    # Return twin axes
    return atwins

if __name__ == '__main__':

    main = True
    supp = False
    ############################################### Main figure ############################################################
    if main:
        dvt_df, vel_all, acc_all, vel_all_decorr, acc_all_decorr = load_data_M1()
        fig = plt.figure(figsize=(10, 8))
        gs = gridspec.GridSpec(2, 3, width_ratios=[2, 2, 1])

        ax = [plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1]), plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1]), 
        plt.subplot(gs[0, 2]), plt.subplot(gs[1, 2])]

        # (1) Vel decoding, (2) Acc decoding, (3) Delta plots, (4) Box plots
        atwins = plot_(dvt_df, vel_all, acc_all, vel_all_decorr, acc_all_decorr,
                       ['velocity'], [(ax[0], ax[2]), ax[4]], 'M1')

        # resize axes
        # Velocity plot
        ax[0].set_ylim([0, 0.55])
        ax[0].set_xlim([0, 1500])
        ax[0].set_yticks([0, 0.5])
        atwins[0][0].set_ylim([0, 0.175])
        atwins[0][0].set_yticks([0, 0.15])

        # # Delta plot
        ax[2].set_ylim([0, 0.4])
        ax[2].set_yticks([-3., 3.])
        atwins[0][1].set_ylim([0, 1.05])
        atwins[0][1].set_yticks([0, 1])

        lg1 = ax[0].legend(loc='upper right', fontsize=10, frameon=True, edgecolor=(0, 0, 0, 1.), facecolor=(1, 1, 1, 1.))
        lg1.get_frame().set_alpha(0.9)

        ax[0].set_zorder(1) # make it on top
        ax[0].set_frame_on(False) # make it transparent
        atwins[0][0].set_frame_on(True) # make sure there is any background

        atwins = plot_(dvt_df, vel_all, acc_all, vel_all_decorr, acc_all_decorr,
                       ['acceleration'], [(ax[1], ax[3]), ax[5]], 'M1')


        # dvt_df, vel_all, acc_all= load_data_S1()
        # # ax = [plt.subplot(gs[0, 2]), plt.subplot(gs[1, 2]), None, None, plt.subplot(gs[1, 3])]
        # atwins = plot_(dvt_df, vel_all, acc_all, ['velocity'], [(ax[1], ax[3]), ax[5]], 'S1')

        # # resize axes
        # # Velocity plot
        # ax[1].set_ylim([0, 0.55])
        # ax[1].set_xlim([0, 1500])
        # ax[1].set_yticks([0, 0.5])
        # atwins[0][0].set_ylim([0, 0.175])
        # atwins[0][0].set_yticks([0, 0.15])

        # # # Delta plot
        # ax[3].set_ylim([0, 2.0])
        # ax[3].set_yticks([0, 1.75])
        # atwins[0][1].set_ylim([0, 1.05])
        # atwins[0][1].set_yticks([0, 1])

        # lg1 = ax[1].legend(loc='upper left', fontsize=10, frameon=True, edgecolor=(0, 0, 0, 1.), facecolor=(1, 1, 1, 1.))
        # lg1.get_frame().set_alpha(0.9)

        # ax[1].set_zorder(1) # make it on top
        # ax[1].set_frame_on(False) # make it transparent
        # atwins[0][0].set_frame_on(True) # make sure there is any background

        # lg2 = ax[2].legend(loc='upper right', fontsize=10, frameon=True, edgecolor=(0, 0, 0, 1.), facecolor=(1, 1, 1, 1.))
        # lg3 = ax[3].legend(bbox_to_anchor=(0.475, 0, 0.5, 0.2), fontsize=10, frameon=True, edgecolor=(0, 0, 0, 1.), facecolor=(1, 1, 1, 1.))
        # lg2.get_frame().set_alpha(0.9)
        # lg3.get_frame().set_alpha(1.)

        fig.tight_layout()
        fig.savefig('/home/akumar/nse/neural_control/figs/loco_indy_merge/decodingvt_residual2.pdf', bbox_inches='tight', pad_inches=0.1)


    ###################################################### Supplementary Figure ###################################################
    if supp:

        dvt_df, vel_all, acc_all = load_data_M1()
        fig = plt.figure(figsize=(18, 8))
        gs = gridspec.GridSpec(2, 5, width_ratios=[2, 2, 2, 2, 1])

        ax = [plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1]), 
            plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1]), 
            plt.subplot(gs[0, 2]), plt.subplot(gs[0, 3]),
            plt.subplot(gs[1, 2]), plt.subplot(gs[1, 3]),
            plt.subplot(gs[0, 4]), plt.subplot(gs[1, 4])]

        # (1) Vel decoding, (2) Acc decoding, (3) Delta plots, (4) Box plots
        atwins = plot_(dvt_df, vel_all, acc_all, ['position', 'acceleration'], 
                    [(ax[0], ax[2]), (ax[1], ax[3]), ax[8]], 'M1')

        # resize axes
        # M1 position prediction
        ax[0].set_ylim([0, 0.35])
        ax[0].set_xlim([0, 1500])
        ax[0].set_yticks([0, 0.3])
        # atwins[0][0].set_ylim([0, 0.175])
        # atwins[0][0].set_yticks([0, 0.15])

        # # # Delta plot
        ax[2].set_ylim([0, 0.45])
        ax[2].set_yticks([0, 0.4])
        # atwins[0][1].set_ylim([0, 1.05])
        # atwins[0][1].set_yticks([0, 1])

        lg1 = ax[0].legend(loc='upper right', fontsize=10, frameon=True, edgecolor=(0, 0, 0, 1.), facecolor=(1, 1, 1, 1.))
        lg1.get_frame().set_alpha(0.9)
        lg2 = ax[2].legend(loc='lower right', fontsize=10, frameon=True, edgecolor=(0, 0, 0, 1.), facecolor=(1, 1, 1, 1.))
        lg2.get_frame().set_alpha(0.9)

        ax[0].set_zorder(1) # make it on top
        # ax[0].set_frame_on(False) # make it transparent
        # atwins[0][0].set_frame_on(True) # make sure there is any background

        # M1 acceleration prediction
        ax[1].set_ylim([0, 0.42])
        ax[1].set_xlim([0, 1500])
        ax[1].set_yticks([0, 0.4])
        atwins[1][0].set_ylim([0, 1.05])
        atwins[1][0].set_yticks([0, 1])

        # # # Delta plot
        ax[3].set_ylim([0, 0.45])
        ax[3].set_yticks([0, 0.4])
        atwins[1][1].set_ylim([0, 1.05])
        atwins[1][1].set_yticks([0, 1])

        lg1 = ax[1].legend(loc='upper right', fontsize=10, frameon=True, edgecolor=(0, 0, 0, 1.), facecolor=(1, 1, 1, 1.))
        lg1.get_frame().set_alpha(0.9)
        lg2 = ax[3].legend(loc='upper right', fontsize=10, frameon=True, edgecolor=(0, 0, 0, 1.), facecolor=(1, 1, 1, 1.))
        lg2.get_frame().set_alpha(0.9)

        dvt_df, vel_all, acc_all= load_data_S1()
        # ax = [plt.subplot(gs[0, 2]), plt.subplot(gs[1, 2]), None, None, plt.subplot(gs[1, 3])]
        atwins = plot_(dvt_df, vel_all, acc_all, ['position', 'acceleration'], 
                    [(ax[4], ax[6]), (ax[5], ax[7]), ax[9]], 'S1')


        # S1 position prediction
        ax[4].set_ylim([0, 0.35])
        ax[4].set_xlim([0, 1500])
        ax[4].set_yticks([0, 0.3])
        # atwins[0][0].set_ylim([0, 0.175])
        # atwins[0][0].set_yticks([0, 0.15])

        # # # Delta plot
        ax[6].set_ylim([0, 2.25])
        ax[6].set_yticks([0, 2.0])
        # atwins[0][1].set_ylim([0, 1.05])
        # atwins[0][1].set_yticks([0, 1])

        lg1 = ax[4].legend(loc='upper right', fontsize=10, frameon=True, edgecolor=(0, 0, 0, 1.), facecolor=(1, 1, 1, 1.))
        lg1.get_frame().set_alpha(0.9)
        lg2 = ax[6].legend(loc='lower right', fontsize=10, frameon=True, edgecolor=(0, 0, 0, 1.), facecolor=(1, 1, 1, 1.))
        lg2.get_frame().set_alpha(0.9)

        ax[4].set_zorder(1) # make it on top
        # ax[0].set_frame_on(False) # make it transparent
        # atwins[0][0].set_frame_on(True) # make sure there is any background

        # S1 acceleration prediction
        ax[5].set_ylim([0, 0.12])
        ax[5].set_xlim([0, 1500])
        ax[5].set_yticks([0, 0.1])
        atwins[1][0].set_ylim([0, 1.05])
        atwins[1][0].set_yticks([0, 1])

        # # # Delta plot
        ax[7].set_ylim([0, 4.5])
        ax[7].set_yticks([0, 4.0])
        atwins[1][1].set_ylim([0, 1.05])
        atwins[1][1].set_yticks([0, 1])

        lg1 = ax[1].legend(loc='upper right', fontsize=10, frameon=True, edgecolor=(0, 0, 0, 1.), facecolor=(1, 1, 1, 1.))
        lg1.get_frame().set_alpha(0.9)
        lg2 = ax[3].legend(loc='upper right', fontsize=10, frameon=True, edgecolor=(0, 0, 0, 1.), facecolor=(1, 1, 1, 1.))
        lg2.get_frame().set_alpha(0.9) 



        fig.tight_layout()
        fig.savefig('/home/akumar/nse/neural_control/figs/loco_indy_merge/decodingvt_revised_supplement.pdf', bbox_inches='tight', pad_inches=0.1)


    # if region == 'M1':
        
    #     # Acceleration plot
    #     ax[2].set_xlim([0, 1500])
    #     ax[2].set_ylim([0, 0.55])
    #     ax[2].set_yticks([0, 0.5])
    #     a3.set_ylim([0, 1.05])
    #     a3.set_xticks([0, 500, 1000, 1500])
    #     a3.set_yticks([0, 1])
    #     ax[2].set_ylabel('Acc. Prediction ' + r'$r^2$', fontsize=14)
    #     ax[2].set_xlabel('Time from reach start (ms)', fontsize=14)

    #     # Delta plot
    #     ax[3].set_xlim([0, 1500])
    #     ax[3].set_xticks([0, 500, 1000, 1500])
    #     a3.set_xticks([0, 500, 1000, 1500])
    #     ax[3].set_ylim([0, 1.15])
    #     ax[3].set_yticks([0, 1])
    #     ax[3].set_ylabel('Normalized ' + r'$\Delta$' + ' ' r'$r^2$', fontsize=14, labelpad=-2)
    #     ax[3].set_xlabel('Time from reach start (ms)', fontsize=14)
    #     a4.set_ylim([0, 1.05])
    #     a4.set_yticks([0, 1])

    # if region == 'M1':
    #     lg1 = ax[0].legend(loc='upper right', fontsize=10, frameon=True, edgecolor=(0, 0, 0, 1.), facecolor=(1, 1, 1, 1.))
    # else:
    #     lg1 = ax[0].legend(loc='upper left', fontsize=10, frameon=True, edgecolor=(0, 0, 0, 1.), facecolor=(1, 1, 1, 1.))

    # lg2 = ax[1].legend(bbox_to_anchor=(0.6, 0, 0.2, 0.2), fontsize=10, frameon=True, edgecolor=(0, 0, 0, 1.), facecolor=(1, 1, 1, 1.))
    # lg1.get_frame().set_alpha(0.9)
    # lg2.get_frame().set_alpha(0.9)

    # if region == 'M1':
    #     lg3 = ax[2].legend(loc='upper right', fontsize=10, frameon=True, edgecolor=(0, 0, 0, 1.), facecolor=(1, 1, 1, 1.))
    #     lg4 = ax[3].legend(bbox_to_anchor=(0.475, 0, 0.2, 0.2), fontsize=10, frameon=True, edgecolor=(0, 0, 0, 1.), facecolor=(1, 1, 1, 1.))
    #     lg3.get_frame().set_alpha(0.9)
    #     lg4.get_frame().set_alpha(1.)

    # ax[0].set_zorder(1) # make it on top
    # ax[0].set_frame_on(False) # make it transparent
    # a.set_frame_on(True) # make sure there is any background

    # ax[1].set_zorder(1) # make it on top
    # ax[1].set_frame_on(False) # make it transparent
    # a2.set_frame_on(True) # make sure there is any background








    # Generate an auxillary figure for the legend
    # fig, ax = plt.subplots(figsize=(4, 4))
    # ax.plot(np.nan, linestyle='--', color='teal', label='Acc. Norm.')
    # ax.plot(np.nan, linestyle='-', color=vel_color, label=r'$\Delta$' + ' Vel.')
    # ax.plot(np.nan, linestyle='-', color=acc_color, label=r'$\Delta$' + ' Acc.')

    # ax.legend(loc='upper left', fontsize=10, frameon=True, edgecolor=(0, 0, 0, 1.), facecolor=(1, 1, 1, 1.))
    # fig.savefig('/home/akumar/nse/neural_control/figs/loco_indy_merge/decodingvt_revised_legend.pdf', bbox_inches='tight', pad_inches=0)
