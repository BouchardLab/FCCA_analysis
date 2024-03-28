import pdb
import os, sys
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
from sklearn.metrics import r2_score

from scipy.ndimage import gaussian_filter1d
from mpl_toolkits.axisartist.axislines import AxesZero

from dca.methods_comparison import JPCA, symmJPCA
from pyuoi.linear_model.var  import VAR
from neurosim.models.var import form_companion

sys.path.append('/home/akumar/nse/neural_control')
from utils import apply_df_filters, calc_loadings
from loaders import load_sabes
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

    calcs = False
    rot_trajectories = False
    dyn_range = True
    boxplots = False

    # Where to save?
    if len(sys.argv) > 1:
        figpath = sys.argv[1]
    else:
        figpath = '/home/akumar/nse/neural_control/figs/loco_indy_merge'


    with open('/mnt/Secondary/data/postprocessed/indy_decoding_df2.dat', 'rb') as f:
        indy_df = pickle.load(f)
    indy_df = pd.DataFrame(indy_df)

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

    sabes_df = pd.concat([loco_df, indy_df])

    data_files = np.unique(sabes_df['data_file'].values)
    dpath = '/mnt/Secondary/data/sabes'

    DIM = 6
    if calcs:
        # Now do subspace identification/VAR inference within these 
        # results = []
        resultsd3 = []
        for i, data_file in tqdm(enumerate(data_files)):
            dat = load_sabes('%s/%s' % (dpath, data_file))
            dat = reach_segment_sabes(dat, data_file=data_file.split('.mat')[0])
            y = np.squeeze(dat['spike_rates'])
            for dimreduc_method in ['LQGCA', 'PCA']:
                df_ = apply_df_filters(sabes_df, data_file=data_file, fold_idx=0, dim=DIM, dimreduc_method=dimreduc_method)
                if dimreduc_method == 'LQGCA':
                    df_ = apply_df_filters(df_, dimreduc_args={'T': 3, 'loss_type': 'trace', 'n_init': 10})

                assert(df_.shape[0] == 1)
                V = df_.iloc[0]['coef']
                if dimreduc_method == 'PCA':
                    V = V[:, 0:DIM]        

                # Project data
                yproj = y @ V
                yproj = np.array([yproj[t0:t0+20] for t0, t1 in dat['transition_times'] if t1 - t0 > 21])

                # Z-score
                #yproj = gaussian_filter1d(yproj, sigma=2, axis=1)
                # Center and normalize trajectories
                #yproj = np.array([yproj[i, :] - yproj[i, 0] for i in range(yproj.shape[0])])
                # yproj = np.array([StandardScaler().fit_transform(yproj[i, :]) for i in range(yproj.shape[0])])
                #trajectory /= np.linalg.norm(trajectory)

                # Rotate trajectory so that the first 5 timesteps all go off at the same angle
                # for i in range(yproj.shape[0]):
                #     theta0 = np.arctan2(yproj[i, 15, 1], yproj[i, 15, 0])

                #     # Rotate *clockwise* by theta
                #     R = lambda theta: np.array([[np.cos(-1*theta), -np.sin(-theta)], \
                #                                 [np.sin(-theta), np.cos(theta)]])        
                #     yproj[i] = np.array([R(theta0 - np.pi/4) @ t[0:2] for t in yproj[i]])

                # x_ = yproj[:-1, :]
                # y_ = yproj[1:, :]
                # x_ = StandardScaler().fit_transform(x_)
                # y_ = StandardScaler().fit_transform(y_)
                # linmodel = LinearRegression()
                # linmodel.fit(x_, y_)
                # print('%s\n' % dimreduc_method)
                # print(np.linalg.eigvals(linmodel.coef_).astype(str))

                # Segment reaches into minimum length 30 timesteps reaches
                # yproj = gaussian_filter1d(yproj, sigma=5)

                result_ = {}
                result_['data_file'] = data_file
                result_['dimreduc_method'] = dimreduc_method

                # 3 fits: 
                # 3. Look at symmetric vs. asymmetric portions of regression onto differences

                jpca = JPCA(n_components=DIM, mean_subtract=False)
                jpca.fit(yproj)
                
                # Look at only the initial phase of the trajectory
                # # x_ = yproj[:, :-1, :].reshape((-1, 6))
                # # x_ = StandardScaler().fit_transform(x_)
                # # y_ = np.diff(yproj, axis=1).reshape((-1, 6))
                # # y_ = StandardScaler().fit_transform(y_)
                # x_ = yproj[:, :-1, :].reshape((-1, 6))
                # y_ = yproj[:, 1:, :].reshape((-1, 6))
                # x_ = StandardScaler().fit_transform(x_)
                # y_ = StandardScaler().fit_transform(y_)

                # #linmodel.fit(yproj[:, :-1, :].reshape((-1, 6)), np.diff(yproj, axis=1).reshape((-1, 6)))
                # linmodel.fit(x_, y_)

                # ypred = yproj[:, :-1, :] @ jpca.M_skew
                #r2_jpca = jpca.r2_score
                #r2_linear = linmodel.score(yproj[:-1, :], np.diff(yproj, axis=0))
                #r2_linear = np.nan
                #print('method: %s, r2_jpca: %f, r2_lin: %f' % (dimreduc_method, r2_jpca, r2_linear))
                result_['jeig'] = jpca.eigen_vals_

                # Record the average dynamic range
                # yprojcent = np.array([y_ - y_[0:1, :] for y_ in yproj])
                yprojcent = yproj

                # For each time step, calculate the least squares projection of the state vector onto the next step
                a = np.eye(DIM)
                for ii in range(yprojcent.shape[1] - 1):
                    lmodel = LinearRegression(fit_intercept=False)
                    lmodel.fit(yprojcent[:, ii, :], yprojcent[:, ii + 1, :])
                    a = a @ lmodel.coef_

                print('%s\n' % dimreduc_method)
                print(np.abs(np.linalg.eigvals(a)))

                dyn_range = np.array([np.max(np.abs(y_)[:, j]) for y_ in yprojcent for j in range(DIM)])
                result_['dyn_range'] = np.mean(dyn_range)
                #print(jpca.eigen_vals_)
                #print(np.mean(dyn_range))
                resultsd3.append(result_)

        # with open('jpcaAtmp_il2.dat', 'wb') as f:
        #     f.write(pickle.dumps(resultsd3))            
    else:
        with open('jpcaAtmp_il2.dat', 'rb') as f:
            resultsd3 = pickle.load(f)

    A_df = pd.DataFrame(resultsd3)

    d_U = np.zeros((len(data_files), 2, 3))
    maxim = np.zeros((len(data_files), 2, 3))

    d1 = []
    d2 = []

    with open('jpcaAtmp_randomcontrol2.dat', 'rb') as f:
        control_results = pickle.load(f)
    controldf = pd.DataFrame(control_results)

    maxim_control = np.zeros((len(data_files), 1000, 3))

    for i in range(len(data_files)):
        for j, dimreduc_method in enumerate(['LQGCA', 'PCA']):
            df_ = apply_df_filters(A_df, data_file=data_files[i], dimreduc_method=dimreduc_method)
            # A = df_.iloc[0]['ssid_A']
            # U, P = scipy.linalg.polar(A)
            # d_U[i, j, 0] = np.linalg.norm(A - U)/np.linalg.norm(A)
            # A = df_.iloc[0]['var3_A_sr']
            # U, P = scipy.linalg.polar(A)
            # d_U[i, j, 1] = np.linalg.norm(A - U)/np.linalg.norm(A)
            #d_U[i, j, 1] = 1 - np.linalg.norm(U)/np.linalg.norm(A)
            # A = df_.iloc[0]['var2_A']
            # U, P = scipy.linalg.polar(A)
            # d_U[i, j, 2] = np.linalg.norm(A - U)/np.linalg.norm(A)

            eigs = df_.iloc[0]['jeig']
            # eigsd = np.linalg.eigvals(A)

            # if j == 2:
            #     d1.append(np.linalg.det(U))
            # if j == 3:
            #     d2.append(np.linalg.det(U))

            maxim[i, j, 0] = np.sum(np.abs(eigs))/2

            maxim[i, j, 1] = np.sum(np.abs(eigs))/2
            maxim[i, j, 2] = df_.iloc[0]['dyn_range']

        for j in range(maxim_control.shape[1]):
            df_ = apply_df_filters(controldf, data_file=data_files[i], inner_rep=j)
            assert(df_.shape[0] == 1)

            eigs = df_.iloc[0]['jeig']
            maxim_control[i, j, 0] = np.sum(np.abs(eigs))/2
            maxim_control[i, j, 1] = np.sum(np.abs(eigs))/2
            eigs = df_.iloc[0]['dyn_range']
            maxim_control[i, j, 2] = eigs

    print(d1)
    print(d2)

    # Next up:
    # Rotational trajectories.
    if rot_trajectories:
        # (5., 7., 10., 12.. 16. 17. 19. 20. 25..)
        data_file = data_files[25]

        df1 = apply_df_filters(sabes_df, data_file=data_file, fold_idx=0, dim=6, dimreduc_method='PCA')
        df2 = apply_df_filters(sabes_df, data_file=data_file, fold_idx=0, dim=6, dimreduc_method='LQGCA', 
                            dimreduc_args={'T':3, 'loss_type':'trace', 'n_init':10})


        datpath = '/mnt/Secondary/data/sabes'
        dat = load_sabes('%s/%s' % (datpath, data_file))
        dat = reach_segment_sabes(dat, data_file=data_file.split('.mat')[0])

        # x = np.array([StandardScaler().fit_transform(dat['spike_rates'][j, ...]) 
        #             for j in range(dat['spike_rates'].shape[0])])
        x = dat['spike_rates']
        xpca = x @ df1.iloc[0]['coef'][:, 0:6]
        xdca = x @ df2.iloc[0]['coef']

        jpca1 = JPCA(n_components=6, mean_subtract=False)
        jpca1.fit(xpca)

        jpca2 = JPCA(n_components=6, mean_subtract=False)
        jpca2.fit(xdca)

        xpca_j = jpca1.transform(xpca)
        xdca_j = jpca2.transform(xdca)


        # Measure the straight_dev of the projected neural data
        pca_straightdev = np.zeros(len(dat['target_pairs']))
        dca_straightdev = np.zeros(len(dat['target_pairs']))
        transition_times = dat['transition_times']
        for i in range(len(dat['target_pairs'])):
            
            trajectory = gaussian_filter1d(xpca_j[0, transition_times[i][0]:transition_times[i][1]], 
                                        sigma=5, axis=0)
            trajectory -= trajectory[0]
            
            start = trajectory[0, :]
            end = trajectory[-1, :]
            
            pca_straightdev[i] = measure_straight_dev(trajectory, start, end)

            trajectory = gaussian_filter1d(xdca_j[0, transition_times[i][0]:transition_times[i][1]], 
                                        sigma=5, axis=0)
            trajectory -= trajectory[0]
            
            start = trajectory[0, :]
            end = trajectory[-1, :]
            dca_straightdev[i] = measure_straight_dev(trajectory, start, end)
            

        #pca_devorder = np.arange(pca_straightdev.size)
        #dca_devorder = np.arange(dca_straightdev.size)    
        pca_devorder = np.argsort(pca_straightdev)[::-1]
        dca_devorder = np.argsort(dca_straightdev)[::-1]

        ############## Trajectory Plots #################


        # Save as two separate figures
        fig1, ax1 = plt.subplots(1, 1, figsize=(5, 5))
        fig2, ax2 = plt.subplots(1, 1, figsize=(5, 5))
        ax = [ax1, ax2]

        for i in range(0, 25):
            
            idx = dca_devorder[i]

            # Plot only 20 timesteps
            t0 = transition_times[idx][0]
            t1 = min(transition_times[idx][0] + 40, transition_times[idx][1])

            trajectory = gaussian_filter1d(xpca_j[0, t0:t1], 
                                        sigma=5, axis=0)[:-3]

            # Center and normalize trajectories
            trajectory -= trajectory[0]
            #trajectory /= np.linalg.norm(trajectory)

            # Rotate trajectory so that the first 5 timesteps all go off at the same angle
            theta0 = np.arctan2(trajectory[15, 1], trajectory[15, 0])

            # Rotate *clockwise* by theta
            R = lambda theta: np.array([[np.cos(-1*theta), -np.sin(-theta)], \
                                        [np.sin(-theta), np.cos(theta)]])        
            trajectory = np.array([R(theta0 - np.pi/4) @ t[0:2] for t in trajectory])

            ax[1].plot(trajectory[:, 0], trajectory[:, 1], 'k', alpha=0.5)
            ax[1].arrow(trajectory[-1, 0], trajectory[-1, 1], 
                        trajectory[-1, 0] - trajectory[-2, 0], trajectory[-1, 1] - trajectory[-2, 1], 
                        head_width=0.08, color="k", alpha=0.5)
            
            
            idx = dca_devorder[i]
            t0 = transition_times[idx][0]
            t1 = min(transition_times[idx][0] + 40, transition_times[idx][1])
            trajectory = gaussian_filter1d(xdca_j[0, t0:t1], sigma=5, axis=0)[:-3]

            # Center trajectories
            trajectory -= trajectory[0]
            #trajectory /= np.linalg.norm(trajectory)

            # Rotate trajectory so that the first 5 timesteps all go off at the same angle
            theta0 = np.arctan2(trajectory[15, 1], trajectory[15, 0])

            trajectory = np.array([R(theta0 - np.pi/4) @ t[0:2] for t in trajectory])

            ax[0].plot(trajectory[:, 0], trajectory[:, 1], '#c73d34', alpha=0.5)
            ax[0].arrow(trajectory[-1, 0], trajectory[-1, 1], 
                        trajectory[-1, 0] - trajectory[-2, 0], trajectory[-1, 1] - trajectory[-2, 1], 
                        head_width=0.05, color="#c73d34", alpha=0.5)

        _, p = scipy.stats.wilcoxon(maxim[:, 0, 2], maxim[:, 1, 2], alternative='less')
        print('Re p:%f' % p)


        # ax[0].set_xticklabels([])
        # ax[0].set_yticklabels([])
        
        # ax[1].set_xticklabels([])
        # ax[1].set_yticklabels([])

        ax[0].set_aspect('equal')   
        ax[1].set_aspect('equal')
        ax[1].set_xlim([-2.2, 3.5])
        ax[1].set_ylim([-2.2, 3.5])

        ax[0].set_xlim([-2.2, 3.5])
        ax[0].set_ylim([-2.2, 3.5])

        # ax[1].set_title('jPCA on PCA', fontsize=14)
        # ax[1].set_ylabel('jPC2', fontsize=14)
        # ax[1].set_xlabel('jPC1', fontsize=14)

        # ax[0].set_title('jPCA on FCCA', fontsize=14)
        # ax[0].set_ylabel('jPC2', fontsize=14)
        # ax[0].set_xlabel('jPC1', fontsize=14)

        ax[0].spines['right'].set_color('none')
        ax[0].spines['top'].set_color('none')
        ax[0].spines['left'].set_position('zero')
        ax[0].spines['bottom'].set_position('zero')
        ax[0].plot(2, 0, ">k", clip_on=False)
        ax[0].plot(0, 2, "^k", clip_on=False)
        ax[0].spines['left'].set_bounds(0, 2)
        ax[0].spines['bottom'].set_bounds(0, 2)
        # ax[0].spines['left'].set_color('none')
        # ax[0].spines['bottom'].set_color('none')
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[1].set_xticks([])
        ax[1].set_yticks([])

        ax[1].spines['right'].set_color('none')
        ax[1].spines['top'].set_color('none')
        ax[1].spines['left'].set_position('zero')
        ax[1].spines['bottom'].set_position('zero')
        ax[1].spines['left'].set_bounds(0, 2)
        ax[1].spines['bottom'].set_bounds(0, 2)
        ax[1].plot(2, 0, ">k", clip_on=False)
        ax[1].plot(0, 2, "^k", clip_on=False)

        # ax[1].spines['left'].set_color('none')
        # ax[1].spines['bottom'].set_color('none')
        # ax[1].set_xticks([])
        # ax[1].set_yticks([])
        fig1.tight_layout()
        fig1.savefig('%s/trajectories_a.pdf' % figpath, bbox_inches='tight', pad_inches=0)

        fig2.tight_layout()
        fig2.savefig('%s/trajectories_b.pdf' % figpath, bbox_inches='tight', pad_inches=0)


    ############## Trajectory Amplification #################
    if dyn_range:
        for didx, data_file in enumerate(data_files):
            datpath = '/mnt/Secondary/data/sabes'
            dat = load_sabes('%s/%s' % (datpath, data_file))
            dat = reach_segment_sabes(dat, data_file=data_file.split('.mat')[0])

            y = np.squeeze(dat['spike_rates'])

            colors = ['k', 'r']
            for j, dimreduc_method in enumerate(['PCA', 'LQGCA']):
                df_ = apply_df_filters(sabes_df, data_file=data_file, fold_idx=0, dim=DIM, dimreduc_method=dimreduc_method)
                if dimreduc_method == 'LQGCA':
                    df_ = apply_df_filters(df_, dimreduc_args={'T': 3, 'loss_type': 'trace', 'n_init': 10})

                assert(df_.shape[0] == 1)
                V = df_.iloc[0]['coef']
                if dimreduc_method == 'PCA':
                    V = V[:, 0:DIM]        

                # Project data
                yproj = y @ V
                yproj = np.array([yproj[t0:t0+40] for t0, t1 in dat['transition_times'] if t1 - t0 > 40])
                yproj = np.array([y_ - y_[0] for y_ in yproj])
                dY = np.concatenate(np.diff(yproj, axis=1), axis=0)
                Y_prestate = np.concatenate(yproj[:, :-1], axis=0)

                # Least squares
                A, _, _, _ = np.linalg.lstsq(Y_prestate, dY, rcond=None)
                _, s, _ = np.linalg.svd(A)
                print('%s' % dimreduc_method + s)

                # Identify the directions in which there is the most amplification over multiple timesteps
                # Project the data along those directions and also record the amplification implied by the model
            
                # Iterate the lyapunov equation for 10 timesteps
                P = np.zeros((DIM, DIM))
                for _ in range(10):
                    dP = A @ P + P @ A.T + np.eye(DIM)
                    P += dP

                eig, U = np.linalg.eig(P)
                # eig, U = np.linalg.eig(scipy.linalg.expm(A.T) @ scipy.linalg.expm(A))
                eig = np.sort(eig)[::-1]
                U = U[:, np.argsort(eig)[::-1]]
                U = U[:, 0:2]
                # Plot smoothed, centered trajectories for all reaches in top 2 dimensions

                # Argsort by the maximum amplitude in the top 2 dimensions
                trajectory = gaussian_filter1d(yproj, sigma=5, axis=1)
                trajectory -= trajectory[:, 0:1, :]
                trajectory = trajectory @ U
                dyn_range = np.max(np.abs(trajectory), axis=1)
                ordering = np.argsort(dyn_range, axis=0)[::-1]

                t0 = trajectory[ordering[:, 0], :, 0]
                t1 = trajectory[ordering[:, 1], :, 1]

                f1, a1 = plt.subplots(1, 1, figsize=(4.2, 4))
                f2, a2 = plt.subplots(1, 1, figsize=(4.2, 4))
                ax = [a1, a2]

                for i in range(min(50, t0.shape[0])):
                    ax[0].plot(50 * np.arange(40), t0[i], color=colors[j], alpha=0.5, linewidth=1.5)
                    ax[1].plot(50 * np.arange(40), t1[i], color=colors[j], alpha=0.5, linewidth=1.5)
                    #ax[2*j].set_title(np.sum(eig))
                    
                for a in ax:
                    a.spines['bottom'].set_position('center')
                    # Eliminate upper and right axes
                    a.spines['right'].set_color('none')
                    a.spines['top'].set_color('none')

                    # Show ticks in the left and lower axes only
                    a.xaxis.set_ticks_position('bottom')
                    a.yaxis.set_ticks_position('left')

                    a.set_xticks([0, 2])
                    a.set_xticklabels([])
                    a.tick_params(axis='both', labelsize=12)

                    a.set_xlabel('Time (s)', fontsize=12)
                    a.xaxis.set_label_coords(1.1, 0.56)
                    
                # Set y scale according to the current yscale on PCA 0
                if j == 0:
                    ylim_max = np.max(np.abs(t0[0])) + 0.25
                    ylim = [-ylim_max, ylim_max]

                for a in ax:
                    a.set_ylim(ylim)
                    a.set_yticks([-int(ylim_max), 0, int(ylim_max)])
                    a.set_ylabel('Amplitude (a.u.)', fontsize=12)

                if j == 0:
                    ax[0].set_title('FFC Component 1', fontsize=12)
                    ax[1].set_title('FFC Component 2', fontsize=12)
                else:
                    ax[0].set_title('FBC Component 1', fontsize=12)
                    ax[1].set_title('FBC Component 2', fontsize=12)

                #f1.tight_layout()
                #f2.tight_layout()
                f1.savefig('/home/akumar/nse/neural_control/figs/amplification/%d_e_%s1.pdf' % (didx, dimreduc_method), bbox_inches='tight', pad_inches=0)
                f2.savefig('/home/akumar/nse/neural_control/figs/amplification/%d_e_%s2.pdf' % (didx, dimreduc_method), bbox_inches='tight', pad_inches=0)
    if boxplots:
        # Boxplots
        fig, ax = plt.subplots(2, 1, figsize=(6, 3))
        medianprops = dict(linewidth=1, color='b')
        whiskerprops=dict(linewidth=0)
        #bplot = ax.boxplot([d_U[:, 2, 1], d_U[:, 3, 1]], patch_artist=True, medianprops=medianprops, notch=True, vert=False, showfliers=False)
        # Plot relative to the control...test for difference from zero

        # Center relative to random - per recording session
        mu = np.mean(maxim_control[..., 1], axis=1)
        sigma = np.std(maxim_control[..., 1], axis=1)
        r1 = maxim[:, 0, 1] - mu
        r2 = maxim[:, 1, 1] - mu

        bplot = ax[0].boxplot([r1, r2], patch_artist=True, 
                        medianprops=medianprops, notch=False, vert=False, showfliers=False, widths=[0.3, 0.3],
                        whiskerprops=whiskerprops, showcaps=False)

        # _, p = scipy.stats.wilcoxon(d_U[:, 2, 1], d_U[:, 3, 1])
        _, p = scipy.stats.wilcoxon(maxim[:, 0, 1], maxim[:, 1, 1], alternative='greater')
        print('Im p:%f' % p)

        # test that each is stochastically greater than the median random
        x1 = maxim[:, 0, 1] - np.median(maxim_control[..., 1].ravel())
        x2 = maxim[:, 1, 1] - np.median(maxim_control[..., 1].ravel())

        _, p1 = scipy.stats.wilcoxon(x1, alternative='greater')
        _, p2 = scipy.stats.wilcoxon(x2, alternative='greater')

       # _, p = scipy.stats.wilcoxon(maxim[:, 0, 2], maxim[:, 1, 2], alternative='less')
        # print('Re p:%f' % p)

        # Mutliple comparison adjusted test of maxim control against PCA and FCCA
        # _, p1 = scipy.stats.mannwhitneyu(maxim[:, 0, 1], maxim_control[..., 1].ravel(), alternative='greater')
        # _, p2 = scipy.stats.mannwhitneyu(maxim[:, 1, 1], maxim_control[..., 1].ravel(), alternative='greater')

        # print((p1, p2))

        method1 = 'FBC'
        method2 = 'FFC'
    
        ax[0].set_yticklabels([method1, method2], fontsize=12)
        ax[0].set_xticks([0.0, 5, 10])
        ax[0].set_xlim([0, 10])
        ax[0].tick_params(axis='both', labelsize=12)
        #ax.set_ylabel(r'$\sum_i Im(\lambda_i)$', fontsize=22)
        ax[0].set_xlabel('Strength of Rotational Component above Random', fontsize=12)
        #ax.set_title('****', fontsize=14)

        ax[0].invert_yaxis()
    
        # fill with colors
        colors = ['red', 'black', 'blue']   
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        whiskerprops = dict(linewidth=0)

        mu = np.mean(maxim_control[..., 2], axis=1)
        sigma = np.std(maxim_control[..., 2], axis=1)
        r1 = maxim[:, 0, 2] - mu
        r2 = maxim[:, 1, 2] - mu

        bplot = ax[1].boxplot([r1, r2], patch_artist=True, 
                        medianprops=medianprops, notch=False, vert=False, showfliers=False, widths=[0.3, 0.3],
                        whiskerprops=whiskerprops, showcaps=False)

        x1 = maxim[:, 0, 2] - np.median(maxim_control[..., 2].ravel())
        x2 = maxim[:, 1, 2] - np.median(maxim_control[..., 2].ravel())

        _, p1 = scipy.stats.wilcoxon(x1, alternative='greater')
        _, p2 = scipy.stats.wilcoxon(x2, alternative='greater')
        print(p1)
        print(p2)
    
        method1 = 'FBC'
        method2 = 'FFC'

        ax[1].set_yticklabels([method1, method2], fontsize=12)
        ax[1].set_xticks([0.0, 30, 60])
        ax[1].set_xlim([0, 70])
        ax[1].tick_params(axis='both', labelsize=12)
        #ax.set_ylabel(r'$\sum_i Im(\lambda_i)$', fontsize=22)
        ax[1].set_xlabel('Average Dynamic Range above Random', fontsize=12)
        #ax.set_title('****', fontsize=14)

        ax[1].invert_yaxis()
    
        # fill with colors
        colors = ['red', 'black', 'blue']   
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)


        # ax.set_xlim([13, 0])

        fig.tight_layout()
        fig.savefig('%s/jpca_eig_bplot_wcontrol2.pdf' % figpath, pad_inches=1)