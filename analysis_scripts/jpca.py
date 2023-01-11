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

from dca.methods_comparison import JPCA
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


    # Where to save?
    if len(sys.argv) > 1:
        figpath = sys.argv[1]
    else:
        figpath = '/home/akumar/nse/neural_control/figs/final'


    with open('/mnt/Secondary/data/postprocessed/indy_decoding_df2.dat', 'rb') as f:
        sabes_df = pickle.load(f)
    sabes_df = pd.DataFrame(sabes_df)

    data_files = np.unique(sabes_df['data_file'].values)
    dpath = '/mnt/Secondary/data/sabes'

    DIM = 6
    if not os.path.exists('jpcaAtmp.dat'):
        # Now do subspace identification/VAR inference within these 
        # results = []
        resultsd3 = []
        for i, data_file in tqdm(enumerate(data_files)):
            dat = load_sabes('%s/%s' % (dpath, data_file))
            dat = reach_segment_sabes(dat, data_file=data_file.split('.mat')[0])

            y = np.squeeze(dat['spike_rates'])
            for dimreduc_method in ['DCA', 'KCA', 'LQGCA', 'PCA']:
                df_ = apply_df_filters(sabes_df, data_file=data_file, fold_idx=0, dim=DIM, dimreduc_method=dimreduc_method)
                if dimreduc_method == 'LQGCA':
                    df_ = apply_df_filters(df_, dimreduc_args={'T': 3, 'loss_type': 'trace', 'n_init': 10})
                V = df_.iloc[0]['coef']
                if dimreduc_method == 'PCA':
                    V = V[:, 0:DIM]        

                # Project data
                yproj = y @ V

                # Segment reaches


                result_ = {}
                result_['data_file'] = data_file
                result_['dimreduc_method'] = dimreduc_method

                # # Fit subspace identification
                # ssid = SubspaceIdentification()
                # A, C, Cbar, L0, Q, R, S = ssid.identify(yproj, order=6)

                # result_['ssid_A'] = A

                # Fit VAR(1) and VAR(2)
                # varmodel = VAR(estimator='ols', order=1)
                # varmodel.fit(yproj)
                # result_['var1_A'] = form_companion(varmodel.coef_) 
                # result_['var1score'] = varmodel.score(yproj)

                # varmodel = VAR(estimator='ols', order=1, self_regress=True)
                # varmodel.fit(yproj)
                # result_['var1_A_sr'] = form_companion(varmodel.coef_) 
                # result_['var1srscore'] = varmodel.score(yproj)


                # varmodel = VAR(estimator='ols', order=2)
                # varmodel.fit(yproj)
                # result_['var2_A'] = form_companion(varmodel.coef_) 
                # result_['var2score'] = varmodel.score(yproj)

                # varmodel = VAR(estimator='ols', order=2, self_regress=True)
                # varmodel.fit(yproj)
                # result_['var2_A_sr'] = form_companion(varmodel.coef_) 
                # result_['var2srscore'] = varmodel.score(yproj)

                # varmodel = VAR(estimator='ols', order=3)
                # varmodel.fit(yproj)
                # result_['var3_A'] = form_companion(varmodel.coef_) 
                # result_['var3score'] = varmodel.score(yproj)

                # varmodel = VAR(estimator='ols', order=3, self_regress=True)
                # varmodel.fit(yproj)
                # result_['var3_A_sr'] = form_companion(varmodel.coef_) 
                # result_['var3srscore'] = varmodel.score(yproj)


                # x = np.array([StandardScaler().fit_transform(dat['spike_rates'][j, ...]) 
                #             for j in range(dat['spike_rates'].shape[0])])
                yproj = StandardScaler().fit_transform(yproj)

                jpca = JPCA(n_components=DIM, mean_subtract=False)
                jpca.fit(yproj[np.newaxis, ...])
                
                linmodel = LinearRegression()
                linmodel.fit(yproj[:-1, :], np.diff(yproj, axis=0))

                ypred = yproj[:-1, :] @ jpca.M_skew
                r2_jpca = jpca.r2_score
                r2_linear = linmodel.score(yproj[:-1, :], np.diff(yproj, axis=0))
                print('method: %s, r2_jpca: %f, r2_lin: %f' % (dimreduc_method, r2_jpca, r2_linear))
                result_['jeig'] = np.imag(np.linalg.eigvals(linmodel.coef_))
                resultsd3.append(result_)


        with open('jpcaAtmp.dat', 'wb') as f:
            f.write(pickle.dumps(resultsd3))            
    else:
        with open('jpcaAtmp.dat', 'rb') as f:
            resultsd3 = pickle.load(f)

    A_df = pd.DataFrame(resultsd3)

    d_U = np.zeros((28, 4, 3))
    maxim = np.zeros((28, 4, 2))

    d1 = []
    d2 = []

    for i in range(28):
        for j, dimreduc_method in enumerate(['DCA', 'KCA', 'LQGCA', 'PCA']):
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

            # maxim[i, j, 1] = np.sum(np.abs(np.imag(eigsd)))/2
            maxim[i, j, 0] = np.sum(np.abs(eigs))/2

    print(d1)
    print(d2)

    # Next up:
    # Rotational trajectories.
    data_file = data_files[20]

    df1 = apply_df_filters(sabes_df, data_file=data_file, fold_idx=0, dim=6, dimreduc_method='PCA')
    df2 = apply_df_filters(sabes_df, data_file=data_file, fold_idx=0, dim=6, dimreduc_method='LQGCA', 
                           dimreduc_args={'T':3, 'loss_type':'trace', 'n_init':10})


    datpath = '/mnt/Secondary/data/sabes'
    dat = load_sabes('%s/%s' % (datpath, data_file))
    dat = reach_segment_sabes(dat, start_times[data_file.split('.mat')[0]])

    x = np.array([StandardScaler().fit_transform(dat['spike_rates'][j, ...]) 
                for j in range(dat['spike_rates'].shape[0])])
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
        start = trajectory[0, :]
        end = trajectory[-1, :]
        
        pca_straightdev[i] = measure_straight_dev(trajectory, start, end)

        trajectory = gaussian_filter1d(xdca_j[0, transition_times[i][0]:transition_times[i][1]], 
                                    sigma=5, axis=0)
        start = trajectory[0, :]
        end = trajectory[-1, :]
        dca_straightdev[i] = measure_straight_dev(trajectory, start, end)
        

    #pca_devorder = np.arange(pca_straightdev.size)
    #dca_devorder = np.arange(dca_straightdev.size)    
    pca_devorder = np.argsort(pca_straightdev)[::-1]
    dca_devorder = np.argsort(dca_straightdev)[::-1]


    ############## Trajectory Plots #################
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    for i in range(0, 25):
        
        idx = pca_devorder[i]

        # Plot only 20 timesteps
        t0 = transition_times[idx][0]
        t1 = min(transition_times[idx][0] + 40, transition_times[idx][1])

        trajectory = gaussian_filter1d(xpca_j[0, t0:t1], 
                                    sigma=5, axis=0)

        # Center and normalize trajectories
        trajectory -= trajectory[0]
        trajectory /= np.linalg.norm(trajectory)

        # Rotate trajectory so that the first 5 timesteps all go off at the same angle
        theta0 = np.arctan2(trajectory[15, 1], trajectory[15, 0])

        # Rotate *clockwise* by theta
        R = lambda theta: np.array([[np.cos(-1*theta), -np.sin(-theta)], \
                                     [np.sin(-theta), np.cos(theta)]])        
        trajectory = np.array([R(theta0) @ t[0:2] for t in trajectory])

        ax[0].plot(trajectory[:, 0], trajectory[:, 1], 'k', alpha=0.5)
        ax[0].arrow(trajectory[-1, 0], trajectory[-1, 1], 
                    trajectory[-1, 0] - trajectory[-2, 0], trajectory[-1, 1] - trajectory[-2, 1], 
                    head_width=0.005, color="k", alpha=0.5)
        
        
        idx = dca_devorder[i]
        t0 = transition_times[idx][0]
        t1 = min(transition_times[idx][0] + 40, transition_times[idx][1])
        trajectory = gaussian_filter1d(xdca_j[0, t0:t1], sigma=5, axis=0)

        # Center trajectories
        trajectory -= trajectory[0]
        trajectory /= np.linalg.norm(trajectory)

        # Rotate trajectory so that the first 5 timesteps all go off at the same angle
        theta0 = np.arctan2(trajectory[15, 1], trajectory[15, 0])

        trajectory = np.array([R(theta0) @ t[0:2] for t in trajectory])

        ax[1].plot(trajectory[:, 0], trajectory[:, 1], '#c73d34', alpha=0.5)
        ax[1].arrow(trajectory[-1, 0], trajectory[-1, 1], 
                    trajectory[-1, 0] - trajectory[-2, 0], trajectory[-1, 1] - trajectory[-2, 1], 
                    head_width=0.005, color="#c73d34", alpha=0.5)

        ax[0].set_xticklabels([])
        ax[0].set_yticklabels([])
        
        ax[1].set_xticklabels([])
        ax[1].set_yticklabels([])
        
    ax[0].set_xlim([-0.3, 0.3])
    ax[1].set_xlim([-0.3, 0.3])

    ax[0].set_ylim([-0.3, 0.3])
    ax[1].set_ylim([-0.3, 0.3])

    ax[0].set_title('jPCA on PCA', fontsize=14)
    ax[0].set_ylabel('jPC2', fontsize=14)
    ax[0].set_xlabel('jPC1', fontsize=14)

    ax[1].set_title('jPCA on FCCA', fontsize=14)
    ax[1].set_ylabel('jPC2', fontsize=14)
    ax[1].set_xlabel('jPC1', fontsize=14)

    ax[0].spines['right'].set_color('none')
    ax[0].spines['top'].set_color('none')
    ax[0].spines['left'].set_color('none')
    ax[0].spines['bottom'].set_color('none')
    ax[0].set_xticks([])
    ax[0].set_yticks([])

    ax[1].spines['right'].set_color('none')
    ax[1].spines['top'].set_color('none')
    ax[1].spines['left'].set_color('none')
    ax[1].spines['bottom'].set_color('none')
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    fig.tight_layout()
    fig.savefig('%s/trajectories.pdf' % figpath, bbox_inches='tight', pad_inches=0)

    # Boxplots
    fig, ax = plt.subplots(1, 1, figsize=(5, 2))

    medianprops = dict(linewidth=0)
    #bplot = ax.boxplot([d_U[:, 2, 1], d_U[:, 3, 1]], patch_artist=True, medianprops=medianprops, notch=True, vert=False, showfliers=False)
    bplot = ax.boxplot([maxim[:, 2, 0], maxim[:, 3, 0]], patch_artist=True, medianprops=medianprops, notch=True, vert=False, showfliers=False)

    # _, p = scipy.stats.wilcoxon(d_U[:, 2, 1], d_U[:, 3, 1])
    _, p = scipy.stats.wilcoxon(maxim[:, 2, 0], maxim[:, 3, 0])
    print('p:%f' % p)

    method1 = 'FCCA'
    method2 = 'PCA'
 
    ax.set_yticklabels([method1, method2], fontsize=14)
    ax.tick_params(axis='y', labelsize=12)
    ax.set_xlabel('Sum of Imaginary Eigenvalues (a.u.)', fontsize=14)
    ax.invert_xaxis()
 
    # fill with colors
    colors = ['red', 'black']
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    # ax.set_xlim([13, 0])

    fig.tight_layout()
    fig.savefig('%s/jpca_eig_bplot.pdf' % figpath, bbox_inches='tight', pad_inches=0)