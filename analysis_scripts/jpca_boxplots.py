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

# Combining M1 and S1 jPCA plots by rotation and amplification to make them easier to fold into the main figure 
if __name__ == '__main__':
    figpath = '/home/akumar/nse/neural_control/figs/loco_indy_merge'

    ##### Load M1 results ########
    with open('jpcaAtmp_il2.dat', 'rb') as f:
        resultsd3 = pickle.load(f)

    A_df = pd.DataFrame(resultsd3)
    data_files = np.unique(A_df['data_file'].values)
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

    maxim_controlM1 = np.copy(maxim_control)
    maximM1 = np.copy(maxim)

    ############################# Load S1 results ############################################

    with open('jpcaAtmp_ilS1_2.dat', 'rb') as f:
        resultsd3 = pickle.load(f)

    A_df = pd.DataFrame(resultsd3)
    data_files = np.unique(A_df['data_file'].values)
    d_U = np.zeros((len(data_files), 2, 3))
    maxim = np.zeros((len(data_files), 2, 3))

    d1 = []
    d2 = []

    with open('jpcaAtmp_randomcontrolS1.dat', 'rb') as f:
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

            # maxim[i, j, 1] = np.sum(np.abs(np.imag(eigsd)))/2
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

    maxim_controlS1 = np.copy(maxim_control)
    maximS1 = np.copy(maxim)

    ############################# Rotation Boxplots ###########################################
    fig, ax = plt.subplots(2, 1, figsize=(6, 3))

    medianprops = dict(linewidth=1, color='b')
    whiskerprops = dict(linewidth=0)

    #bplot = ax.boxplot([d_U[:, 2, 1], d_U[:, 3, 1]], patch_artist=True, medianprops=medianprops, notch=True, vert=False, showfliers=False)
    muM1 = np.mean(maxim_controlM1[..., 1], axis=1)
    muS1 = np.mean(maxim_controlS1[..., 1], axis=1)
    r1 = maximM1[:, 0, 1] - muM1
    r2 = maximM1[:, 1, 1] - muM1

    r3 = maximS1[:, 0, 1] - muS1
    r4 = maximS1[:, 1, 1] - muS1
    r = [[r1, r2], [r3, r4]]
    # _, p = scipy.stats.wilcoxon(d_U[:, 2, 1], d_U[:, 3, 1])
    _, p1 = scipy.stats.wilcoxon(maximM1[:, 0, 1], maximM1[:, 1, 1], alternative='greater')
    _, p2 = scipy.stats.wilcoxon(maximS1[:, 0, 1], maximS1[:, 1, 1], alternative='greater')
    print(p1)
    print(p2)
    method1 = 'FBC'
    method2 = 'FFC'

    for i, a in enumerate(ax):
        bplot = a.boxplot(r[i], patch_artist=True, 
                          medianprops=medianprops, notch=False, vert=False, 
                          showfliers=False, widths=[0.3, 0.3], whiskerprops=whiskerprops, showcaps=False)


        a.set_yticklabels([method1, method2], fontsize=12)
        a.set_xticks([0.0, 0.075, 0.15])
        a.set_xlim([0, 0.15])
        a.tick_params(axis='both', labelsize=12)
        #ax.set_ylabel(r'$\sum_i Im(\lambda_i)$', fontsize=22)
        a.set_xlabel('Strength of Rotational Component above Random', fontsize=12)
        #ax.set_title('*', fontsize=14)
        a.invert_yaxis()

        # fill with colors
        colors = ['red', 'black', 'blue']   
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

    fig.tight_layout()
    fig.savefig('%s/jpca_rot_bplot.pdf' % figpath, pad_inches=1)

    ############################# Amplitude Bxoplots ##########################################

    fig, ax = plt.subplots(2, 1, figsize=(6, 3))

    medianprops = dict(linewidth=1, color='b')
    whiskerprops = dict(linewidth=0)

    #bplot = ax.boxplot([d_U[:, 2, 1], d_U[:, 3, 1]], patch_artist=True, medianprops=medianprops, notch=True, vert=False, showfliers=False)
    muM1 = np.mean(maxim_controlM1[..., 2], axis=1)
    muS1 = np.mean(maxim_controlS1[..., 2], axis=1)
    r1 = maximM1[:, 0, 2] - muM1
    r2 = maximM1[:, 1, 2] - muM1

    r3 = maximS1[:, 0, 2] - muS1
    r4 = maximS1[:, 1, 2] - muS1
    r = [[r1, r2], [r3, r4]]
    # _, p = scipy.stats.wilcoxon(d_U[:, 2, 1], d_U[:, 3, 1])
    _, p1 = scipy.stats.wilcoxon(maximM1[:, 1, 2], maximM1[:, 0, 2], alternative='greater')
    _, p2 = scipy.stats.wilcoxon(maximS1[:, 1, 2], maximS1[:, 0, 2], alternative='greater')
    print(p1)
    print(p2)

    method1 = 'FBC'
    method2 = 'FFC'

    for i, a in enumerate(ax):
        bplot = a.boxplot(r[i], patch_artist=True, 
                          medianprops=medianprops, notch=False, vert=False, 
                          showfliers=False, widths=[0.3, 0.3], whiskerprops=whiskerprops, showcaps=False)


        a.set_yticklabels([method1, method2], fontsize=12)
        a.set_xticks([0.0, 1.5, 3.0])
        a.set_xlim([0, 3])
        a.tick_params(axis='both', labelsize=12)
        #ax.set_ylabel(r'$\sum_i Im(\lambda_i)$', fontsize=22)
        a.set_xlabel('Average Dynamic Range above Random', fontsize=12)
        #ax.set_ylabel(r'$\sum_i Im(\lambda_i)$', fontsize=22)
        #a.set_xlabel('Strength of Rotational Component above Random', fontsize=12)
        #ax.set_title('*', fontsize=14)
        a.invert_yaxis()

        # fill with colors
        colors = ['red', 'black', 'blue']   
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

    fig.tight_layout()
    fig.savefig('%s/jpca_dynrange_bplot.pdf' % figpath, pad_inches=1)


