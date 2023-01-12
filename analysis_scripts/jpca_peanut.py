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
from loaders import load_peanut

if __name__ == '__main__':


    # Where to save?
    if len(sys.argv) > 1:
        figpath = sys.argv[1]
    else:
        figpath = '/home/akumar/nse/neural_control/figs/loco_indy_merge'


    with open('/mnt/Secondary/data/postprocessed/peanut_decoding_df.dat', 'rb') as f:
        peanut_decoding_df = pickle.load(f)

    peanut_decoding_df = pd.DataFrame(peanut_decoding_df)

    #fig, ax = plt.subplots(4, 2, figsize=(10, 12))
    epochs = np.unique(peanut_decoding_df['epoch'].values)
    folds = np.unique(peanut_decoding_df['fold_idx'].values)
    dimvals = np.unique(peanut_decoding_df['dim'].values)
    decoder_args = [{'trainlag': 0, 'testlag': 0, 'decoding_window': 6}, {'trainlag': 3, 'testlag': 3, 'decoding_window': 6}, {'trainlag': 6, 'testlag': 6, 'decoding_window': 6}]


    data_file = '/mnt/Secondary/data/peanut/data_dict_peanut_day14.obj'

    DIM = 6
    if not os.path.exists('jpcaAtmp_peanut.dat'):
        # Now do subspace identification/VAR inference within these 
        resultsd3 = []
        for i, epoch in tqdm(enumerate(epochs)):

            dat = load_peanut('/mnt/Secondary/data/peanut/data_dict_peanut_day14.obj', epoch=epoch, spike_threshold=200)
            y = np.squeeze(dat['spike_rates'])
            for dimreduc_method in ['LQGCA', 'PCA']:
                df_ = apply_df_filters(peanut_decoding_df, epoch=epoch, fold_idx=0, dim=DIM, dimreduc_method=dimreduc_method, decoder_args=decoder_args[0])
                if dimreduc_method == 'LQGCA':
                    df_ = apply_df_filters(df_, dimreduc_args={'T':3, 'loss_type':'trace', 'n_init':5})

                try:
                    assert(df_.shape[0] == 1)
                except:
                    pdb.set_trace()

                V = df_.iloc[0]['coef']
                if dimreduc_method == 'PCA':
                    V = V[:, 0:DIM]        

                # Project data
                yproj = y @ V

                # yproj = gaussian_filter1d(yproj, sigma=5)

                result_ = {}
                result_['epoch'] = epoch
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
#                result_['jeig'] = np.imag(np.linalg.eigvals(linmodel.coef_))
                result_['jeig'] = jpca.eigen_vals_
                resultsd3.append(result_)


        with open('jpcaAtmp_peanut.dat', 'wb') as f:
            f.write(pickle.dumps(resultsd3))            
    else:
        with open('jpcaAtmp_peanut.dat', 'rb') as f:
            resultsd3 = pickle.load(f)

    A_df = pd.DataFrame(resultsd3)

    d_U = np.zeros((len(epochs), 2, 3))
    maxim = np.zeros((len(epochs), 2, 2))

    d1 = []
    d2 = []

    for i in range(len(epochs)):
        for j, dimreduc_method in enumerate(['LQGCA', 'PCA']):
            df_ = apply_df_filters(A_df, epoch=epochs[i], dimreduc_method=dimreduc_method)
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

    # Boxplots
    fig, ax = plt.subplots(1, 1, figsize=(5, 2.5))

    medianprops = dict(linewidth=0)
    #bplot = ax.boxplot([d_U[:, 2, 1], d_U[:, 3, 1]], patch_artist=True, medianprops=medianprops, notch=True, vert=False, showfliers=False)
    bplot = ax.boxplot([maxim[:, 0, 0], maxim[:, 1, 0]], patch_artist=True, medianprops=medianprops, notch=True, vert=False, showfliers=False, widths=[0.25, 0.25])

    # _, p = scipy.stats.wilcoxon(d_U[:, 2, 1], d_U[:, 3, 1])
    _, p = scipy.stats.wilcoxon(maxim[:, 0, 0], maxim[:, 1, 0])
    print('p:%f' % p)

    method1 = 'FCCA'
    method2 = 'PCA'
 
    ax.set_yticklabels([method1, method2], fontsize=18, rotation=45)
    ax.set_xticks([0, 0.05, 0.1])
    ax.tick_params(axis='both', labelsize=16)
    #ax.set_ylabel(r'$\sum_i Im(\lambda_i)$', fontsize=22)
    ax.set_xlabel('Sum Imaginary Eigenvalues', fontsize=18
    )
    ax.set_title('****', fontsize=22)
    #ax.invert_xaxis()
 
    # fill with colors
    colors = ['red', 'black']
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # ax.set_xlim([13, 0])

    fig.tight_layout()
    fig.savefig('%s/jpca_eigbplot_peanut.pdf' % figpath, bbox_inches='tight', pad_inches=0)