import numpy as np
import scipy
import sys
import pdb
import matplotlib.pyplot as plt
from glob import glob
import pickle
from pyuoi.linear_model.var import VAR
from tqdm import tqdm
import pandas as pd
from neurosim.models.var import VAR as VARss
from neurosim.models.var import form_companion
from copy import deepcopy
from sklearn.model_selection import KFold

from pseudopy import Normal

sys.path.append('/home/akumar/nse/neural_control')
from loaders import load_sabes, load_peanut, load_cv
from subspaces import estimate_autocorrelation
from utils import apply_df_filters
from dstableFGM import dstable_descent


if __name__ == '__main__':

    # Indy VAR
    with open('/home/akumar/nse/neural_control/data/indy_var_df.dat', 'rb') as f:
        rl = pickle.load(f)
    indy_df = pd.DataFrame(rl)
    # # Loco VAR
    # with open('/home/akumar/nse/neural_control/data/loco_var_df.dat', 'rb') as f:
    #     rl = pickle.load(f)
    # loco_df = pd.DataFrame(rl)

    # # CV VAR
    # with open('cv_var_list.dat', 'rb') as f:
    #     cv_result_list = pickle.load(f)
    # cv_df = pd.DataFrame(cv_result_list)

    data_files = np.unique(indy_df['data_file'].values)
    df_ = apply_df_filters(indy_df, data_file=data_files[0], fold_idx = 2, order=3)
    A_M1 = form_companion(df_.iloc[0]['coef'])

    # df_ = apply_df_filters(loco_df, fold_idx=0, order=2, region='S1', self_regress=False)
    # A_S1 = form_companion(df_.iloc[0]['coef'])

    # df_ = apply_df_filters(cv_df, fold_idx =0, var_order=2)
    # A_CV = form_companion(df_.iloc[0]['coef'])


    # VAR calculations for peanut
    data_file = '/mnt/Secondary/data/peanut/data_dict_peanut_day14.obj'
    loader_args = {'bin_width':25, 'epoch': 4, 'filter_fn':'none', 'filter_kwargs':{}, 'boxcox':0.5, 'spike_threshold':100}
    dat = load_peanut(data_file, **loader_args, region='HPc')
    y = np.squeeze(dat['spike_rates'])
    varmodel = VAR(estimator='ols', order=2)
    varmodel.fit(y)
    A_HPc = form_companion(varmodel.coef_)

    # loader_args = {'bin_width':25, 'epoch': 4, 'filter_fn':'none', 'filter_kwargs':{}, 'boxcox':0.5, 'spike_threshold':100}
    # dat = load_peanut(data_file, **loader_args, region='OFC')
    # y = np.squeeze(dat['spike_rates'])
    # varmodel = VAR(estimator='ols', order=2)
    # varmodel.fit(y)
    # A_OFC = form_companion(varmodel.coef_)

    with open('/home/akumar/nse/neural_control/data/pseudospectral_calculations.dat', 'rb') as f:
        nn_M1 = pickle.load(f)
        nn_HPc = pickle.load(f)
        nn_CV = pickle.load(f)

    n_M1 = Normal(A_M1)
    n_HPc = Normal(A_HPc)
    # n_CV = Normal(A_CV)
    # n_S1 = Normal(A_S1)
    # n_OFC = Normal(A_OFC)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    levels = [1e-6, 1e-1]

    ax.set_aspect('equal')
    ax.tricontourf(nn_M1.triang, nn_M1.vals, levels=levels, colors=['k'], alpha=0.2)
    ax.scatter(np.real(np.linalg.eigvals(A_M1)), np.imag(np.linalg.eigvals(A_M1)), s=15, alpha=0.5, marker='o', edgecolor='k', color='r')

    epsilons = list(np.sort(levels))
    padepsilons = [epsilons[0]*0.9] + epsilons + [epsilons[-1]*1.1]
    X = []
    Y = []
    Z = []
    for epsilon in padepsilons:
        paths = n_M1.contour_paths(epsilon)
        for path in paths:
            X += list(np.real(path.vertices[:-1]))
            Y += list(np.imag(path.vertices[:-1]))
            Z += [epsilon] * (len(path.vertices) - 1)
    ax.tricontour(X, Y, Z, levels=[1e-1], colors='k')


    # Add stability circle
    circle1 = plt.Circle((0, 0), 1, color='k', fill=False, linestyle='--')
    ax.add_patch(circle1)
    ax.set_ylabel('Im' + r'$(z)$', fontsize=14)
    ax.set_yticks([-1., -0.5, 0, 0.5, 1.])
    ax.set_xlabel('Re' + r'$(z)$', fontsize=14)
    ax.set_xticks([-1., -0.5, 0, 0.5, 1.])

    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])

    ax.tick_params(axis='both', labelsize=14)
    ax.set_title('Macaque M1', fontsize=16)
    fig.savefig('/home/akumar/nse/neural_control/figs/final/M1_pseudospectra.pdf', bbox_inches='tight', pad_inches=0)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    levels = [1e-6, 1e-1]

    nn_obj = nn_HPc
    n_obj = n_HPc

    ax.set_aspect('equal')
    ax.tricontourf(nn_obj.triang, nn_obj.vals, levels=levels, colors=['k'], alpha=0.2)
    ax.scatter(np.real(np.linalg.eigvals(A_HPc)), np.imag(np.linalg.eigvals(A_HPc)), s=15, alpha=0.5, marker='o', edgecolor='k', color='r')

    epsilons = list(np.sort(levels))
    padepsilons = [epsilons[0]*0.9] + epsilons + [epsilons[-1]*1.1]
    X = []
    Y = []
    Z = []
    for epsilon in padepsilons:
        paths = n_obj.contour_paths(epsilon)
        for path in paths:
            X += list(np.real(path.vertices[:-1]))
            Y += list(np.imag(path.vertices[:-1]))
            Z += [epsilon] * (len(path.vertices) - 1)
    ax.tricontour(X, Y, Z, levels=[1e-1], colors='k')


    # Add stability circle
    circle1 = plt.Circle((0, 0), 1, color='k', fill=False, linestyle='--')
    ax.add_patch(circle1)
    ax.set_ylabel('Im' + r'$(z)$', fontsize=14)
    ax.set_yticks([-1., -0.5, 0, 0.5, 1.])
    ax.set_xlabel('Re' + r'$(z)$', fontsize=14)
    ax.set_xticks([-1., -0.5, 0, 0.5, 1.])

    ax.set_xlim([-1.3, 1.3])
    ax.set_ylim([-1.3, 1.3])

    ax.tick_params(axis='both', labelsize=14)
    ax.set_title('Rat Hippocampus', fontsize=16)
    fig.savefig('/home/akumar/nse/neural_control/figs/final/HPc_pseudospectra.pdf', bbox_inches='tight', pad_inches=0)