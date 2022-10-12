import pdb
import numpy as np
import matplotlib.pyplot as plt
import scipy 
import time
import glob
import pickle
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LinearRegression, LogisticRegression
import itertools
from sklearn.model_selection import KFold
import sys
sys.path.append('/home/akumar/nse/neural_control')
from utils import apply_df_filters, calc_loadings

from loaders import load_cv
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from mpi4py import MPI


def fit():


    with open('/home/akumar/nse/neural_control/data/cv_dimreduc_df2.dat', 'rb') as f:
        cv_df = pickle.load(f)
    # pdb.set_trace()

    # with open('/home/akumar/nse/neural_control/data/cv_dimreduc_df.dat', 'rb') as f:
    #     cv_df = pickle.load(f)

    cv_df = pd.DataFrame(cv_df)
    data_files = np.unique(cv_df['data_file'].values)

    #KCA_args = [{'T':10, 'causal_weights':(1, 0), 'n_init':5}, {'T':20, 'causal_weights':(1, 0), 'n_init':5}]
    #LQGCA_args = [{'T':10, 'loss_type':'trace', 'n_init':5}, {'T':20, 'loss_type':'trace', 'n_init':5}]
    #DCA_args = [{'T': 10, 'n_init': 5}, {'T':20, 'n_init':5}]
    LQGCA_args = [{'T':40, 'loss_type':'trace', 'n_init':5}]

    folds = np.arange(5)
    dimvals = np.unique(cv_df['dim'].values)
    #dimreduc_methods = ['DCA10', 'DCA20', 'KCA10', 'KCA20', 'LQGCA10', 'LQGCA20', 'PCA']
    dimreduc_methods = ['LQGCA40', 'PCA']

    # Do the decoding
    decoding_list =[]

    comm = MPI.COMM_WORLD

    for data_file in data_files:
        dat = load_cv(data_file)
        for f, fold in enumerate([folds[comm.rank]]):    
            for dr_method in dimreduc_methods:
                for d, dimval in tqdm(enumerate(dimvals)):            
                    if 'KCA' in dr_method:
                        df_ = apply_df_filters(cv_df, dimreduc_method='KCA', fold_idx=fold, dim=dimval)
                        # Further filter by dimreduc_args
                        if dr_method == 'KCA10':
                            df_ = apply_df_filters(df_, dimreduc_args=KCA_args[0])
                        elif dr_method == 'KCA20':
                            df_ = apply_df_filters(df_, dimreduc_args=KCA_args[1])
                    elif 'LQGCA' in dr_method:
                        df_ = apply_df_filters(cv_df, dimreduc_method='LQGCA', fold_idx=fold, dim=dimval, data_file=data_file)
                        # Further filter by dimreduc_args 
                        # if dr_method == 'LQGCA10':
                        #     df_ = apply_df_filters(df_, dimreduc_args=LQGCA_args[0])
                        # elif dr_method == 'LQGCA20':
                        #     df_ = apply_df_filters(df_, dimreduc_args=LQGCA_args[1])
                    elif 'DCA' in dr_method:
                        df_ = apply_df_filters(cv_df, dimreduc_method='DCA', fold_idx=fold, dim=dimval)
                        if dr_method == 'DCA10':
                            df_ = apply_df_filters(df_, dimreduc_args=DCA_args[0])
                        elif dr_method == 'DCA20':
                            df_ = apply_df_filters(df_, dimreduc_args=DCA_args[1])

                    else:
                        df_ = apply_df_filters(cv_df, dimreduc_method=dr_method, fold_idx=fold, dim=dimval, data_file=data_file)

                    assert(df_.shape[0] == 1)
                    
                    Xtrain = dat['spike_rates'][df_.iloc[0]['train_idxs']] @ df_.iloc[0]['coef']
                    Ytrain = dat['behavior'][df_.iloc[0]['train_idxs']]

                    Xtest = dat['spike_rates'][df_.iloc[0]['test_idxs']] @ df_.iloc[0]['coef']
                    Ytest = dat['behavior'][df_.iloc[0]['test_idxs']]
                    
                    classifier = LogisticRegression(multi_class='multinomial', max_iter=500, solver='lbfgs', tol=1e-4).fit(Xtrain.reshape((Xtrain.shape[0], -1)), Ytrain)
                    acc = classifier.score(Xtest.reshape((Xtest.shape[0], -1)), Ytest)
                    
                    result = {}
                    result['dr_method'] = dr_method
                    result['fold'] = fold
                    result['data_file'] = data_file
                    result['dimval'] = d
                    result['acc'] = acc
                    result['classifier_coef'] = classifier.coef_
                    decoding_list.append(result)

    with open('decoding_list3_%d.dat' % comm.rank, 'wb') as f:
        f.write(pickle.dumps(decoding_list))

def plot():


    # Open and consolidate dimreduc
    with open('/mnt/Secondary/data/postprocessed/cv_dimreduc_df.dat', 'rb') as f:
        cv_df = pickle.load(f)

    cv_df = pd.DataFrame(cv_df)
    data_files = np.unique(cv_df['data_file'].values)

    # Open and consolidate decoding
    decoding_files = glob.glob('decoding_list2_*')
    result_list = []
    for file_ in decoding_files:
        with open(file_, 'rb') as f:
            result_list.extend(pickle.load(f))
    
    decoding_files = glob.glob('decoding_list3_*')
    for file_ in decoding_files:
        with open(file_, 'rb') as f:
            result_list.extend(pickle.load(f))

    df = pd.DataFrame(result_list)

    # Get null accuracy

    null_accuracy = np.zeros((2, 5, 1, 2))

    for i, data_file in enumerate(data_files):
        dat = load_cv(data_file)
        for fold in range(5):
            df_ = apply_df_filters(cv_df, dim=2, data_file=data_file, fold_idx=fold, dimreduc_method='PCA')
            ytrain = dat['behavior'][df_.iloc[0]['train_idxs']]
            ytest = dat['behavior'][df_.iloc[0]['test_idxs']]
            
            options, counts = np.unique(ytrain, return_counts=True)
            guess = options[counts.argmax()]
            null_accuracy[i, fold, 0, 0] = (ytrain == guess).mean()
            null_accuracy[i, fold, 0, 1] = (ytest == guess).mean()

    
#    dimvals = np.unique(df['dimval'].values)
    dimvals = np.arange(1, 13)

    r2_fca = np.zeros((len(data_files), dimvals.size, 5))
    r2_pca = np.zeros((len(data_files), dimvals.size, 5))

    for h, data_file in enumerate(data_files):
        for i, dim in enumerate(dimvals):
            for j, fold in enumerate(range(5)):        
                df_ = apply_df_filters(df, dimval=dim, fold=fold, dr_method='PCA', data_file='/mnt/Secondary/data/cv/EC2_hg.h5')
                r2_pca[h, i, j] = df_['acc']
                df_ = apply_df_filters(df, dimval=dim, fold=fold, dr_method='LQGCA20')
                assert(df_.shape[0] == 1)
                r2_fca[h, i, j] = df_['acc']

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    accuracy_dca = np.load('/home/akumar/nse/neural_control/data/pca_dca_cv.npz')    
    accuracy_pca = accuracy_dca['accuracy'][0, :, :, 1]

    yvals = r2_pca[0]
    mean = yvals.mean(axis=1)
    std = yvals.std(axis=1)
    ax[0].plot(dimvals, mean, c='k', label='PCA')
    ax[0].fill_between(dimvals, mean-std/np.sqrt(5), mean+std/np.sqrt(5), color='k', alpha=.25)

    yvals = r2_fca[0]
    mean = yvals.mean(axis=1)
    std = yvals.std(axis=1)
    ax[0].plot(dimvals, mean, c='r', label='FCCA')
    ax[0].fill_between(dimvals, mean-std/np.sqrt(5), mean+std/np.sqrt(5), color='r', alpha=.25)

    ax[0].set_xlabel('Projected dim. (out of 86)', fontsize=14)
    ax[0].set_ylabel('Accuracy', fontsize=14)
    ax[0].axhline(null_accuracy[0, ..., 1].mean(axis=0), 0, 1, color='b', ls='--', label='chance')
    ax[0].set_yticks([0, .1, .2, .3])
    ax[0].tick_params(axis='both', labelsize=12)
    ax[0].legend(fontsize=12)
    ax[0].set_title(data_files[0])

    yvals = r2_pca[1]
    mean = yvals.mean(axis=1)
    std = yvals.std(axis=1)
    ax[1].plot(dimvals, mean, c='k', label='PCA')
    ax[1].fill_between(dimvals, mean-std/np.sqrt(5), mean+std/np.sqrt(5), color='k', alpha=.25)

    yvals = r2_fca[1]
    mean = yvals.mean(axis=1)
    std = yvals.std(axis=1)
    ax[1].plot(dimvals, mean, c='r', label='FCCA')
    ax[1].fill_between(dimvals, mean-std/np.sqrt(5), mean+std/np.sqrt(5), color='r', alpha=.25)

    ax[1].set_xlabel('Projected dim. (out of 86)', fontsize=14)
    ax[1].set_ylabel('Accuracy', fontsize=14)
    ax[1].axhline(null_accuracy[1, ..., 1].mean(axis=0), 0, 1, color='b', ls='--', label='chance')
    ax[1].set_yticks([0, .1, .2, .3])
    ax[1].tick_params(axis='both', labelsize=12)
    ax[1].legend(fontsize=12)
    ax[1].set_title(data_files[1])

    # fig.tight_layout()
    fig.savefig('/home/akumar/nse/neural_control/figs/loco_indy_merge/cv_decoding.pdf', bbox_inches='tight', pad_inches=0)
    #plt.savefig('pca_dca_cv.pdf', dpi=300)    


if __name__ == '__main__':
    #fit()
    plot()