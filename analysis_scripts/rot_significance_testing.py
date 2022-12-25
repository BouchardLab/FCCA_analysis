import pdb
import random
import string
import os, sys
import numpy as np
import scipy 
import time
import glob
import pickle
from tqdm import tqdm

from dca.methods_comparison import JPCA
from pyuoi.linear_model.var  import VAR

from dca_research.lqg import LQGComponentsAnalysis as LQGCA

def form_companion(w, u=None):
    order = w.shape[0]
    size = w.shape[1]
    I = np.eye(size * (order - 1))
    wcomp = np.block([list(w), [I, np.zeros((size * (order - 1), size))]])

    if u is not None:
        ucomp = [[u]]
        for i in range(order - 1):
            ucomp.append([np.zeros((size, size))])
        ucomp = np.block(ucomp)

        return wcomp, ucomp
    else:
        return wcomp

if __name__ == '__main__':

    # repo_path = sys.argv[1]
    # dpath = sys.argv[2]
    # didx = int(sys.argv[3])
    # results_dir = sys.argv[4]
    repo_path = '/home/akumar/nse/neural_control'
    dpath = '/mnt/Secondary/data/sabes'
    results_dir = '/mnt/Secondary/data/rot'

    sys.path.append(repo_path)
    from loaders import load_sabes

    N_comparisons = int(1e4)

    data_files = glob.glob('%s/indy*' % dpath)

    DIM = 6

    for didx, data_file in enumerate(data_files):
        dat = load_sabes(data_file)

        X = dat['spike_rates'].squeeze()

        #dist_U = np.zeros(N_comparisons)
        jpca_eigs = np.zeros((N_comparisons, DIM))
        scores = np.zeros(N_comparisons)

        lqgmodel = LQGCA(T=4)

        for i in tqdm(range(N_comparisons)):
            V = scipy.stats.ortho_group.rvs(X.shape[1])[:, 0:DIM]
            x = X @ V

            # # Fit OLS(1) model
            # varmodel = VAR(estimator='ols', order=1)
            # varmodel.fit(x)
            # A = form_companion(varmodel.coef_)
            # U, P = scipy.linalg.polar(A)
            # dist_U[i] = np.linalg.norm(A - U)/np.linalg.norm(A)
            # Fit JPCA
            jpca = JPCA(n_components=DIM, mean_subtract=False)
            jpca.fit(x[np.newaxis, ...])

            jpca_eigs[i, :] = jpca.eigen_vals_

            # Calculate the LQGCA score
            scores[i] = lqgmodel.score(coef=V, X=X)

        # To-do:
        # Saving. Push onto NERSC and test/parallelize

        # Generate a unique hash associated with htis file
        hsh = "".join(random.choices(string.ascii_letters, k=26))
        fname = 'didx_%d_%s' % (didx, hsh)
        with open('%s/%s.pkl' % (results_dir, fname), 'wb') as f:
            f.write(pickle.dumps(dist_U))
            f.write(pickle.dumps(scores))
            f.write(pickle.dumps(data_file))
