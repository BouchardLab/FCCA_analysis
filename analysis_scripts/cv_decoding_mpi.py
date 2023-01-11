import numpy as np
import sys
import os
import itertools
import pickle
from tqdm import tqdm
import time
import pdb

def gen_sbatch(savepath, T, reps, nseeds):

    if not os.path.exists(savepath):
        os.makedirs(savepath)

    combs = itertools.product(T, np.arange(reps), np.arange(nseeds))

    # Each combination of T, rep, and seed requires 3 nodes to run

    total_nodes = 3 * len(T) * reps * nseeds

    with open('cv_sbatch%d.sh' % T[0], 'w') as sb:
        sb.write('#!/bin/bash\n')
        sb.write('#SBATCH --qos=regular\n')
        sb.write('#SBATCH --constraint=knl\n')
        sb.write('#SBATCH --image=docker:akumar25/nersc_ncontrol:latest\n')
        sb.write('#SBATCH -N %d\n' % total_nodes)
        sb.write('#SBATCH -t 04:00:00\n')
        sb.write('#SBATCH --job-name=cv_decoding\n')
        sb.write('#SBATCH --out=%s/cv_decoding.o\n' % savepath)
        sb.write('#SBATCH --error=%s/cv_decoding.e\n' % savepath)
        sb.write('#SBATCH --mail-user=ankit_kumar@berkeley.edu\n')
        sb.write('#SBATCH --mail-type=FAIL\n')

        sb.write('source ~/anaconda3/bin/activate\n')
        sb.write('source activate dyn\n')

        # Critical to prevent threads competing for resources
        sb.write('export OMP_NUM_THREADS=1\n')
        sb.write('export KMP_AFFINITY=disabled\n')

        for comb in combs:
            t, _, seed = comb
            sb.write('srun -N 3 -n 130 shifter --entrypoint python cv_decoding_mpi.py %s %d %d &\n' % (savepath, t, seed))

def manual_onehot_encode(y):

    cvs = np.unique(y)
    
    yoh = np.zeros(y.size)
    for i in range(y.size):
        yoh[i] = list(cvs).index(y[i])

    return yoh
        
if __name__ == '__main__':
    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegressionCV
    import h5py

    from mpi4py import MPI
    from sklearn.preprocessing import OneHotEncoder    
    from sklearn.decomposition import PCA
    from sklearn.model_selection import KFold
    from dca_research.lqg import LQGComponentsAnalysis as LQGCA

    comm = MPI.COMM_WORLD

    if comm.rank == 0:
        #fname = '/mnt/Secondary/data/cv/EC2_hg.h5'
        fname = os.environ['SCRATCH'] + '/cv/EC2_hg.h5'
        with h5py.File(fname, 'r') as f:
            X = np.squeeze(f['X'][:])
            y = f['y'][:]
        n_trials, n_time, n_ch = X.shape
        cvs = np.unique(y)
    else:
        X = None
        y = None

    X = comm.bcast(X)
    y = comm.bcast(y)

    #encoder = OneHotEncoder()
    #yoh = encoder.fit_transform(np.reshape(y, (-1, 1)))
    yoh = manual_onehot_encode(y)

    savepath = sys.argv[1]
    T = int(sys.argv[2])
    lqg_seed = int(sys.argv[3])


    trte = StratifiedKFold(n_splits=10, shuffle=True, random_state=20200831)
    train_test_idxs = list(trte.split(X, y))
    # Split dims and folds across ranks
    dims = np.concatenate([np.arange(1, 7), np.arange(7, 14, 2), [20, 25, 30]])
    tasks = list(itertools.product(dims, np.arange(10)))
    tasks = np.array_split(tasks, comm.size)[comm.rank]

    
    accuracy = np.zeros((len(tasks), 4))

    for ii, task in enumerate(tasks):
        t0 = time.time()
        d, fold_idx = task
        print('rank %d working on (fold %d, dim %d)' % (comm.rank, fold_idx, d))

        train_idx = train_test_idxs[fold_idx][0]
        test_idx = train_test_idxs[fold_idx][1]

        Xtrain = X[train_idx]
        ytrain = yoh[train_idx]
        Xtest = X[test_idx]
        ytest = yoh[test_idx]
        # dca_model = DCA(T=T, d=dims.max())
        # dca_model.estimate_data_statistics(Xtrain)
        lqg_model = LQGCA(T=T, d=d, rng_or_seed=lqg_seed)
        lqg_model.estimate_data_statistics(Xtrain)
        pca_model = PCA(d)
        Xtrain_pca = pca_model.fit_transform(Xtrain.reshape(Xtrain.shape[0], -1))
        Xtest_pca = pca_model.transform(Xtest.reshape(Xtest.shape[0], -1))
        print('fit PCA') 
        lr_model = LogisticRegressionCV(cv=5, multi_class='multinomial', max_iter=500)
        lr_model.fit(Xtrain_pca, ytrain)
        print('fit PCA LR')
        accuracy[ii, 0] = lr_model.score(Xtrain_pca, ytrain)
        accuracy[ii, 1] = lr_model.score(Xtest_pca, ytest)
        
        lqg_model.fit_projection(d=d, n_init=20)
        print('fit LQG')        

        Xtrain_dca = lqg_model.transform(Xtrain).reshape(Xtrain.shape[0], -1)
        Xtest_dca = lqg_model.transform(Xtest).reshape(Xtest.shape[0], -1)
        lr_model = LogisticRegressionCV(cv=5, multi_class='multinomial', max_iter=500)
        lr_model.fit(Xtrain_dca, ytrain)
        print('fit LQG LR')
        accuracy[ii, 2] = lr_model.score(Xtrain_dca, ytrain)
        accuracy[ii, 3] = lr_model.score(Xtest_dca, ytest)

        lqg_coef = lqg_model.coef_
        pca_coef = pca_model.components_.T[:, 0:d]
        fca_scores = np.zeros(2)
        fca_scores[0] = -1 * np.max(lqg_model.scores)
        fca_scores[1] = lqg_model.score(coef=pca_model.components_.T[:, 0:d])

        # pca_scores = np.zeros(2)
        # pca_scores[0] = np.sum(pca_model.explained_variance_ratio_)
        # pca_scores[1] = np.trace(np.cov(Xtrain.reshape(Xtrain.shape[0], -1)) @ lqg_coef )

        # Save coefficients as well

        print('rank %d completed in %f' % (comm.rank, time.time() - t0))
        with open('%s/rank%d.pkl' % (savepath, comm.rank), 'ab') as f:
            f.write(pickle.dumps(task))
            f.write(pickle.dumps(accuracy))
            f.write(pickle.dumps(lqg_coef))
            f.write(pickle.dumps(pca_coef))
            f.write(pickle.dumps(fca_scores))
    #        f.write(pickle.dumps(pca_scores))
