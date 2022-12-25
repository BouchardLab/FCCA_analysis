import itertools
import numpy as np
import scipy
import torch
import sys
import sdeint
import pickle
from dca.methods_comparison import JPCA

from dca_research.lqg import LQGComponentsAnalysis as LQGCA
from soc import stabilize, gen_init_W, stabilize_discrete, comm_mat

from mpi4py import MPI

def gen_activity(tau, W, activ_func, sigma, T, h):

    # f
    def f_(x, t):
        return 1/tau * (-1 * np.eye(W.shape[0]) @ x + W @ activ_func(x))

    # G: linear i.i.d noise with sigma
    def g_(x, t):
        return sigma * np.eye(W.shape[0])

    # Generate random initial condition and then integrate over the desired time period
    tspace = np.linspace(0, T, int(T/h))
    
    x0 = np.random.normal(size=(W.shape[0],))

    return  sdeint.itoSRI2(f_, g_, x0, tspace)    


if __name__ == '__main__':

    comm = MPI.COMM_WORLD
    savepath = sys.argv[1]

    reps = 20
    inner_reps = 10
    M = 100
    p = 0.25
    #gamma = np.array([2])
    g = 2
    R = np.linspace(0.75, 10, 25)
    dt = 1
    d = 6

    tasks = list(itertools.product(np.arange(reps), np.arange(inner_reps), R))
    tasks = np.array_split(tasks, comm.size)[comm.rank]

    phi = np.zeros(len(tasks))
    scores = np.zeros(len(tasks))
    nn = np.zeros((len(tasks), 2))
    jpca_eig = np.zeros((len(tasks), 2))

    Alist = []

    print(len(tasks))

    for i, task in enumerate(tasks):
        rep, inner_rep, r = task
        A = gen_init_W(M, p, g, r, -1)
        eig = np.linalg.eigvals(A)
        if np.max(np.real(eig)) >= 0:
            A = stabilize(A)
            eig = np.linalg.eigvals(A)

        assert(np.max(np.real(eig)) < 0)

        nn[i] = np.linalg.norm(A @ A.T - A.T @ A)
        Alist.append(A)

        # Solve for the exact covarinace function and evaluate it at intervals of dt
        Pi = scipy.linalg.solve_continuous_lyapunov(A, -np.eye(A.shape[0]))
        t_ = [j * dt for j in range(10)]
        cross_covs = [scipy.linalg.expm(tau * A) @ Pi for tau in t_]

        cross_covs_rev = [np.linalg.inv(cross_covs[0]) @ c.T @ np.linalg.inv(cross_covs[0]) for c in cross_covs]

        cross_covs = torch.tensor(cross_covs)
        cross_covs_rev = torch.tensor(cross_covs_rev)

        e, Upca = np.linalg.eig(cross_covs[0])
        eigorder = np.argsort(e)[::-1]
        Upca = Upca[:, eigorder][:, 0:d]

        lqgmodel = LQGCA(d=d, T=4, rng_or_seed=int(inner_rep))
        lqgmodel.cross_covs = cross_covs
        lqgmodel.cross_covs_rev = cross_covs_rev
        # Simulate from the model, apply projection, and then fit jPCA

        x = gen_activity(1, A, lambda x: x, 1, 1000, 1e-1)   

        if np.any(np.isnan(x)):
            jpca_eig[i, ...] = np.nan


        coef_, score = lqgmodel._fit_projection()

        phi[rep, i, j, k] = np.mean(scipy.linalg.subspace_angles(Upca, coef_))
        scores[rep, i, j, k] = score            

        xfca = x @ coef_
        xpca = x @ Upca

        jpca = JPCA(n_components=d, mean_subtract=False)
        jpca.fit(xfca[np.newaxis, :])
        jpca_eig[rep, i, j, k, 0] = np.sum(np.abs(jpca.eigen_vals_))

        jpca = JPCA(n_components=d, mean_subtract=False)
        jpca.fit(xpca[np.newaxis, :])
        jpca_eig[rep, i, j, k, 1] = np.sum(np.abs(jpca.eigen_vals_))

        xfca = x @ coef_
        xpca = x @ Upca

        jpca = JPCA(n_components=d, mean_subtract=False)
        jpca.fit(xfca[np.newaxis, :])
        jpca_eig[i, 0] = np.sum(np.abs(jpca.eigen_vals_))

        jpca = JPCA(n_components=d, mean_subtract=False)
        jpca.fit(xpca[np.newaxis, :])
        jpca_eig[i, 1] = np.sum(np.abs(jpca.eigen_vals_))

    # save
    with open('%s/rank%d.pkl' % (savepath, comm.rank), 'wb') as f:
        f.write(pickle.dumps(tasks))
        f.write(pickle.dumps(Alist))
        f.write(pickle.dumps(phi))
        f.write(pickle.dumps(scores))
        f.write(pickle.dumps(nn))
        f.write(pickle.dumps(jpca_eig))