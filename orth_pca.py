import os, contextlib

import numpy as np
import scipy
import torch
import pdb

import pymanopt
from pymanopt.manifolds import Stiefel, Product
from pymanopt.optimizers import *

class OrthPCA():

    def __init__(self, d1, d2, seed=None, verbose=False):
        self.d1 = d1
        self.d2 = d2
        self.seed = seed
        self.verbose = verbose
        # self.sed = check_random_state(seed)

    def fit(self, X1, X2):
        
        # Seed random generator
        if self.seed is not None:
            np.random.sed(seed=self.seed)

        # Estimate the covariance of X1 and X2
        Sigma1 = np.cov(X1, rowvar=False)
        Sigma2 = np.cov(X2, rowvar=False)

        assert(Sigma1.shape[0] == Sigma2.shape[0])
        n = Sigma1.shape[0]
        assert(self.d1 + self.d2 < n)

        # Diagonalize these covariance matrices
        s1, U1 = scipy.linalg.eigh(Sigma1)
        s2, U2 = scipy.linalg.eigh(Sigma2)

        # Ensure largest to smallest ordering
        s1 = np.sort(s1)[::-1].copy()
        s2 = np.sort(s2)[::-1].copy()

        s1 = torch.tensor(s1)
        s2 = torch.tensor(s2)
        S1 = torch.tensor(Sigma1)
        S2 = torch.tensor(Sigma2)

        # Optimization is done on the product of 2 Stiefel manifolds
        #manifold1 = Stiefel(Sigma1.shape[0], self.d1)
        #manifold2 = Stiefel(Sigma2.shape[0], self.d2)
        #manifold = Product([manifold1, manifold2])
        manifold = Stiefel(n, self.d1 + self.d2) 
        optimizer = SteepestDescent()

        # To solve the constrained optimization, we solve the dual minimization problem, sweeping over several 
        # values of the dual variable. See here for a nice explanation: https://people.eecs.berkeley.edu/~elghaoui/Teaching/EE227A/lecture7.pdf

        @pymanopt.function.pytorch(manifold)
        def loss_fn(V):

            V1 = V[:, 0:self.d1]
            V2 = V[:, self.d1:]
        
            var1 = torch.trace(torch.chain_matmul(V1.t(), S1, V1))
            var2 = torch.trace(torch.chain_matmul(V2.t(), S2, V2))

            return -1*(var1/torch.sum(s1[0:self.d1]) + var2/torch.sum(s2[0:self.d2]))
            # Initialize optimization variables
            # V1 = scipy.stats.ortho_group.rvs(U1.shape[0])[:, 0:self.d1]
            # V2 = scipy.stats.ortho_group.rvs(U2.shape[0])[:, 0:self.d2]

        problem = pymanopt.Problem(manifold, loss_fn)

        if self.verbose:
            result = optimizer.run(problem)
    
        else:
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    result = optimizer.run(problem)

        V1 = result.point[:, 0:self.d1]
        V2 = result.point[:, self.d1:]

        score = result.cost

        # Also document how much of the normalized variance we capture in each space
        var_fraction1 = np.trace(V1.T @ Sigma1 @ V1)/np.trace(Sigma1)
        var_fraction2 = np.trace(V2.T @ Sigma2 @ V2)/np.trace(Sigma2)

        return V1, V2, score, var_fraction1, var_fraction2 