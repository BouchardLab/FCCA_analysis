import numpy as np
import scipy 
import pdb

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeRegression

from dca.cov_util import form_lag_matrix
from neurosim.utils.riccati import discrete_generalized_riccati
from cov_estimation import estimate_autocorrelation 


# Arrange cross-covariance matrices in Hankel form
# Allow for rectangular Hankel matries (required for subspace identification)
def gen_hankel_from_blocks(blocks, order1=None, order2=None, shift=0):

    if order1 is None or order2 is None:
        order = int(blocks.shape[0]/2) - shift
        order1 = order2 = order
    hankel_blocks = [[blocks[i + j + 1 + shift, ...] for j in range(order1)] for i in range(order2)]
    return np.block(hankel_blocks)

# Allow for rectangular Toeplitz matrices (required for subspace identification)
def gen_toeplitz_from_blocks(blocks, order=None):
    
    if order is None:
        order = int(blocks.shape[0])

    toeplitz_block_index = lambda idx: blocks[idx, ...] if idx >= 0 else blocks[-1*idx, ...].T
   
    toeplitz_blocks = [[toeplitz_block_index(j - i) for j in range(order)] for i in range(order)]
    T1 = np.block(toeplitz_blocks)

    toeplitz_blocks = [[toeplitz_block_index(i - j) for j in range(order)] for i in range(order)]
    T2 = np.block(toeplitz_blocks)

    return T1, T2

class OLSEstimator():

    def __init__(self):
        pass

    def fit(y, zt, zt1, zbart, zbart1):

        # Form an augmented forward time and reverse time feature vector, solve using OLS
        Xforward = np.hstack([)
        
class RidgeEstimator():

# Method of Siddiqi et. al.
class IteratedStableEstimator():


class SubspaceIdentification():

    def __init__(self, ar_order=3,maxent_extend=True, 
                 stability_regularization=True):

        self.ar_order = ar_order
        self.maxent_extend = maxent_extend
        self.stability_regularization = stability_regularization
    
    def identify(self, y, T, min_order=None, max_order=None):

        if min_order is None:
            min_order = y.shape[1]
        if max_order is None:
            max_order = T
        
        orders = np.linspace(min_order, max_order, 2)
        scores = np.zeros(order.size)
        for i, order in enumerate(orders):
            # Factorize
            zt, zt1, zbart, zbart1 = self.get_predictor_space(y, T, order)
            # Identify
            A, C, Cbar = self.estimator.fit(ymt1, ymt, ypt1, zt, zt1, zbart, zbart1)
            # Predict
            ypred = self.estimator.predict()
            # Score
            scores[i] = self.score(y, ypred)

    # Identify a predictor space from autocovariance sequence. Conceptually, we follow chapter 12 of Lindquist and Picci
    # in that we use Hankel/Toeplitz matrices formed from pre-estimated autocorrelation sequences (which may be pre-regularized).
    # However, in some of our manipulations we follow chapter 13 of LP. For example, we normalize the larger Hankel matrix and obtain 
    # shifted Hankel matrices and shifted coherent factorizations by truncating this larger Hankel matrix appropriately.
    def get_predictor_space(self, y, T, truncated_order):

        N = y.shape[0]
        m = y.shape[1]

        # Return the autocorrelation sequence from 0 to 2t + 1
        ccm = estimate_autocorrelation(y, 2*T + 1)
        # if maxent_extend:
        #     ccm = maxent_extend(y, ccm, 10 * ar_order)

        # Generate Hankel matrix with lambda_1 in the top left corner and 2T in the bottom right corner
        Hlarge = gen_hankel_from_blocks(ccm, order1=T + 1, order2=T + 1)
        
        # Chopping off the last row and column
        Htt = Hlarge[:-m, :-m]
        # Shifted Hankel matrices
        Htt1 = Hlarge[:-m, :]
        Ht1t = Hlarge[:, :-m]

        # Toeplitz matrices for normalization
        Tmt1, Tpt1 = gen_toeplitz_from_blocks(ccm, order=T+1)

        try:
            Lmt1 = np.linalg.cholesky(Tmt1)
            Lpt1 = np.linalg.cholesky(Tpt1)
        except np.linalg.LinAlgError:
            # Add white noise
            ccm[0] += 1e-4 * np.eye(ccm.shape[0])
            Tmt1, Tpt1 = gen_toeplitz_from_blocks(ccm, order=T+1)
            Lmt1 = np.linalg.cholesky(Tmt1)
            Lpt1 = np.linalg.cholesky(Tpt1)


        # Normalized Hankel matrix
        Hnorm = np.linalg.inv(Lpt1) @ Hlarge @ np.linalg.inv(Lmt1).T
        # SVD
        Ut1, St1, Vht1 = np.linalg.svd(Hnorm)
        # Balanced truncation
        St1 = np.diag(St1[0:truncated_order])
        Ut1 = Ut[:, 0:St1.shape[0]]
        Vht1 = Vht[0:St1.shape[0], :]

        # Obtain the cholesky factors of length T by truncating the larger ones. For data generated
        # from a rational system, this is basically the same thing
        Lmt = Lmt1[:-m, :-m]
        Lpt = Lpt1[:-m, :-m]

        # Normalized shifted Hankel matrices. Note the typo in 12.124 (missing inverse on Lpt)
        Htt1norm = np.linalg.inv(Lpt) @ Htt1 @ np.linalg.inv(Lmt1).T
        Ht1tnorm = np.linalg.inv(Lpt1) @ Ht1t @ np.linalg.inv(Lmt).T

        # Constructability/Observability Operators
        Sigmat1 = Ut1 @ scipy.linalg.sqrtm(St1)
        Sigmat1bar = Vht1.T @ scipy.linalg.sqrtm(St1)

        # Truncate
        Sigmat = Sigmat1[m:, :]
        Sigmatbar = Sigmat1bar[m:, :]

        # Form a basis in the predictor space X^-_t and X^-_{t + 1}. Note that it is very important that this is 
        # done *coherently*
        ymt1 = form_lag_matrix(y, T + 1)
        ymt = form_lag_matrix(y, T)

        # Time reversed
        yrev = np.flipud(y)
        ypt1 = form_lag_matrix(yrev, T + 1)
        ypt = form_lag_matrix(yrev, T)

        zt = np.array([Sigmatbar.T @ np.linalg.inv(Lmt) @ ymt[j, :] for j in range(ymt.shape[0])])
        zt1 = np.array([Sigmat1bar.T @ np.linalg.inv(Lmt1) @ ymt1[j, :] for j in range(ymt1.shape[0])])

        zbart = np.array([Sigmatbar.T @ np.linalg.inv(Lpt) @ ypt[j, :] for j in range(ypt.shape[0])])
        zbart1 = np.array([Sigmat1bar.T @ np.linalg.inv(Lpt1) @ ypt1[j, :] for j in range(ypt1.shape[0])])

        return zt, zt1, zbart, zbart1

