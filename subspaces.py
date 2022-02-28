import numpy as np
from numpy.lib.stride_tricks import as_strided
import scipy 
import pdb

from sklearn.linear_model import LinearRegression
#from sklearn.linear_model import RidgeRegression

from neurosim.utils.riccati import discrete_generalized_riccati
from cov_estimation import estimate_autocorrelation 

def form_lag_matrix(X, T, stride=1, stride_tricks=True, rng=None, writeable=False):
    """Form the data matrix with `T` lags.

    Parameters
    ----------
    X : ndarray (n_time, N)
        Timeseries with no lags.
    T : int
        Number of lags.
    stride : int or float
        If stride is an `int`, it defines the stride between lagged samples used
        to estimate the cross covariance matrix. Setting stride > 1 can speed up the
        calculation, but may lead to a loss in accuracy. Setting stride to a `float`
        greater than 0 and less than 1 will random subselect samples.
    rng : NumPy random state
        Only used if `stride` is a float.
    stride_tricks : bool
        Whether to use numpy stride tricks to form the lagged matrix or create
        a new array. Using numpy stride tricks can can lower memory usage, especially for
        large `T`. If `False`, a new array is created.
    writeable : bool
        For testing. You should not need to set this to True. This function uses stride tricks
        to form the lag matrix which means writing to the array will have confusing behavior.
        If `stride_tricks` is `False`, this flag does nothing.

    Returns
    -------
    X_with_lags : ndarray (n_lagged_time, N * T)
        Timeseries with lags.
    """
    if not isinstance(stride, int) or stride < 1:
        if not isinstance(stride, float) or stride <= 0. or stride >= 1.:
            raise ValueError('stride should be an int and greater than or equal to 1 or a float ' +
                             'between 0 and 1.')
    N = X.shape[1]
    frac = None
    if isinstance(stride, float):
        frac = stride
        stride = 1
    n_lagged_samples = (len(X) - T) // stride + 1
    if n_lagged_samples < 1:
        raise ValueError('T is too long for a timeseries of length {}.'.format(len(X)))
    if stride_tricks:
        X = np.asarray(X, dtype=float, order='C')
        shape = (n_lagged_samples, N * T)
        strides = (X.strides[0] * stride,) + (X.strides[-1],)
        X_with_lags = as_strided(X, shape=shape, strides=strides)
    else:
        X_with_lags = np.zeros((n_lagged_samples, T * N))
        for i in range(n_lagged_samples):
            X_with_lags[i, :] = X[i * stride:i * stride + T, :].flatten()
    if frac is not None:
        rng = check_random_state(rng)
        idxs = np.sort(rng.choice(n_lagged_samples, size=int(np.ceil(n_lagged_samples * frac)),
                                  replace=False))
        X_with_lags = X_with_lags[idxs]

    return X_with_lags

# Mimic flipud and fliplr but take the 
def flip_blocks(A, d, axis=0):
    if axis == 1:
        A = A.T

    Ablocks = np.array_split(A, d, axis=0)
    Ablocks.reverse()

    Aflipped = np.vstack(Ablocks)

    if axis == 1:
        return Aflipped.T
    else:
        return Aflipped

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

    def __init__(self, T):
        self.T = T
        self.linregressor = LinearRegression(fit_intercept=False)

    def fit(self, y, zt, zt1, zbart, zbart1):
        A1 = 1/(zt1.shape[0]) * zt1.T @ zt @ np.linalg.inv(np.cov(zt.T))
        A2 = self.linregressor.fit(zt, zt1).coef_
        C = self.linregressor.fit(zt, y[self.T:]).coef_

        # Separate A.T and Cbar
        At = self.linregressor.fit(zbart, zbart1).coef_
        Cbar = self.linregressor.fit(zbart, y[self.T:]).coef_

        return A, At, C, Cbar

class RidgeEstimator():
    pass
# Method of Siddiqi et. al.
class IteratedStableEstimator():
    pass

class SubspaceIdentification():

    def __init__(self, T=3, estimator=OLSEstimator):

        self.T = T
        self.estimator = OLSEstimator(T)

    def identify(self, y, T=None, min_order=None, max_order=None):
        
        if T is None:
            T = self.T
        if min_order is None:
            min_order = y.shape[1]
        if max_order is None:
            max_order = T * y.shape[1]
        
        orders = np.arange(min_order, max_order)
        scores = np.zeros(orders.size)
       
        for i, order in enumerate(orders):
            # Factorize
            zt, zt1, zbart, zbart1 = self.get_predictor_space(y, T, int(order))
            # Identify
            A, At, C, Cbar = self.estimator.fit(y, zt, zt1, zbart, zbart1)
            # # Predict
            # ypred = self.estimator.predict()
            # # Score
            # scores[i] = self.score(y, ypred)
        return A, At, C, Cbar


    # Identify a predictor space from autocovariance sequence. Conceptually, we follow chapter 12 of Lindquist and Picci
    # in that we use Hankel/Toeplitz matrices formed from pre-estimated autocorrelation sequences (which may be pre-regularized).
    # However, in some of our manipulations we follow chapter 13 of LP. For example, we normalize the larger Hankel matrix and obtain 
    # shifted Hankel matrices and shifted coherent factorizations by truncating this larger Hankel matrix appropriately.
    def get_predictor_space(self, y, T, truncation_order):

        M = y.shape[0]
        m = y.shape[1]
        N = M - 2 * T - 2

        # Estimate *biased* autocorrelation in line with eq. 13.24 and 13.30. Do *not* toeplitzify in the way that DCA does it
        ypt1 = form_lag_matrix(y, T + 1).T
        ymt1 = flip_blocks(ypt1, T + 1)
        
        ypt = form_lag_matrix(y, T).T
        ymt = flip_blocks(ypt, T)

        Hlarge = 1/(N + 1) * np.dot(ypt1[:, T+1:], ymt1[:, :N + 1].T)
        Tmt1 = 1/(N + 1) * np.dot(ymt1[:, :N+1], ymt1[:, :N+1].T)
        Tpt1 = 1/(N + 1) * np.dot(ypt1[:, T+1:], ypt1[:, T+1:].T)

# #         # Return the autocorrelation sequence from 0 to 2t + 1
# # #        ccm = estimate_autocorrelation(y, 2*T + 2)
# #         ccm = calc_cross_cov_mats_from_data(y, 2*T + 2)
# #         # if maxent_extend:
# #         #     ccm = maxent_extend(y, ccm, 10 * ar_order)

        # Toeplitz matrices for normalization
#        Tmt1, Tpt1 = gen_toeplitz_from_blocks(ccm, order=T+1)

#         # Generate Hankel matrix with lambda_1 in the top left corner and 2T in the bottom right corner
#         Hlarge = gen_hankel_from_blocks(ccm, order1=T + 1, order2=T + 1)
        
        # Chopping off the last row and column
        Htt = Hlarge[:-m, :-m]
        # Shifted Hankel matrices
        Htt1 = Hlarge[:-m, :]
        Ht1t = Hlarge[:, :-m]

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
        St1 = np.diag(St1[0:truncation_order])
        Ut1 = Ut1[:, 0:St1.shape[0]]
        Vht1 = Vht1[0:St1.shape[0], :]

        # # Obtain the cholesky factors of length T by truncating the larger ones. For data generated
        # # from a rational system, this is basically the same thing
        Lmt = Lmt1[:-m, :-m]
        Lpt = Lpt1[:-m, :-m]

        # # # Normalized shifted Hankel matrices. Note the typo in 12.124 (missing inverse on Lpt)
        # # Htt1norm = np.linalg.inv(Lpt) @ Htt1 @ np.linalg.inv(Lmt1).T
        # # Ht1tnorm = np.linalg.inv(Lpt1) @ Ht1t @ np.linalg.inv(Lmt).T

        # Constructability/Observability Operators
        Sigmat1 = Ut1 @ scipy.linalg.sqrtm(St1)
        Sigmat1bar = Vht1.T @ scipy.linalg.sqrtm(St1)

        # Truncate
        Sigmat = Sigmat1[m:, :]
        Sigmatbar = Sigmat1bar[m:, :]

        zt = Sigmatbar.T @ np.linalg.inv(Lmt) @ ymt
        zt1 = Sigmat1bar.T @ np.linalg.inv(Lmt1) @ ymt1

        # Chop off the last sample
        zt = zt[:, :-1]

        zbart = Sigmat.T @ np.linalg.inv(Lpt) @ ypt
        zbart1 = Sigmat1.T @ np.linalg.inv(Lpt1) @ ypt1

        # Chop off the first sample
        zbart = zbart[:, 1:]

        cov1 = np.cov(zt1)
        cov2 = np.cov(zbart1)

        # Assert finite interval balancing
        assert(np.allclose(cov1, cov2, atol=1e-2))
        assert(np.allclose(np.diag(cov1), np.diag(St1), atol=1e-2))

        return zt.T, zt1.T, zbart.T, zbart1.T

