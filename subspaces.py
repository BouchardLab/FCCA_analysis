from tkinter import E
from xml.etree.ElementTree import QName
import quadprog
from tqdm import tqdm
import numpy as np
from numpy.lib.stride_tricks import as_strided
import scipy 
import pdb

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV

from riccati import discrete_generalized_riccati, check_gdare
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

# Calculate the innovation covariance using the forward Kalman filter
def filter_log_likelihood(y, A, C , Cbar, L0=None):

    if L0 is None:
        L0 = np.cov(y, rowvar=False)

    # Initalize the state covariance at 0 and propagate the Riccati equation
    P = np.zeros(A.shape)
    
    # Innovation covariance
    SigmaE = np.zeros((y.shape[0],) + L0.shape)
    # Innovations
    e = np.zeros(y.shape)

    # Initialization
    e[0, :] = y[0, :]
    SigmaE[0] = L0

    xhat = np.zeros(A.shape[0])

    # Solve steady state, keep track of convergence - continuous iteration is not numerically stable
    # scipy's requirements on hermiticity are quite stringent...
    L0 = 0.5 * (L0 + L0.T)
    Pinf = scipy.linalg.solve_discrete_are(A.T, C.T, np.zeros(A.shape), L0, s=Cbar.T)

    # Propagation
    tol = 1e-5
    norm_diff_trace = []
    for i in range(1, y.shape[0]):
        if norm_diff < tol:
            P = Pinf
        else:
            PP = discrete_generalized_riccati(P, A, C, Cbar, L0)
            norm_diff = np.linalg.norm(P - PP)
            norm_diff_trace.append(norm_diff)
            P = PP

        R = L0 - C @ P @ C.T

        # try:
        #     assert(np.all(np.linalg.eigvals(R) >= 0))
        # except:
        #     pdb.set_trace()
        if np.any(np.isinf(P)) or np.any(np.isnan(P)):
            pdb.set_trace()
        
        K = (Cbar.T - A @ P @ C.T) @ np.linalg.pinv(R)
        
        SigmaE[i] = R
        e[i] = y[i] - C @ xhat
        xhat = A @ xhat + K @ e[i]

    T = y.shape[0]

    # Note the expression given by Hannan and Deistler is the *negative* of the log likelihood
    return -1/(2) * sum([np.linalg.slogdet(SigmaE[j])[1] for j in range(T)]) - 1/(2) * sum([e[j] @ np.linalg.pinv(SigmaE[j]) @ e[j] for j in range(T)])\
            -L0.shape[0]/2 * np.log(2 * np.pi)

# For number of parameters in a state space model, see: Uniquely identifiable state-space and ARMA parametrizations for multivariable linear systems
def BIC(ll, state_dim, obs_dim, n_samples):
    #1/T normalization assumes the likelihood 
    return -2*ll + np.log(n_samples) * (2 * state_dim * obs_dim)

def AIC(ll, state_dim, obs_dim, **kwargs):
    return -2 * ll + 2 * (2 * state_dim * obs_dim)

## These criteria are described here: Order estimation for subspace methods
# They rely on the canonical correlation coefficients
def NIC_BIC(cc, state_dim, obs_dim, n_samples):
    pass
def NIC_AIC(cc, state_dim, obs_dim, **kwargs):
    pass
def SVIC_BIC(cc, state_dim, obs_dim, n_samples):
    return cc + np.log(n_samples) * (2 * state_dim * obs_dim)
def SVIC_AIC(cc, state_dim, obs_dim, **kwargs):
    return cc + 2 * (2 * state_dim * obs_dim)

score_fn_dict = {'BIC': BIC, 'AIC':AIC, 'NIC_BIC':NIC_BIC, 
                 'NIC_AIC':NIC_AIC, 'SVIC_BIC':SVIC_BIC, 'SVIC_AIC':SVIC_AIC}

# Method 1: Use factorization of the residuals in OLS fits to
# ensure Positive Real lemma is satisfied
def pr_correction_method1(A, C, Cbar, L0, rho_A, rho_C):

    Q = np.cov(rho_A, rowvar=False)
    S = 1/rho_A.shape[0] * rho_A.T @ rho_C
    R = np.cov(rho_C, rowvar=False)

    try:
        B = np.linalg.cholesky(Q)
    except:
        # Add some white noise
        Q += 1e-8 * np.eye(Q.shape[0])
        B = np.linalg.cholesky(Q)
    try:
        D = np.linalg.cholesky(R)
    except:
        R += 1e-8 * np.eye(R.shape[0])
        D = np.linalg.cholesky(R)

    P = scipy.linalg.solve_discrete_lyapunov(A, Q)
    
    L0 = C @ P @ C.T + R
    Cbar = (A @ P @ C.T + S).T
    return L0, B, D, Cbar

# Solve a LMI so that modified Cbar (and possibly L0) satisfy
# the Positive Real Lemma
def pr_correction_method2(A, C, Cbar, L0):
    pass

#### Estimators for A, C, Cbar

class OLSEstimator():

    def __init__(self, T):
        self.T = T
        self.state_lr = LinearRegression(fit_intercept=False)
        self.obs_lr = LinearRegression(fit_intercept=False)

    def fit(self, y, Xt, Xt1, return_residuals=False):
        # Regression of predictor variables
        A = self.state_lr.fit(Xt.T, Xt1.T).coef_
        C = self.obs_lr.fit(Xt.T, y).coef_
        Cbar = 1/y.shape[0] * (y.T @ Xt1.T)

        if return_residuals:
            Xt1pred = self.state_lr.predict(Xt.T)
            rho_A = Xt1.T - Xt1pred
            ypred = self.obs_lr.predict(Xt.T)
            rho_C = y - ypred
            return A, C, Cbar, rho_A, rho_C
        else:
            return A, C, Cbar

class RidgeEstimator():

    def __init__(self, T):
        self.T = T

        self.state_lr = RidgeCV(alphas=np.logspace(-2, 1, num=10), fit_intercept=False)
        self.obs_lr = RidgeCV(alphas=np.logspace(-2, 1, num=10), fit_intercept=False)

    def fit(self, y, Xt, Xt1, return_residuals=False):

        # Regression of predictor variables
        A = self.state_lr.fit(Xt.T, Xt1.T).coef_
        # Be careful to match indices here
        C = self.obs_lr.fit(Xt.T, y[self.T-1:-1, :]).coef_
        Cbar = 1/y.shape[0] * (y.T @ Xt1.T)
        # Same thing but backwards in time
        if return_residuals:
            Xt1pred = self.state.predict(Xt.T)
            rho_A = Xt1.T - Xt1pred
            ypred = self.obs_lr.predict(Xt.T)
            rho_C = y - ypred
            return A, C, Cbar, rho_A, rho_C
        else:   
            return A, C, Cbar

# Method of Siddiqi et. al.
class IteratedStableEstimator():

    def __init__(self, T, interp_iter=10, obs_regressor='OLS'):
        self.T = T
        self.interp_iter = interp_iter

        # The observational regressions are unchanged
        if obs_regressor =='OLS':
            self.obs_lr = LinearRegression(fit_intercept=False)
        else:
            self.obs_lr = RidgeCV(alphas=np.logspace(-2, 1, num=10), fit_intercept=False)
        # Use these as initial estimates
        self.state_lr = LinearRegression(fit_intercept=False)

        # Add some slack from the stability boundary
        self.check_stability = lambda A: np.all(np.abs(np.linalg.eigvals(A)) < 0.99)

    def solve_qp(self, A, x0, x1):
        # Setup the quadprog    
        P = 0.5 * np.kron(np.eye(A.shape[0]), x0 @ x0.T)

        # It may be necessary to regularize the diagonal in order to 
        # render P positive definite
        eigvals = np.linalg.eigvals(P)
        if np.any(np.isclose(eigvals, 0)):
            P += 1e-6 * np.eye(P.shape[0])

        P = P.astype(np.double)

        # This coincides with the vectorize operator
        q = 0.5 * (x0 @ x1.T).flatten('F')
        q = q.astype(np.double)

        U, S, Vh = np.linalg.svd(A)

        # Constraint vector
        G = -1*np.outer(U[:, 0], Vh[0, :]).flatten('F')[:, np.newaxis]
        G = G.astype(np.double)
        h = -1*np.array([1]).astype(np.double)

        # Solve QP 
        a = quadprog.solve_qp(P, q, G, h, 0)[0]
    
        A0 = A            
        A1 = np.reshape(a, A.shape, order='F')

        while not self.check_stability(A1):
            # Append to constraints
            U, S, Vh = np.linalg.svd(A1)
            g = -1*np.outer(U[:, 0], Vh[0, :]).flatten('F')[:, np.newaxis]
            G = np.hstack([G, g]).astype(np.double)
            h = -1*np.ones(G.shape[1]).astype(np.double)               
            # Solve QP 
            a = quadprog.solve_qp(P, q, G, h, 0)[0]
            A0 = A1
            A1 = np.reshape(a, A.shape, order='F')

        # Binary search to the stability boundary
        gamma = 0.5
        for i in range(self.interp_iter):    
            A_ = gamma * A1 + (1 - gamma) * A0
            if self.check_stability(A_):
                # Bring A_ closer to A0 (gamma -> 0)
                gamma = gamma - 0.5**(i + 1)
            else:
                # Bring A_ closer to A1 (gamma -> 1)
                gamma = gamma + 0.5**(i + 1)

        return A_

    def fit(self, y, Xt, Xt1, return_residuals=False):

        C = self.obs_lr.fit(Xt.T, y).coef_
        Cbar = 1/y.shape[0] * (y.T @ Xt1.T)

        # First, do ordinary OLS and check for stability. If stable, then return
        A = self.state_lr.fit(Xt.T, Xt1.T).coef_

        if not self.check_stability(A):
            A = self.solve_qp(A, Xt, Xt1)

        if return_residuals:
            # Prediction done using the stabilized A
            Xt1pred = (A @ Xt).T
            rho_A = Xt1.T - Xt1pred
            ypred = self.obs_lr.predict(Xt.T)
            rho_C = y - ypred
            return A, C, Cbar, rho_A, rho_C
        else:
            return A, C, Cbar

class SubspaceIdentification():

    def __init__(self, T=3, estimator=OLSEstimator, score='BIC', **estimator_kwargs):

        self.T = T
        self.estimator = estimator(T, **estimator_kwargs)
        self.score = score

    def identify(self, y, T=None, min_order=None, max_order=None):
        
        if T is None:
            T = self.T
        if min_order is None:
            min_order = y.shape[1]
        if max_order is None:
            max_order = T * y.shape[1]

        orders = np.arange(min_order, max_order)
        # Score in forward and reverse time
        scores = np.zeros((orders.size, 2))

        # Should have an option to "not Toeplitzify"
        ccm = estimate_autocorrelation(y, 2*T + 2)

        # Get Toeplitz, Hankel structures
        hankel_toeplitz = self.form_hankel_toeplitz(ccm, T)

        for i, order in enumerate(orders):
            # Factorize
            zt, zt1, zbart, zbart1 = self.get_predictor_space(y, hankel_toeplitz, T, int(order))
            # Identify (forward time)
            A, C, Cbar, rho_A, rho_C = self.estimator.fit(y[self.T-1:-1, :], zt, zt1, return_residuals=True)

            # Need to correct positive realness
            if not check_gdare(A, C, Cbar, ccm[0]):
                L0, B, D, Cbar = pr_correction_method1(A, C, Cbar, ccm[0], rho_A, rho_C)
                # Make sure it worked
                assert(check_gdare(A, C, Cbar, L0))
            else:
                L0 = ccm[0]
            # Identify (reverse time)
            # At, Cbarrev, Crev = self.estimator.fit(y[1:-self.T+1,:], zbart, zbart1)
            # if not check_gdare(At, Cbbar, Cbar, ccm[0]):
            #     L0rev, Brev, Drev, Cbarrev = pr_correction_method1(At, Cbarrev, Crev, L0)
                
            # # Score
            if self.score in ['AIC', 'BIC']:
                llfwd = filter_log_likelihood(y, A, C, Cbar, L0)
                # llrev = filter_log_likelihood(y, At.T, Crev, Cbarrev)
                scores[i, 0] = score_fn_dict[self.score](llfwd, A.shape[0], C.shape[0], n_samples=y.shape[0])
                # scores[i, 1] = score_fn_dict[self.score](llrev, A.shape[0], Crev.shape[0], n_samples=y.shape[0])

            elif self.score in ['SVIC_BIC', 'SVIC_AIC']:
                # Pass in the first canonical correlation coefficient beyond the current model order
                if i < order.size - 1:
                    scores[i, :] = score_fn_dict[self.score](hankel_toeplitz[1][i + 1], A.shape[0], 
                                                        C.shape[0], n_samples=y.shape[0])
                else:
                    scores[i, :] = np.inf
        best_score_idx = np.argmin(scores[:, 0])
#        best_score_idx = np.unravel_index(np.argmin(scores[:, 0]), scores.shape)
        order = orders[best_score_idx]

        # Re-estimate
        zt, zt1, zbart, zbart1 = self.get_predictor_space(y, hankel_toeplitz, T, int(order))
        # In forward time
        A, C, Cbar, rho_A, rho_C = self.estimator.fit(y[self.T-1:-1, :], zt, zt1, return_residuals=True)
        # Correct
        if not check_gdare(A, C, Cbar, ccm[0]):
            L0, B, D, Cbar = pr_correction_method1(A, C, Cbar, L0, rho_A, rho_C)
        # else:
        #     pdb.set_trace()
        #     # # In reverse time:
        #     # At, Cbar, C = self.estimator.fit(y[1:-self.T+1,:], zbart, zbart1)
        #     # A = At.T

        return A, C, Cbar, scores

    def form_hankel_toeplitz(self, ccm, T):

        # T quantities
        Tm, Tp = gen_toeplitz_from_blocks(ccm, order=T)

        # T + 1 quantities
        Tm1, Tp1 = gen_toeplitz_from_blocks(ccm, order=T + 1)
        Lm1 = np.linalg.cholesky(Tm1)
        Lp1 = np.linalg.cholesky(Tp1)

        H1 = gen_hankel_from_blocks(ccm, order1=T + 1, order2=T + 1)
        H1norm = np.linalg.inv(Lp1) @ H1 @ np.linalg.inv(Lm1).T
        Ut1, St1, Vht1 = np.linalg.svd(H1norm)

        return Ut1, St1, Vht1, Tm1, Lm1, Tp1, Lp1, Tm, Tp

    # Follow chapter 13 of LP except for in how we form the autocorrelation matrices
    def get_predictor_space(self, y, hankel_toeplitz, T, truncation_order):

        m = y.shape[1]

        Ut1, St1, Vht1, Tm1, Lm1, Tp1, Lp1, Tm, Tp = hankel_toeplitz

        St1 = np.diag(St1[0:truncation_order])
        Ut1 = Ut1[:, 0:truncation_order]
        Vht1 = Vht1[0:truncation_order, :]

        Sigmat1 = Lp1 @ Ut1 @ scipy.linalg.sqrtm(St1)
        Sigmabart1 = Lm1 @ Vht1.T @ scipy.linalg.sqrtm(St1) 
        
        Sigmat = Sigmat1[:-m, :] 
        Sigmabart = Sigmabart1[:-m, :]

        # Form the predictor spaces
        ypt1 = form_lag_matrix(y, T + 1).T
        ymt1 = flip_blocks(ypt1, T + 1)

        Xt = Sigmabart.T @ np.linalg.inv(Tm) @ ymt1[m:, :]
        Xt1 = Sigmabart1.T @ np.linalg.inv(Tm1) @ ymt1

        Xrevt = Sigmat.T @ np.linalg.inv(Tp) @ ypt1[m:, :]
        Xrevt1 = Sigmat1.T @ np.linalg.inv(Tp1) @ ypt1

        return Xt, Xt1, Xrevt, Xrevt1

class CVSubspaceIdentification():
    pass
