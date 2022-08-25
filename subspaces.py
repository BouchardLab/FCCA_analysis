from tkinter import E
from xml.etree.ElementTree import QName
import quadprog
from tqdm import tqdm
import numpy as np
from numpy.lib.stride_tricks import as_strided
import scipy 
import pdb
import pickle
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.decomposition import TruncatedSVD

from riccati import discrete_generalized_riccati, discrete_riccati, check_gdare
from cov_estimation import estimate_autocorrelation 

class RiccatiPSDError(Exception):
    pass


# Add some slack from the stability boundary
check_stability = lambda A: np.all(np.abs(np.linalg.eigvals(A)) < 0.99)

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

def check_riccati_ic(Pinf, P0, A, C, Re, Kinf):
    O = scipy.linalg.solve_discrete_lyapunov((A - Kinf @ C).T, C.T @ Re @ C)
    try:
        Op2 = scipy.linalg.cholesky(O)
    except:
        pdb.set_trace()
    eig = np.linalg.eigvals(np.eye(Pinf.shape[0]) + Op2.T @ (P0 - Pinf) @ Op2)

    eig2 = np.linalg.eigvals(np.eye(Pinf.shape[0]) + (P0 - Pinf) @ O)

    if np.all(eig > 0) and not np.any(np.isclose(eig2, 0)):
        return True
    else:
        return False

# Obtain remaining state space parameters from solution of the Riccati equation
def factorize(A, C, Cbar, L0):
    # scipy PSD Check is quite stringent
    L0 = 0.5 * (L0 + L0.T)

    Pinf = scipy.linalg.solve_discrete_are(A.T, -C.T, np.zeros(A.shape), -L0, s=Cbar.T)
    assert(np.all(np.linalg.eigvals(Pinf)) > 0)
    D = scipy.linalg.sqrtm(L0 - C @ Pinf @ C.T)
    B = (Cbar.T - A @ Pinf @ C.T) @ np.linalg.inv(D)
    return B, D

# Calculate the innovation covariance using the forward Kalman filter
#def filter_log_likelihood(y, A, C, Q, R, S):
def filter_log_likelihood(y, A, C, Cbar, L0):
    # Solve steady state, keep track of convergence - continuous iteration is not numerically stable
    # scipy's requirements on hermiticity are quite stringent...
    #R = 0.5 * (R + R.T)
    L0 = 0.5 * (L0 + L0.T)
    #Pinf = scipy.linalg.solve_discrete_are(A.T, C.T, Q, R, s=S)
    Pinf = scipy.linalg.solve_discrete_are(A.T, -C.T, np.zeros(A.shape), -L0, s=Cbar.T)

    if not np.all(np.linalg.eigvals(Pinf) > 0):
        raise RiccatiPSDError

    #Re = R + C @ Pinf @ C.T 
    Re = L0 - C @ Pinf @ C.T
    if not np.all(np.linalg.eigvals(Re) > 0):
        raise RiccatiPSDError

    Kinf = (Cbar.T - A @ Pinf @ C.T) @ np.linalg.inv(Re)
    # Kinf = (A @ Pinf @ C.T + S) @ np.linalg.inv(Re)
    # # Try initializing with the zero matrix. Check the condition Kailath 14.5.24 to make sure the 
    # # iterations will converge. 
    # P0 = scipy.linalg.solve_discrete_lyapunov(A, Q)
    # print(check_riccati_ic(Pinf, P0, A, C, R + C @ Pinf @ C.T, Kinf))
    #     # print('Lyapunov initialization failed!')
    #     # # # Try random initialization
    #     # # while not check_riccati_ic(Pinf, P0, A, C, R + C @ Pinf @ C.T, Kinf):
    #     # #     P0 = np.random.uniform(1, 2, size=A.shape)
    #     # if not check_riccati_ic(Pinf, P0, A, C, R + C @ Pinf @ C.T, Kinf):
 
    #     # print('Lyapunov initialization succeeded!')

    # Innovation covariance
    SigmaE = np.zeros((y.shape[0], y.shape[1], y.shape[1]))
    # Innovations
    e = np.zeros(y.shape)

    # Initialization
    e[0, :] = y[0, :]
    # fix this at some point
    SigmaE[0] = 1
    xhat = np.zeros(A.shape[0])
    # Propagation
    # tol = 1e-5
    # norm_diff_trace = []
    # norm_diff = np.inf

    # What the fuck do youw ant from me you god damn whore

    P = Pinf
    K = Kinf
    #Re = R + C @ P @ C.T
    for i in range(1, y.shape[0]):
        # # if norm_diff < tol:
        # #     P = Pinf
        # # else:
        # P = discrete_riccati(P, A, C, Q, R, S=S)
        # # PP = discrete_generalized_riccati(P, A, C, Cbar, L0)
        # # norm_diff = np.linalg.norm(P - PP)
        # # norm_diff_trace.append(norm_diff)
        # # P = PP

        # Re = R + C @ P @ C.T

        # try:
        #     assert(np.all(np.linalg.eigvals(Re) > 1e-8))
        # except:
        #     with open('test_case5.dat', 'wb') as f:
        #         f.write(pickle.dumps([A, C, Q, R, S, Pinf]))
        #     pdb.set_trace()

        # if np.any(np.isinf(P)) or np.any(np.isnan(P)):
        #     pdb.set_trace()
        
        # K = (A @ P @ C.T + S) @ np.linalg.pinv(Re)
        # if np.any(np.isinf(K)) or np.any(np.isnan(K)):
        #     pdb.set_trace()

        SigmaE[i] = Re
        e[i] = y[i] - C @ xhat
        xhat = A @ xhat + K @ e[i]

    T = y.shape[0]

    # Note the expression given by Hannan and Deistler is the *negative* of the log likelihood
    # Throw away the first 10 samples
    return -1/(2) * sum([np.linalg.slogdet(SigmaE[j])[1] for j in range(10, T)]) - 1/(2) * sum([e[j] @ np.linalg.pinv(SigmaE[j]) @ e[j] for j in range(10, T)])\
            -y.shape[1]/2 * np.log(2 * np.pi)

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

# Alternative positive realness check - Kailath 8.3.2
# This ensures existsence of a psd asymptotic solution, but not uniquenesss (i.e. the psd solution is stabilizing)
# One could additionally check generic controllability (stricter than stabilizability)
def unit_controllability_test(A, C, Q, S, R):
    Fs = A - S @ np.linalg.inv(R) @ C
    Qs = Q - S @ np.linalg.inv(R) @ S.T
    eig = np.linalg.eigvals(Fs)
    return np.any(np.abs(np.abs(eig) - 1) < 1e-5)

# Method 1: Use factorization of the residuals in OLS fits to
# ensure Positive Real lemma is satisfied
def pr_correction_method1(A, C, Cbar, L0, rho_A, rho_C):

    Q = np.cov(rho_A, rowvar=False)
    S = 1/rho_A.shape[0] * rho_A.T @ rho_C
    R = np.cov(rho_C, rowvar=False)

    # try:
    #     B = np.linalg.cholesky(Q)
    # except:
    #     # Add some white noise
    #     Q += 1e-8 * np.eye(Q.shape[0])
    #     B = np.linalg.cholesky(Q)
    # try:
    #     D = np.linalg.cholesky(R)
    # except:
    #     R += 1e-8 * np.eye(R.shape[0])
    #     D = np.linalg.cholesky(R)

    P = scipy.linalg.solve_discrete_lyapunov(A, Q)
    
    L0 = C @ P @ C.T + R
    Cbar = (A @ P @ C.T + S).T
    return L0, Cbar, Q, R, S

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

    def __init__(self, T=None, interp_iter=10, obs_regressor='OLS'):
        self.T = T
        self.interp_iter = interp_iter

        # The observational regressions are unchanged
        if obs_regressor =='OLS':
            self.obs_lr = LinearRegression(fit_intercept=False)
        else:
            self.obs_lr = RidgeCV(alphas=np.logspace(-2, 1, num=10), fit_intercept=False)
        # Use these as initial estimates
        self.state_lr = LinearRegression(fit_intercept=False)

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

        while not check_stability(A1):
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
            if check_stability(A_):
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

        if not check_stability(A):
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

    def __init__(self, T=3, estimator=IteratedStableEstimator, score='BIC', **estimator_kwargs):

        self.T = T
        self.estimator = estimator(T, **estimator_kwargs)
        self.score = score

    def identify(self, y, order, ccm=None, hankel_toeplitz=None, T=None):
        if T is None:
            T = self.T
        if ccm is None:
            # Should have an option to "not Toeplitzify"
            ccm = estimate_autocorrelation(y, 2*T + 2)
        if hankel_toeplitz is None:
            # Get Toeplitz, Hankel structures
            hankel_toeplitz = self.form_hankel_toeplitz(ccm, T)
        # Factorize
        zt, zt1 = self.get_predictor_space(y, hankel_toeplitz, T, int(order))
        # Identify (forward time)
        A, C, Cbar, rho_A, rho_C = self.estimator.fit(y[self.T-1:-1, :], zt, zt1, return_residuals=True)

        # Need to correct positive realness
        if not check_gdare(A, C, Cbar, ccm[0]):
            #print('pr correction employed')
            L0, Cbar, Q, R, S = pr_correction_method1(A, C, Cbar, ccm[0], rho_A, rho_C)
            # B, D = factorize(A, C, Cbar, L0)
        else:
            L0 = ccm[0]
            # Obtain B, D, Q, R from the riccati equation
            B, D = factorize(A, C, Cbar, L0)
            Q = B @ B.T
            R = D @ D.T
            S = B @ D.T
        
        if np.any(np.abs(np.linalg.eigvals(R)) < 1e-12):
            #print('R not > 0!')
            R += 1e-6 * np.eye(R.shape[0])

        # Identify (reverse time)
        # At, Cbarrev, Crev = self.estimator.fit(y[1:-self.T+1,:], zbart, zbart1)
        # if not check_gdare(At, Cbbar, Cbar, ccm[0]):
        #     L0rev, Brev, Drev, Cbarrev = pr_correction_method1(At, Cbarrev, Crev, L0)


        return A, C, Cbar, L0, Q, R, S


    def identify_and_score(self, y, T=None, min_order=None, max_order=None):
        
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
            A, C, Cbar, L0, Q, R, S = self.identify(y, order, ccm, hankel_topelitz)
                
            # # Score
            if self.score in ['AIC', 'BIC']:
                try:
                    #llfwd = filter_log_likelihood(y, A, C, Q, R, S)
                    llfwd = filter_log_likelihood(y, A, C, Cbar, L0)
                except RiccatiPSDError:
                    llfwd = np.inf
                # llfwd2 = pykalman_filter_wrapper(y, A, C, Q, R, S)
                # llrev = filter_log_likelihood(y, At.T, Crev, Cbarrev)
                scores[i, 0] = score_fn_dict[self.score](llfwd, A.shape[0], C.shape[0], n_samples=y.shape[0])
                # scores[i, 1] = score_fn_dict[self.score](llrev, A.shape[0], Crev.shape[0], n_samples=y.shape[0])

            elif self.score in ['SVIC_BIC', 'SVIC_AIC']:
                # Pass in the first canonical correlation coefficient beyond the current model order
                if i <  hankel_toeplitz[1].size - 1:
                    scores[i, :] = score_fn_dict[self.score](hankel_toeplitz[1][i + 1], A.shape[0], 
                                                        C.shape[0], n_samples=y.shape[0])
                else:
                    scores[i, :] = np.inf

        best_score_idx = np.argmin(scores[:, 0])
#        best_score_idx = np.unravel_index(np.argmin(scores[:, 0]), scores.shape)
        order = orders[best_score_idx]

        A, C, Cbar, L0, Q, R, S = self.identify(y, order, ccm, hankel_toeplitz)

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

        # Reverse time estimate is not used

        return Xt, Xt1

class CVSubspaceIdentification():
    pass

class CrossSubspaceIdentification(SubspaceIdentification):

    def identify(self, y, z, order, ccm=None, hankel_toeplitz=None, T=None):
        dim_y = y.shape[1]
        dim_z = z.shape[1]

        if T is None:
            T = self.T
        if ccm is None:
            ccm = estimate_autocorrelation(np.hstack([y,z]), 2*T + 2)
        if hankel_toeplitz is None:
            # Get Toeplitz, Hankel structures
            hankel_toeplitz = self.form_hankel_toeplitz(ccm, T, dim_y)

        xt, xt1 = self.get_predictor_space(y, hankel_toeplitz, T, int(order))

        A, C, Cbar, rho_A, rho_C = self.estimator.fit(np.hstack([y[self.T-1:-1, :], z[self.T-1:-1, :]]), xt, xt1, return_residuals=True)

        # Need to correct positive realness
        if not check_gdare(A, C, Cbar, ccm[0]):
            #print('pr correction employed')
            L0, Cbar, Q, R, S = pr_correction_method1(A, C, Cbar, ccm[0], rho_A, rho_C)
            # B, D = factorize(A, C, Cbar, L0)
        else:
            L0 = ccm[0]
            # Obtain B, D, Q, R from the riccati equation
            B, D = factorize(A, C, Cbar, L0)
            Q = B @ B.T
            R = D @ D.T
            S = B @ D.T
        if np.any(np.abs(np.linalg.eigvals(R)) < 1e-12):
            #print('R not > 0!')
            R += 1e-6 * np.eye(R.shape[0])

        # Partition C
        Cy = C[:y.shape[1], :]
        Cz = C[y.shape[1]:, :]

        return A, Cy, Cz, Cbar, L0, Q, R, S

    def form_hankel_toeplitz(self, ccm, T, dim_y):
        # Separate into neural data and behavior
        ccm_y = ccm[:, :dim_y, :dim_y]
        ccm_z = ccm[:, dim_y:, dim_y:]
        ccmyz = ccm[:, :dim_y, dim_y:]
        ccmzy = ccm[:, dim_y:, :dim_y]

        #### Y
        
        # T quantities
        # Tm_y, Tp_y = gen_toeplitz_from_blocks(ccm_y, order=T)
        
        # T + 1 quantities
        Tm1_y, Tp1_y = gen_toeplitz_from_blocks(ccm_y, order=T + 1)
        Lm1_y = np.linalg.cholesky(Tm1_y)
        # Lp1_y = np.linalg.cholesky(Tp1_y)

        ### Z
        # Tm_z, Tp_z = gen_toeplitz_from_blocks(ccm_z, order=T)
        
        # T + 1 quantities
        Tm1_z, Tp1_z = gen_toeplitz_from_blocks(ccm_z, order=T + 1)
        # Lm1_z = np.linalg.cholesky(Tm1_z)
        Lp1_z = np.linalg.cholesky(Tp1_z)

        # Project future of z onto y, therefore using yz block
        H1 = gen_hankel_from_blocks(ccmzy, order1=T + 1, order2=T + 1) 
        H1norm = np.linalg.inv(Lp1_z) @ H1 @ np.linalg.inv(Lm1_y).T
        Ut1, St1, Vht1 = np.linalg.svd(H1norm)

        return Ut1, St1, Vht1, Tm1_y, Lm1_y

    # Follow chapter 13 of LP except for in how we form the autocorrelation matrices
    def get_predictor_space(self, y, hankel_toeplitz, T, truncation_order):

        m = y.shape[1]
        

        Ut1, St1, Vht1, Tm1_y, Lm1_y = hankel_toeplitz

        St1 = np.diag(St1[0:truncation_order])
        Ut1 = Ut1[:, 0:truncation_order]
        Vht1 = Vht1[0:truncation_order, :]

        Sigmabart1 = Lm1_y @ Vht1.T @ scipy.linalg.sqrtm(St1) 
        Sigmabart = Sigmabart1[:-m, :]

        # Form the predictor spaces
        ypt1 = form_lag_matrix(y, T + 1).T
        ymt1 = flip_blocks(ypt1, T + 1)

        Xt = Sigmabart.T @ np.linalg.inv(Tm1_y) @ ymt1[m:, :]
        Xt1 = Sigmabart1.T @ np.linalg.inv(Tm1_y) @ ymt1

        # Reverse time estimate is not used

        return Xt, Xt1

# Second implementation that directly implements the method described in Nature Neuroscience paper
def brssid(y, z, order, T):
    ydim = y.shape[1]
    zdim = z.shape[1]

    yt = form_lag_matrix(y, 2*T)
    zt = form_lag_matrix(z, 2*T)

    # "Past" of y and "Future" of z
    ypast = yt[:, :T*ydim]
    zfut = zt[:, -T*zdim:]

    Z = scipy.linalg.lstsq(ypast, zfut)[0].T @ ypast.T

    svd = TruncatedSVD(n_components=order)
    svd.fit(Z.T)

    U = svd.components_.T
    Gamma_t = U @ np.diag(np.sqrt(svd.singular_values_))
    Xt = np.linalg.pinv(Gamma_t) @ Z

    linmodel = LinearRegression().fit(Xt.T, z[:-2*T + 1])
    r2_z = linmodel.score(Xt.T, z[:-9])

    linmodel = LinearRegression().fit(Xt.T, y[:-2*T + 1])
    Cy = linmodel.coef_
    return r2_z, Cy