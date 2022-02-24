import numpy as np
import scipy
import pdb
from copy import deepcopy as copy

from dca.cov_util import calc_cross_cov_mats_from_cov, form_lag_matrix, calc_chunked_cov,toeplitzify

# Ripped from dca.cov_util. Main difference is that we use the biased sample estimates (dividing by 1/n)
def sample_cross_cov_estimate(X, T, mean=None, chunks=None, stride=1,
                              rng=None, regularization=None, reg_ops=None,
                              stride_tricks=True, logger=None):
    """Compute the N-by-N cross-covariance matrix, where N is the data dimensionality,
    for each time lag up to T-1.

    Parameters
    ----------
    X : np.ndarray, shape (# time-steps, N)
        The N-dimensional time series data from which the cross-covariance
        matrices are computed.
    T : int
        The number of time lags.
    chunks : int
        Number of chunks to break the data into when calculating the lagged cross
        covariance. More chunks will mean less memory used
    stride : int or float
        If stride is an `int`, it defines the stride between lagged samples used
        to estimate the cross covariance matrix. Setting stride > 1 can speed up the
        calculation, but may lead to a loss in accuracy. Setting stride to a `float`
        greater than 0 and less than 1 will random subselect samples.
    rng : NumPy random state
        Only used if `stride` is a float.
    regularization : string
        Regularization method for computing the spatiotemporal covariance matrix.
    reg_ops : dict
        Paramters for regularization.
    stride_tricks : bool
        Whether to use numpy stride tricks in form_lag_matrix. True will use less
        memory for large T.

    Returns
    -------
    cross_cov_mats : np.ndarray, shape (T, N, N), float
        Cross-covariance matrices. cross_cov_mats[dt] is the cross-covariance between
        X(t) and X(t+dt), where X(t) is an N-dimensional vector.
    """
    if reg_ops is None:
        reg_ops = dict()
    if chunks is not None and regularization is not None:
        raise NotImplementedError

    if isinstance(X, list) or X.ndim == 3:
        for Xi in X:
            if len(Xi) <= T:
                raise ValueError('T must be shorter than the length of the shortest ' +
                                 'timeseries.')
        if mean is None:
            mean = np.concatenate(X).mean(axis=0, keepdims=True)
        X = [Xi - mean for Xi in X]
        N = X[0].shape[-1]
        if chunks is None:
            cov_est = np.zeros((N * T, N * T))
            n_samples = 0
            for Xi in X:
                X_with_lags = form_lag_matrix(Xi, T, stride=stride, stride_tricks=stride_tricks,
                                              rng=rng)
                cov_est += np.dot(X_with_lags.T, X_with_lags)
                n_samples += len(X_with_lags)
            cov_est /= float(n_samples)
        else:
            n_samples = 0
            cov_est = np.zeros((N * T, N * T))
            for Xi in X:
                cov_est, ni_samples = calc_chunked_cov(Xi, T, stride, chunks, cov_est=cov_est,
                                                       stride_tricks=stride_tricks, rng=rng)
                n_samples += ni_samples
            cov_est /= float(n_samples)
    else:
        if len(X) <= T:
            raise ValueError('T must be shorter than the length of the shortest ' +
                             'timeseries. If you are using the DCA model, 2 * DCA.T must be ' +
                             'shorter than the shortest timeseries.')
        if mean is None:
            mean = X.mean(axis=0, keepdims=True)
        X = X - mean
        N = X.shape[-1]
        if chunks is None:
            X_with_lags = form_lag_matrix(X, T, stride=stride, stride_tricks=stride_tricks,
                                          rng=rng)
            n_samples = X_with_lags.shape[0]
            # Need to identify n_samples here
            cov_est = np.cov(X_with_lags, rowvar=False, bias=True)
        else:
            cov_est, n_samples = calc_chunked_cov(X, T, stride, chunks,
                                                  stride_tricks=stride_tricks, rng=rng)
            cov_est /= float(n_samples)

    cov_est = toeplitzify(cov_est, T, N)

    # Handle PSD requirement separately
    #    rectify_spectrum(cov_est, logger=logger)
    cross_cov_mats = calc_cross_cov_mats_from_cov(cov_est, T, N)
    return cross_cov_mats, n_samples

def taper_covariance(ccm, q):

    if np.isscalar(q):
        q = q * np.ones(ccm.shape[1:])

    # Use trapezoidal taper function
    def taper(x):
        if np.abs(x) < 1:
            return 1
        elif 1 < np.abs(x) < 2:
            return 2 - np.abs(x)
        else:
            return 0

    for h in range(ccm.shape[0]):
        for i in range(ccm.shape[1]):
            for j in range(ccm.shape[2]):
                if q[i, j] > 0:
                    ccm[h, i, j] *= taper(h/q[i, j])
    return ccm

"""Follow the regularization method outlined in 
COVARIANCE MATRIX ESTIMATION AND LINEAR PROCESS
BOOTSTRAP FOR MULTIVARIATE TIME SERIES
OF POSSIBLY INCREASING DIMENSION
"""
def estimate_autocorrelation(X, T, M=2, tapering='local'):

    # First get the sample autocorrelation
    ccm_sample, n_samples = sample_cross_cov_estimate(X, T)

    # Select tapering bandwidth

    # Normalize cross covariance to be a correlation
    ccm_sample_normalized = copy(ccm_sample)
    for k in range(ccm_sample_normalized.shape[0]):
        for i in range(ccm_sample_normalized.shape[1]):
            for j in range(ccm_sample_normalized.shape[2]):
                ccm_sample_normalized[k, i, j] /= np.sqrt(ccm_sample[0, i, i] * ccm_sample[0, j, j]) 

    # Identify integers q_{kl}. These hyperparameters are set to the recommended default values
    M = M * np.sqrt(np.log10(n_samples)/n_samples)
    K = max(5, np.sqrt(np.log10(n_samples)))

    # Basic question, is K basically going to be always greater than 5?
    qij = np.zeros((ccm_sample_normalized.shape[1:]))
    for h in range(K):
        # T should be fairly large in this case
        for q in range(K - T):
            for i in range(ccm_sample_normalized.shape[1]):
                for j in range(ccm_sample_normalized.shape[2]):
                    if np.abs(ccm_sample_normalized(h + q, i, j)) < M:
                        if qij[i, j] == 0: 
                            qij[i, j] = q
    
    # Taper covariance    
    if tapering == 'local':
        ccm_tapered = taper_covariance(ccm_sample, qij)
    elif tapering == 'global':
        ccm_tapered = taper_covariance(ccm_sample, np.max(qij))
    else:
        raise ValueError

    return ccm_tapered
