import numpy as np
import scipy
from scipy import signal
import pdb

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pykalman import KalmanFilter

#from pyuoi.linear_model.var import VAR
from dca.dca import DynamicalComponentsAnalysis as DCA
from dca.cov_util import (calc_cross_cov_mats_from_data, 
                          calc_pi_from_cross_cov_mats, form_lag_matrix)

def decimate_(X, q):

    Xdecimated = []
    for i in range(X.shape[1]):
        Xdecimated.append(signal.decimate(X[:, i], q))

    return np.array(Xdecimated).T

# If X has trial structure, need to seperately normalize each trial
def standardize(X):

    scaler = StandardScaler()

    if type(X) == list:
        Xstd = [scaler.fit_transform(x) for x in X] 
    elif np.ndim(X) == 3:
        Xstd = np.array([scaler.fit_transform(X[idx, ...]) 
                         for idx in range(X.shape[0])])
    else:
        Xstd = scaler.fit_transform(X)

    return Xstd

# Turn position into velocity and acceleration with finite differences
def expand_state_space(Z, X, include_vel=True, include_acc=True):

    concat_state_space = []
    for i, z in enumerate(Z):
        if include_vel and include_acc:
            pos = z[2:, :]
            vel = np.diff(z, 1, axis=0)[1:, :]
            acc = np.diff(z, 2, axis=0)

            # Trim off 2 samples from the neural data to match lengths
            X[i] = X[i][2:, :]

            concat_state_space.append(np.concatenate((pos, vel, acc), axis=-1))
        elif include_vel:
            pos = z[1:, :]
            vel = np.diff(z, 1, axis=0)
            # Trim off only one sample in this case
            X[i] = X[i][1:, :]
            concat_state_space.append(np.concatenate((pos, vel), axis=-1))
        else:
            concat_state_space.append(z)

    return concat_state_space, X
    
def KF(X, Z):

    # Assemble kinematic state variable (6D)
    # Chop off the first 2 points for equal length vectors
    pos = Z[2:, :]
    vel = np.diff(Z, 1, axis=0, )[1:, :]
    acc = np.diff(Z, 2, axis=0, )

    z = np.hstack([pos, vel, acc])

    # Trim neural data accordingly
    x = X[2:, :]

    # Kinematic mean and variance (same-time)
    mu0 = np.mean(z, axis=0)
    Sigma0 = np.cov(z.T)

    # Kinematic state transition matrices
    linregressor = LinearRegression(normalize=True, fit_intercept=True)
    varmodel = VAR(estimator='ols', fit_intercept=True, order=1, 
                   self_regress=True)
    varmodel.fit(z)

    A = np.squeeze(varmodel.coef_)
    az = varmodel.intercept_

    # Get the residual covariance
    zpred, z_ = varmodel.predict(z)

    epsilon = z_ - zpred
    Sigmaz = np.cov(epsilon.T)

    # Predict the neural data from the kinematic data
    try:
        linregressor.fit(z, x)
    except:
        pdb.set_trace()
    Cxz = linregressor.coef_
    cxz = linregressor.intercept_

    # Can try to do poisson regression here

    yypred = linregressor.predict(z)
    epsilon = x - yypred
    Sigmaxz = np.cov(epsilon.T)

    # Instantiate a Kalman filter using these parameters
    kf = KalmanFilter(transition_matrices = A, observation_matrices = Cxz,
                      transition_covariance = Sigmaz, observation_covariance = Sigmaxz,
                      transition_offsets = az, observation_offsets = cxz,
                      initial_state_mean = mu0, initial_state_covariance=Sigma0)

    return kf

def kf_decoder(Xtest, Xtrain, Ztest, Ztrain, trainlag, testlag, decoding_window=1):

    Ztrain = scaler.fit_transform(Ztrain)
    Xtrain = scaler.fit_transform(Xtrain)

    Xtest = scaler.fit_transform(Xtest)
    Ztest = scaler.fit_transform(Ztest)

    # Apply train lag
    if trainlag > 0:
        Xtrain = Xtrain[:-trainlag, :]
        Ztrain = Ztrain[trainlag:, :]

    if testlag > 0:
        # Apply test lag
        Xtest = Xtest[:-testlag, :]
        Ztest = Ztest[testlag:, :]

    kf = KF(Xtrain, Ztrain)

    state_estimates, _ = kf.filter(Xtest)

    pos_estimates = state_estimates[:, 0:2]
    vel_estimates = state_estimates[:, 2:4]
    acc_estimates = state_estimates[:, 2:, ]

    pos_true = Ztest[2:, :]
    vel_true = np.diff(Ztest, 1, axis=0, )[1:, :]
    acc_true = np.diff(Ztest, 2, axis=0, )

    kf_r2_pos = r2_score(pos_true, pos_estimates)
    kf_r2_vel = r2_score(vel_true, vel_estimates)
    kf_r2_acc = r2_score(acc_true, acc_estimates)

    return kf_r2_pos, kf_r2_vel, kf_r2_acc, kf

def lr_preprocess(Xtest, Xtrain, Ztest, Ztrain, trainlag, testlag, decoding_window, include_velocity, include_acc):

    # If no trial structure is present, convert to a list for easy coding
    if np.ndim(Xtrain) == 2:
        Xtrain = [Xtrain]
        Xtest = [Xtest]

        Ztrain = [Ztrain]
        Ztest = [Ztest]

    Ztrain = standardize(Ztrain)
    Xtrain = standardize(Xtrain)

    Ztest = standardize(Ztest)
    Xtest = standardize(Xtest)

    # Apply train lag
    if trainlag > 0:
        Xtrain = [x[:-trainlag, :] for x in Xtrain]
        Ztrain = [z[trainlag:, :] for z in Ztrain]
    elif trainlag < 0:
        Xtrain = [x[-trainlag:, :] for x in Xtrain]
        Ztrain = [z[:trainlag, :] for z in Ztrain]


    # Apply test lag
    if testlag > 0:
        Xtest = [x[:-trainlag, :] for x in Xtest]
        Ztest = [z[trainlag:, :] for z in Ztest]
    elif testlag < 0:
        Xtest = [x[-trainlag:, :] for x in Xtest]
        Ztest = [z[:trainlag, :] for z in Ztest]

    # Apply decoding window
    Xtrain = [form_lag_matrix(x, decoding_window) for x in Xtrain]
    Xtest = [form_lag_matrix(x, decoding_window) for x in Xtest]

    Ztrain = [z[decoding_window//2:, :] for z in Ztrain]
    Ztrain = [z[:x.shape[0], :] for z, x in zip(Ztrain, Xtrain)]

    Ztest = [z[decoding_window//2:, :] for z in Ztest]
    Ztest = [z[:x.shape[0], :] for z, x in zip(Ztest, Xtest)]

    # Expand state space to include velocity and acceleration
    if np.any([include_velocity, include_acc]):
        Ztrain, Xtrain = expand_state_space(Ztrain, Xtrain, include_velocity, include_acc)
        Ztest, Xtest = expand_state_space(Ztest, Xtest, include_velocity, include_acc)

    # Flatten trial structure as regression will not care about it
    Xtrain = np.concatenate(Xtrain)
    Xtest = np.concatenate(Xtest)
    Ztrain = np.concatenate(Ztrain)
    Ztest = np.concatenate(Ztest)

    return Xtest, Xtrain, Ztest, Ztrain

# Sticking with consistent nomenclature, Z is the behavioral data and X is the neural data
def lr_encoder(Xtest, Xtrain, Ztest, Ztrain, trainlag, testlag, decoding_window=1, include_velocity=True, include_acc=False):

    # By default, we look only at pos and vel
    Xtest, Xtrain, Ztest, Ztrain = lr_preprocess(Xtest, Xtrain, Ztest, Ztrain, trainlag, testlag, decoding_window, include_velocity, include_acc)

    # Apply the decoding window to the behavioral data
    # Ztrain, _ = form_lag_matrix(Ztrain, decoding_window)
    # Ztest, _ = form_lag_matrix(Ztest, decoding_window)

    # Xtrain = Xtrain[decoding_window//2:, :]
    # Xtest = Xtest[:Ztest.shape[1], :]

    encodingregressor = LinearRegression(fit_intercept=True)

    # Throw away acceleration
    # Ztest = Ztest[:, 0:4]
    # Ztrain = Ztrain[:, 0:4]

    encodingregressor.fit(Ztrain, Xtrain)
    Xpred = encodingregressor.predict(Ztest)

    r2 = r2_score(Xtest, Xpred)
    return r2, encodingregressor

def lr_decoder(Xtest, Xtrain, Ztest, Ztrain, trainlag, testlag, decoding_window=1, include_velocity=True, include_acc=True):

    behavior_dim = Ztrain[0].shape[-1]

    Xtest, Xtrain, Ztest, Ztrain = lr_preprocess(Xtest, Xtrain, Ztest, Ztrain, trainlag, testlag, decoding_window, include_velocity, include_acc)
    decodingregressor = LinearRegression(fit_intercept=True)
    decodingregressor.fit(Xtrain, Ztrain)
    Zpred = decodingregressor.predict(Xtest)

    if include_velocity and include_acc:
        lr_r2_pos = r2_score(Ztest[..., 0:behavior_dim], Zpred[..., 0:behavior_dim])
        lr_r2_vel = r2_score(Ztest[..., behavior_dim:2*behavior_dim], Zpred[..., behavior_dim:2*behavior_dim])
        lr_r2_acc = r2_score(Ztest[..., 2*behavior_dim:], Zpred[..., 2*behavior_dim:])

        return lr_r2_pos, lr_r2_vel, lr_r2_acc, decodingregressor
    elif include_velocity:
        lr_r2_pos = r2_score(Ztest[..., 0:behavior_dim], Zpred[..., 0:behavior_dim])
        lr_r2_vel = r2_score(Ztest[..., behavior_dim:2*behavior_dim], Zpred[..., behavior_dim:2*behavior_dim])
        return lr_r2_pos, lr_r2_vel, decodingregressor
    else:
        lr_r2_pos = r2_score(Ztest, Zpred)
        return lr_r2_pos, decodingregressor

def _draw_bootstrap_sample(rng, X, y):
    sample_indices = np.arange(X.shape[0])
    bootstrap_indices = rng.choice(
        sample_indices, size=sample_indices.shape[0], replace=True
    )
    return X[bootstrap_indices], y[bootstrap_indices]

def lr_bias_variance(Xtest, Xtrain, Ztest, Ztrain, trainlag, testlag, decoding_window=1, n_boots=200, random_seed=None):

    if random_seed is None:
        rand = np.random
    else:
        rand = np.random.RandomState(random_seed)

    # To bootstrap, we need to preprocess and flatten the data
    Xtest, Xtrain, Ztest, Ztrain = lr_preprocess(Xtest, Xtrain, Ztest, Ztrain, trainlag, testlag, decoding_window, include_velocity=True, include_acc=True)
    # Run lr_decoder over bootstrapped samples of xtrain and xtest. Use this to calculate bias and variance of the estimator
    zpred_boot = []
    for k in range(n_boots):

        xboot, zboot = _draw_bootstrap_sample(rand, Xtrain, Ztrain)
        decodingregressor = LinearRegression(fit_intercept=True)
        decodingregressor.fit(xboot, zboot)
        zpred = decodingregressor.predict(Xtest)
        zpred_boot.append(zpred)

    zpred_boot = np.array(zpred_boot)

    assert(np.allclose((zpred_boot - Ztest).shape, zpred_boot.shape))

    # Bias/Variance/MSE
    mse = np.mean(np.mean(np.power(zpred_boot - Ztest, 2), axis=1), axis=0)
    Ezpred = np.mean(zpred_boot, axis=0)
    bias = np.sum((Ezpred - Ztest)**2, axis=0)/Ztest.shape[0]
    var = np.mean(np.mean(np.power(zpred_boot - Ezpred, 2), axis=1), axis=0)
    return mse, bias, var

def lr_bv_windowed(X, Z, lag, window, transition_times, train_idxs, test_idxs, pkassign=None, apply_pk_to_train=False, 
                   decoding_window=1, n_boots=200, random_seed=None, offsets=None):

    if random_seed is None:
        rand = np.random
    else:
        rand = np.random.RandomState(random_seed)

    # We have been given a list of windows for each transition
    if len(window) > 2:
        W = [w for win in window for w in win]
        win_min = min(W)
    else:
        win_min = window[0]

    if win_min >= 0:
        win_min = 0

    # Filter out by transitions that lie within the train idxs, and stay clear of the start and end
    tt_train = [(t, idx) for idx, t in enumerate(transition_times) 
                if idx in train_idxs and t[0] > (lag + np.abs(win_min)) and t[1] < (Z.shape[0] - lag - np.abs(win_min))]
    # Re-assign train idxs removing those reaches that were outside the start/end region
    train_idxs = [x[1] for x in tt_train]
    tt_train = [x[0] for x in tt_train]

    if offsets is not None:
        offsets_train = offsets[train_idxs]
    else:
        offsets_train = None

    # Get trialized, windowed data
    if pkassign is not None and apply_pk_to_train:
        assert(np.all([s.size == np.arange(t[0], t[1]).size for (s, t) in zip(pkassign[train_idxs], tt_train)]))
        subset_selection = [np.argwhere(np.array(s) == 0).squeeze() for s in pkassign[train_idxs]]

        Xtrain, Ztrain = apply_window(X, Z, lag, window, tt_train, decoding_window, True, True, False, subset_selection, offsets=offsets_train)
    else:
        Xtrain, Ztrain = apply_window(X, Z, lag, window, tt_train, decoding_window, True, True, False, offsets=offsets_train)

    # Filter out by transitions that lie within the test idxs, and stay clear of the start and end
    tt_test = [(t, idx) for idx, t in enumerate(transition_times) 
                if idx in test_idxs and t[0] > (lag + np.abs(win_min)) and t[1] < (Z.shape[0] - lag - np.abs(win_min))]
    # Re-assign test idxs removing those reaches that were outside the start/end region
    test_idxs = [x[1] for x in tt_test]
    tt_test = [x[0] for x in tt_test]
    if offsets is not None:
        offsets_test = offsets[test_idxs]
    else:
        offsets_test = None

    if pkassign is not None:
        assert(np.all([s.size == np.arange(t[0], t[1]).size for (s, t) in zip(pkassign[test_idxs], tt_test)]))
        subset_selection = [np.argwhere(np.array(s) != 0).squeeze() for s in pkassign[test_idxs]]
        Xtest, Ztest = apply_window(X, Z, lag, window, tt_test, decoding_window, True, True, False, subset_selection, offsets=offsets_test)
    else:
        Xtest, Ztest = apply_window(X, Z, lag, window, tt_test, decoding_window, True, True, False, offsets=offsets_test)

    num_test_reaches = len(Xtest)

    # verify dimensionalities
    if len(Xtrain) > 0:
        Xtrain = np.concatenate(Xtrain)
        Ztrain = np.concatenate(Ztrain)
    else:
        return np.nan, np.nan, np.nan, num_test_reaches

    if len(Xtest) > 0:
        Xtest = np.concatenate(Xtest)
        Ztest = np.concatenate(Ztest)

        Xtrain = StandardScaler().fit_transform(Xtrain)
        Ztrain = StandardScaler().fit_transform(Ztrain)
        Xtest = StandardScaler().fit_transform(Xtest)
        Ztest = StandardScaler().fit_transform(Ztest)

        # Run lr_decoder over bootstrapped samples of xtrain and xtest. Use this to calculate bias and variance of the estimator
        zpred_boot = []
        for k in range(n_boots):

            xboot, zboot = _draw_bootstrap_sample(rand, Xtrain, Ztrain)
            decodingregressor = LinearRegression(fit_intercept=True)
            decodingregressor.fit(xboot, zboot)
            zpred = decodingregressor.predict(Xtest)
            zpred_boot.append(zpred)

        zpred_boot = np.array(zpred_boot)

        assert(np.allclose((zpred_boot - Ztest).shape, zpred_boot.shape))

        # Bias/Variance/MSE
        mse = np.mean(np.mean(np.power(zpred_boot - Ztest, 2), axis=1), axis=0)
        
        Ezpred = np.mean(zpred_boot, axis=0)
        bias = np.sum((Ezpred - Ztest)**2, axis=0)/Ztest.shape[0]
        var = np.mean(np.mean(np.power(zpred_boot - Ezpred, 2), axis=1), axis=0)
        return mse, bias, var, num_test_reaches
    else:
        return np.nan, np.nan, np.nan, num_test_reaches


def apply_window(X, Z, lag, window, transition_times, decoding_window, include_velocity, include_acc, measure_from_end, subset_selection=None, offsets=None):

    # subset_selection: set of indices of the same length as transition_times that indicate whether a subset of the transition
    # is to be included. This is used when we enforce peak membership in decoding.

    # Apply decoding window
    X = form_lag_matrix(X, decoding_window)
    Z = Z[decoding_window//2:, :]
    Z = Z[:X.shape[0], :]

    # This *also* requires shifting the transition times, as behavior will have been affected
    if decoding_window > 1:
        transition_times = [(t[0] - decoding_window//2, t[1] - decoding_window//2) for t in transition_times]        

    assert(X.shape[0] == Z.shape[0])

    # Expand state space to include velocity and acceleration
    if np.any([include_velocity, include_acc]):
        Z, X = expand_state_space([Z], [X], include_velocity, include_acc)

    # Flatten list structure imposed by expand_state_space
    Z = Z[0]
    X= X[0]

    # Segment the time series with respect to the transition times (including lag)
    xx = []
    zz = []

    valid_idxs = []

    def valid_reach(t0, t1, w, measure_from_end):
        if measure_from_end:
            window_in_reach = t1 - w[1] > t0
        else:
            window_in_reach = t0 + w[1] < t1
        return window_in_reach

    # If given a single window, duplicate it across all transition times
    if len(window) == 2:
        window = [window for _ in range(len(transition_times))]

    # If no offsets provided, let it be 0 for all transition times
    if offsets is None:
        offsets = np.zeros(len(transition_times))

    print('Initial reach count: %d' % len(transition_times))
    for i, (t0, t1) in enumerate(transition_times):
        # Enforce that the previous reach must not have began after the window begins
        # UPDATE: no longer used, see the code block below
        # try:
        #     assert(valid_reach(t0,  t1, window[i], measure_from_end))
        # except:
        #     pdb.set_trace()
        if measure_from_end:
            if subset_selection is not None:
                raise ValueError('Not supported for measuring from end')
            if offsets[i] != 0:
                raise ValueError('Not supported for measuring from end')
            xx_ = X[t1 - lag - window[i][1]:t1 - lag - window[i][0]]
            zz_ = Z[t1 - window[i][1]:t1 - window[i][0]]
        else:
            window_indices = np.arange(t0 + window[i][0] + offsets[i], t0 + window[i][1] + offsets[i])
            if subset_selection is not None:
                # Select only indices that do not belong to the first velocity peak
                subset_indices = np.arange(t0, t1)[subset_selection[i]]
                window_indices = np.intersect1d(window_indices, subset_indices)

            # No matter what, we should remove segments that overlap with the next transition
            if i < len(transition_times) - 1:
                window_indices = window_indices[window_indices < transition_times[i + 1][0]]
            else:
                # Or else make sure that we don't exceed the length of the time series
                window_indices = window_indices[window_indices < Z.shape[0]]

            window_indices = window_indices.astype(int)
            zz_ = Z[window_indices]

            # Shift x indices by lag
            window_indices -= lag
            xx_ = X[window_indices]

        if len(xx_) > 0:
            assert(xx_.shape[0] == zz_.shape[0])
            xx.append(xx_)
            zz.append(zz_)

    print('Final reach count: %d' % len(zz))

    return xx, zz

def lr_decode_windowed(X, Z, lag, window, transition_times, train_idxs, test_idxs=None, 
                       decoding_window=1, include_velocity=True, include_acc=True, measure_from_end=False, pkassign=None, apply_pk_to_train=False, offsets=None):

    behavior_dim = Z.shape[-1]

    # We have been given a list of windows for each transition
    if len(window) > 2:
        W = [w for win in window for w in win]
        win_min = min(W)
    else:
        win_min = window[0]

    if win_min >= 0:
        win_min = 0

    # Filter out by transitions that lie within the train idxs, and stay clear of the start and end
    tt_train = [(t, idx) for idx, t in enumerate(transition_times) 
                if idx in train_idxs and t[0] > (lag + np.abs(win_min)) and t[1] < (Z.shape[0] - lag - np.abs(win_min))]
    # Re-assign train idxs removing those reaches that were outside the start/end region
    train_idxs = [x[1] for x in tt_train]
    tt_train = [x[0] for x in tt_train]

    if offsets is not None:
        offsets_train = offsets[train_idxs]
    else:
        offsets_train = None

    if apply_pk_to_train:
        # Train on the first velocity peak only
        assert(np.all([s.size == np.arange(t[0], t[1]).size for (s, t) in zip(pkassign[train_idxs], tt_train)]))
        subset_selection = [np.argwhere(np.array(s) == 0).squeeze() for s in pkassign[train_idxs]]
        Xtrain, Ztrain = apply_window(X, Z, lag, window, tt_train, decoding_window, include_velocity, include_acc, measure_from_end, subset_selection, offsets=offsets_train)
    else:
        Xtrain, Ztrain = apply_window(X, Z, lag, window, tt_train, decoding_window, include_velocity, include_acc, measure_from_end, offsets=offsets_train)

    if Xtrain is None:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, None, 0
    else:
        n = len(Xtrain)

    if test_idxs is not None:
        # Filter out by transitions that lie within the test idxs, and stay clear of the start and end
        tt_test = [(t, idx) for idx, t in enumerate(transition_times) 
                   if idx in test_idxs and t[0] > (lag + np.abs(win_min)) and t[1] < (Z.shape[0] - lag - np.abs(win_min))]
        # Re-assign test idxs removing those reaches that were outside the start/end region
        test_idxs = [x[1] for x in tt_test]
        tt_test = [x[0] for x in tt_test]


        if offsets is not None:
            offsets_test = offsets[test_idxs]
        else:
            offsets_test = None

        if pkassign is not None:
            assert(np.all([s.size == np.arange(t[0], t[1]).size for (s, t) in zip(pkassign[test_idxs], tt_test)]))
            subset_selection = [np.argwhere(np.array(s) != 0).squeeze() for s in pkassign[test_idxs]]
            Xtest, Ztest = apply_window(X, Z, lag, window, tt_test, decoding_window, include_velocity, include_acc, measure_from_end, subset_selection, offsets=offsets_test)        
        else:
            Xtest, Ztest = apply_window(X, Z, lag, window, tt_test, decoding_window, include_velocity, include_acc, measure_from_end, offsets=offsets_test)

    else:
        Xtest = None
        Ztest = None

    # Standardize
    # X = StandardScaler().fit_transform(X)
    # Z = StandardScaler().fit_transform(Z)
    decodingregressor = LinearRegression(fit_intercept=True)

    # Fit and score
    if len(Xtrain) == 0:
        return tuple([np.nan] * 9) + (0,)
    decodingregressor.fit(np.concatenate(Xtrain), np.concatenate(Ztrain))
    Zpred = decodingregressor.predict(np.concatenate(Xtrain))

    # Re-segment Zpred
    idx = 0
    Zpred_segmented = []
    for i, z in enumerate(Ztrain):
        Zpred_segmented.append(Zpred[idx:idx+z.shape[0]])
        idx += z.shape[0]

    assert(np.all([z1.shape[0] == z2.shape[0] for (z1, z2) in zip(Zpred_segmented, Ztrain)]))
    Ztrain = np.concatenate(Ztrain)

    if Xtest is not None:
        if len(Xtest) > 0:
            num_test_reaches = len(Xtest)
            Zpred_test = decodingregressor.predict(np.concatenate(Xtest))
        
            idx = 0
            Zpred_test_segmented = []
            for i, z in enumerate(Ztest):
                Zpred_test_segmented.append(Zpred_test[idx:idx+z.shape[0]])
                idx += z.shape[0]

            assert(np.all([z1.shape[0] == z2.shape[0] for (z1, z2) in zip(Zpred_test_segmented, Ztest)]))

            Ztest = np.concatenate(Ztest)
        else:
            Xtest = None
            Ztest = None
            num_test_reaches = 0

    if include_velocity and include_acc:

        # Additionally calculate the individual MSE
        mse_train = [np.linalg.norm(z1 - z2, axis=0)/z1.shape[0] for (z1, z2) in zip(Zpred_segmented, Ztrain)]

        lr_r2_pos = r2_score(Ztrain[..., 0:behavior_dim], Zpred[..., 0:behavior_dim])
        lr_r2_vel = r2_score(Ztrain[..., behavior_dim:2*behavior_dim], Zpred[..., behavior_dim:2*behavior_dim])
        lr_r2_acc = r2_score(Ztrain[..., 2*behavior_dim:], Zpred[..., 2*behavior_dim:])

        if Xtest is not None:
            mse_test = [np.linalg.norm(z1 - z2, axis=0)/z1.shape[0] for (z1, z2) in zip(Zpred_test_segmented, Ztest)]
            lr_r2_post = r2_score(Ztest[..., 0:behavior_dim], Zpred_test[..., 0:behavior_dim])
            lr_r2_velt = r2_score(Ztest[..., behavior_dim:2*behavior_dim], Zpred_test[..., behavior_dim:2*behavior_dim])
            lr_r2_acct = r2_score(Ztest[..., 2*behavior_dim:], Zpred_test[..., 2*behavior_dim:])
        else:
            mse_test = np.nan
            lr_r2_post = np.nan
            lr_r2_velt = np.nan
            lr_r2_acct = np.nan

        return lr_r2_pos, lr_r2_vel, lr_r2_acc, lr_r2_post, lr_r2_velt, lr_r2_acct, mse_train, mse_test, decodingregressor, num_test_reaches

    elif include_velocity:
        raise NotImplementedError
    else:
        raise NotImplementedError