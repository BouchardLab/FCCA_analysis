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

from pyuoi.linear_model.var import VAR
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
            print(z.shape)
            pos = z[2:, :]
            vel = np.diff(z, 1, axis=0)[1:, :]
            acc = np.diff(z, 2, axis=0)

            # Trim off 2 samples from the neural data to match lengths
            X[i] = X[i][2:, :]

            concat_state_space.append(np.concatenate((pos, vel, acc), axis=-1))
        elif inlclude_vel:
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

    # Apply test lag
    if testlag > 0:
        Xtest = [x[:-trainlag, :] for x in Xtest]
        Ztest = [z[trainlag:, :] for z in Ztest]


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

    encodingregressor = LinearRegression(normalize=True, fit_intercept=True)

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
    decodingregressor = LinearRegression(normalize=True, fit_intercept=True)

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