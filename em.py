from re import I
import numpy as np
import scipy
from scipy.optimize import minimize
import pdb
from pyuoi.linear_model.var import VAR
from subspaces import SubspaceIdentification, IteratedStableEstimator

# Experiment with either pykalman or pylds implementations
from pykalman.standard import (KalmanFilter, _filter, _smooth, _smooth_pair, _em_observation_matrix, 
                               _em_observation_covariance, _em_initial_state_mean, _em_initial_state_covariance,
                               _loglikelihoods)

def _em_stable_transition_matrix(Ainit, Pt, Pt1, Ptt1, lambda_A, T):

    n = Ainit.shape[0]

    # Portion of the cost function that depends on A. Note that this is the *negative* of the log likelihood so 
    # we can minimize
    Q = lambda A: np.eye(A.shape[0]) - A @ A.T

    def f(A):        
        A = np.reshape(A, (n, n))
        return 0.5 * np.linalg.slogdet(Q(A))[1] + \
                lambda_A/(2*(T - 1)) * np.trace((A - np.eye(A.shape[0])) @ (A - np.eye(A.shape[0])).T) + \
                0.5 * np.trace(np.linalg.inv(Q(A)) @ (A @ Pt1 @ A.T - A @ Ptt1.T - Ptt1 @ A.T + Pt))

    def df(A):
        A = np.reshape(A, (n, n))
        dfdA =  np.linalg.inv(Q(A)) @ (-A + (A @ Pt1 - Ptt1) \
               + (A @ Pt @ A.T - A @ Ptt1.T - Ptt1 @ A.T + Pt1) @ np.linalg.inv(Q(A)) @ A)
        return dfdA.ravel()

    # Gradient with respect to A
    opt = minimize(f, Ainit.ravel(), method='Newton-CG', jac=df)
    
    return opt.x.reshape(n, n)

class StableStateSpaceML():

    def __init__(self, init_strategy='SSID', tol=1e-2, max_iter=100, 
                 lambda_A=1, rand_state=None, optimize_init_cond=True, **init_kwargs):

        # We do not currently allow for there to be correlation between Q and R

        self.tol = tol
        self.max_iter = max_iter
        self.lambda_A = lambda_A
        self.optimize_init_cond = optimize_init_cond

        if init_strategy not in ['PCA', 'SSID', 'manual']:
            raise ValueError('Unknown initialization strategy specified.')

        self.init_strategy = init_strategy

    def fit(self, y, state_dim, **init_kwargs):

        if self.init_strategy == 'manual':
            Ainit = init_kwargs['Ainit']
            Cinit = init_kwargs['Cinit']
            Qinit = init_kwargs['Qinit']
            Rinit = init_kwargs['Rinit']
            x0 = init_kwargs['x0']
            Sigma0 = init_kwargs['Sigma0']
        elif self.init_strategy == 'AR':
            # Fit an autoregressive model to y and then take supplement the dynamics matrix
            varmodel = VAR(order = state_dim//y.shape[1], estimator='OLS')
            varmodel.fit(y)
            
            Ainit = form_companion(varmodel.coef_)
            # Supplement
            Ainit = scipy.linalg.block_diag(Ainit, 0.5*np.eye(state_dim - Ainit.shape[0]))
            Qinit = np.eye(Ainit.shape[0])
            Rinit = 1e-3 * np.eye(y.shape[1])
            Cinit = np.block([[np.eye(y.shape[1])], [np.zeros((Ainit.shape[0] - y.shape[1], y.shape[1]))]])
            Sigma0 = scipy.linalg.solve_discrete_lyapunov(Ainit, Qinit)
            x0 = np.zeros(state_dim)

        elif self.init_strategy == 'SSID':
            # Fit stable subpsace identification
            ssid = SubspaceIdentification(estimator=IteratedStableEstimator, obs_regressor='OLS', **init_kwargs)
            A, C, Cbar, L0, Q, R, S = ssid.identify(y, order=state_dim, T=3)
            Ainit = A
            Cinit = C
            Qinit = Q
            Rinit = R
            
            # Initial state mean set to PCA initialization, covariance set to solution of Lyapunov equation
            x0 = np.zeros(state_dim)
            Sigma0 = scipy.linalg.solve_discrete_lyapunov(A, Q)
        

        filter_params = {
            'transition_matrices':Ainit,
            'observation_matrices':Cinit,
            'transition_covariance':Qinit,
            'observation_covariance':Rinit,
            'initial_state_mean':x0,
            'initial_state_covariance':Sigma0,
            'observation_offsets':np.zeros(Cinit.shape[0]),
            'transition_offsets':np.zeros(Ainit.shape[0])
        }

        for key, val in filter_params.items():
            setattr(self, key, val)     

        iter = 0
        tol = np.inf

        while iter < self.max_iter and tol > self.tol:
            Exhat, Ptt, Ptt1, logll = self.E(y)
            self.M(y, Exhat, Ptt, Ptt1)
            iter += 1
            print('Iteration %d, Log Likelihood: %f' % (iter, logll))

    def E(self, y):
        # Ripped out of pykalman KalmanFilter.em
        (predicted_state_means, predicted_state_covariances,
            kalman_gains, filtered_state_means,
            filtered_state_covariances) = (
            _filter(
                self.transition_matrices, self.observation_matrices,
                self.transition_covariance, self.observation_covariance,
                self.transition_offsets, self.observation_offsets,
                self.initial_state_mean, self.initial_state_covariance,
                y
            )
        )
        (smoothed_state_means, smoothed_state_covariances,
            kalman_smoothing_gains) = (
            _smooth(
                self.transition_matrices, filtered_state_means,
                filtered_state_covariances, predicted_state_means,
                predicted_state_covariances
            )
        )
        sigma_pair_smooth = _smooth_pair(
            smoothed_state_covariances,
            kalman_smoothing_gains
        )

        loglikelihoods = _loglikelihoods(
                self.observation_matrices, self.observation_offsets, self.observation_covariance,
                predicted_state_means, predicted_state_covariances, y
            )

        logll = np.sum(loglikelihoods)

        return smoothed_state_means, smoothed_state_covariances, sigma_pair_smooth, logll      

    def M(self, y, smoothed_state_means, smoothed_state_covariances, pairwise_covariances):
        
        # Copy all M steps from pykalman implementation except for A and C

        # Offset and sum
        Pt = 1/(y.shape[0] - 1) * np.sum(smoothed_state_covariances[1:], axis=0)
        Pt1 = 1/(y.shape[0] - 1) * np.sum(smoothed_state_covariances[:-1], axis=0)
        Ptt1 =  1/(y.shape[0] - 1) * np.sum(pairwise_covariances, axis=0)

        transition_matrices = _em_stable_transition_matrix(self.transition_matrices, 
                                                         Pt, Pt1, Ptt1, self.lambda_A, y.shape[0])        

        observation_matrices = _em_observation_matrix(
            y, self.observation_offsets,
            smoothed_state_means, smoothed_state_covariances
        )

        observation_covariance = _em_observation_covariance(
            y, self.observation_offsets,
            self.observation_matrices, smoothed_state_means,
            smoothed_state_covariances
        )   

        # transition_covariance = _em_transition_covariance(
        #     self.transition_matrix, transition_offsets,
        #     smoothed_state_means, smoothed_state_covariances,
        #     pairwise_covariances
        # )
        # transition covariance is constrained to be 1 - A A^\top


        if self.optimize_init_cond:
            initial_state_mean = _em_initial_state_mean(smoothed_state_means)
            initial_state_covariance = _em_initial_state_covariance(
                                       initial_state_mean, smoothed_state_means,
                                       smoothed_state_covariances)
        else:
            initial_state_mean = self.initial_state_mean
            initial_state_covariance = self.initial_state_covariance        
        
        self.update_parameters(transition_matrices = transition_matrices, 
                               observation_matrices = observation_matrices, 
                               observation_covariance = observation_covariance,
                               initial_state_mean = initial_state_mean,
                               initial_state_covariance = initial_state_covariance)

    def update_parameters(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

        # Transition covariance is constrained
        self.transition_covariance = np.eye(self.transition_matrices.shape[0]) - self.transition_matrices @ self.transition_matrices.T
