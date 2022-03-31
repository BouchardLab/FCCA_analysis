import numpy as np
import scipy
from scipy.optimize import minimize
from tqdm import tqdm
import time
import pdb
import torch
from torch import nn
from pyuoi.linear_model.var import VAR
from subspaces import SubspaceIdentification, IteratedStableEstimator, form_lag_matrix

from pykalman.standard import (_em_observation_matrix, _em_observation_covariance, 
                               _em_initial_state_mean, _em_initial_state_covariance,
                               _smooth_pair)
from pykalman.utils import log_multivariate_normal_density
from riccati import check_dare, sqrt_filter, sqrt_smoother
from subspaces import form_lag_matrix
from dstableFGM import dstable_descent

def filter(y, A, C, Q, R, P0, x0, S=None):

    # Run the square root Riccati recursion

    # Keep track of square root factors as they are needed by the smoother. Also assemble Re factors
    Psqrt_filt = np.zeros((y.shape[0],) + P0.shape)
    Psqrt_pred = np.zeros(Psqrt_filt.shape)
    Ppred = np.zeros(Psqrt_pred.shape)
    Re = np.zeros((y.shape[0], y.shape[1], y.shape[1]))    

    K = np.zeros((y.shape[0], P0.shape[0], y.shape[1]))
    mufilt = np.zeros((y.shape[0], P0.shape[0]))
    mupred = np.zeros((y.shape[0], P0.shape[0]))

    if S is not None:
        A = A - S @ np.linalg.inv(R) @ C
        Q = Q - S @ np.linalg.inv(R) @ S.T

    Psqrt_pred_ = np.linalg.cholesky(P0)
    Qsqrt = scipy.linalg.sqrtm(Q)
    Rsqrt = np.linalg.cholesky(R)

    x = x0
    for i in range(y.shape[0]):
        Psqrt_pred[i] = Psqrt_pred_
        Ppred[i] = Psqrt_pred_ @ Psqrt_pred_.T
        Re[i] = np.linalg.pinv(R + C @ Ppred[i] @ C.T)
        # Note this definition differs from the usual definition of Kalman gain (missing a factor of A), but 
        # is the one needed to accomplish separate filtering/prediction updates
        K[i] = Ppred[i] @ C.T @ Re[i]

        if i == 0:
            mupred[i] = x0
        else:
            mupred[i] = A @ mufilt[i - 1]

        # Incorporate new observation and step the prediction by 1
        Psqrt_pred_, Psqrt_filt_ = sqrt_filter(Psqrt_pred_, A, C, Qsqrt, Rsqrt)

        Psqrt_filt[i] = Psqrt_filt_              
        mufilt[i] = mupred[i] + K[i] @ (y[i] - C @ mupred[i])

    return mufilt, mupred, Psqrt_filt, Psqrt_pred, Ppred, K

def smooth(y, A, C, Q, R, mufilt, mupred, Psqrt_filt, Ppred, S=None):
    
    Psmooth = np.zeros(Psqrt_filt.shape)
    # In the process of calculating the smoothing covariances, we can assemble the filter covariances as well
    Pfilt = np.zeros(Psqrt_filt.shape)
    musmooth = np.zeros(mufilt.shape)
    J = np.zeros((Psqrt_filt.shape[0] - 1, Psqrt_filt.shape[1], mufilt.shape[1]))

    if S is not None:
        A = A - S @ np.linalg.inv(R) @ C
        Q = Q - S @ np.linalg.inv(R) @ S.T

    Pfilt[-1] = Psqrt_filt[-1] @ Psqrt_filt[-1].T
    Psmooth[-1] = Pfilt[-1]
    musmooth[-1] = mufilt[-1]

    Psqrt_smooth = np.linalg.cholesky(Psmooth[-1])
    Qsqrt = scipy.linalg.sqrtm(Q)
    for i in reversed(range(Psmooth.shape[0] - 1)):        
        Pfilt[i] = Psqrt_filt[i] @ Psqrt_filt[i].T
        J[i] = Pfilt[i] @ A.T @ np.linalg.pinv(Ppred[i + 1])  

        musmooth[i] = mufilt[i] + J[i] @ (musmooth[i + 1] - mupred[i + 1])
        Psqrt_smooth = sqrt_smoother(Psqrt_smooth, Psqrt_filt[i], A, Qsqrt, J[i])
        Psmooth[i] = Psqrt_smooth @ Psqrt_smooth.T

    return musmooth, Psmooth, Pfilt, J

def loglikelihood(y, A, C, R, mupred, Ppred):
    """
    Returns
    -------
    loglikelihoods: [n_timesteps] array
        `loglikelihoods[t]` is the probability density of the observation
        generated at time step t
    innovations: [n_timesteps] array
        difference between predicted and actual observtions at time step t
    """
    n_timesteps = y.shape[0]
    loglikelihoods = np.zeros(n_timesteps)
    innovations = np.zeros(y.shape)
    for t in range(n_timesteps):
        ypred = C @ mupred[t] 
        Sigmay = R + C @ Ppred[t] @ C.T
        innovations[t] = y[t] - ypred
#        loglikelihoods[t] = scipy.stats.multivariate_normal.logpdf()

        loglikelihoods[t] = log_multivariate_normal_density(
            y[t][np.newaxis, :],
            ypred[np.newaxis, :],
            Sigmay[np.newaxis, :, :]
        )

    return loglikelihoods, innovations 


def _em_stable_transition_matrix(Ainit, Pt, Pt1, Pt1t, lambda_A, T):

    n = Ainit.shape[0]

    # Portion of the cost function that depends on A. Note that this is the *negative* of the log likelihood so 
    # we can minimize
    Q = lambda A: np.eye(A.shape[0]) - A @ A.T

    def f(A):        
        A = np.reshape(A, (n, n))
        return 0.5 * np.linalg.slogdet(Q(A))[1] + \
                lambda_A/(2*(T - 1)) * np.trace((A - np.eye(A.shape[0])) @ (A - np.eye(A.shape[0])).T) + \
                0.5 * np.trace(np.linalg.inv(Q(A)) @ (A @ Pt1 @ A.T - A @ Pt1t.T - Pt1t @ A.T + Pt))

    def df(A):
        A = np.reshape(A, (n, n))
        dfdA =  np.linalg.inv(Q(A)) @ (-A + (A @ Pt1 - Pt1t) \
               + (A @ Pt @ A.T - A @ Pt1t.T - Pt1t @ A.T + Pt1) @ np.linalg.inv(Q(A)) @ A)
        return dfdA.ravel()

    # Gradient with respect to A
    opt = minimize(f, Ainit.ravel(), method='Newton-CG', jac=df)
    
    return opt.x.reshape(n, n)

class StateSpaceML():

    def __init__(self, init_strategy='SSID', tol=1e-2, max_iter=100,
                 rand_state=None, optimize_init_cond=True):

        self.tol = tol
        self.max_iter = max_iter
        self.optimize_init_cond = optimize_init_cond
        if init_strategy not in ['PCA', 'SSID', 'manual']:
            raise ValueError('Unknown initialization strategy specified.')

        self.init_strategy = init_strategy
        self.correlated_noise = True

    def init_params(self, y, state_dim, **init_kwargs):
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
            Sinit = np.zeros((Q.shape[0], R.shape[1]))
            Cinit = np.block([[np.eye(y.shape[1])], [np.zeros((Ainit.shape[0] - y.shape[1], y.shape[1]))]])
            Sigma0 = scipy.linalg.solve_discrete_lyapunov(Ainit, Qinit)
            x0 = np.zeros(state_dim)

        elif self.init_strategy == 'SSID':
            # Fit stable subpsace identification
            ssid = SubspaceIdentification(estimator=IteratedStableEstimator, obs_regressor='OLS', **init_kwargs)
            A, C, Cbar, L0, Q, R, S = ssid.identify(y, order=state_dim, T=3)
            Ainit = A
            Cinit = C
            # Q is constrained to be the identity
            Qinit = np.eye(A.shape[0])
            Rinit = R
            Sinit = S

            # Initial state mean set to PCA initialization, covariance set to solution of Lyapunov equation
            x0 = np.zeros(state_dim)
            Sigma0 = scipy.linalg.solve_discrete_lyapunov(A, Q)
        

        filter_params = {
            'A':Ainit,
            'C':Cinit,
            'Q':Qinit,
            'R':Rinit,
            'x0':x0,
            'Sigma0':Sigma0,
            'S':Sinit if self.correlated_noise else np.zeros((Qinit.shape[0], Rinit.shape[1]))
        }

        for key, val in filter_params.items():
            setattr(self, key, val)     

    def fit(self, y, state_dim, **init_kwargs):
        self.init_params(y, state_dim, **init_kwargs)

        iter = 0
        tol = np.inf

        while iter < self.max_iter and tol > self.tol:
            t0 = time.time()
            M_args, logll = self.E(y)
            print('E step: %f' % (time.time() - t0))
            t0 = time.time()
            self.M(y, *M_args)
            print('M step: %f' % (time.time() - t0))
            iter += 1
            # print('Iteration %d, Log Likelihood: %f' % (iter, logll))


    def E(self, y):
        (mufilt, mupred, Psqrt_filt, 
        Psqrt_pred, Ppred, K) = filter(y, self.A, self.C, self.Q, self.R, 
                                       self.Sigma0, self.x0, S=self.S)

        musmooth, Psmooth, Pfilt, J = smooth(y, self.A, self.C, self.Q, self.R,
                                             mufilt, mupred, Psqrt_filt, Ppred, S=self.S) 


        Pt1t_smooth = _smooth_pair(Psmooth, J)

        # Test whether the sequence of predicted state covariances and smoothed state covariances
        # remains positive definite
        e1 = []
        for i in range(Ppred.shape[0]):
            eig = np.linalg.eigvals(Ppred[i])
            e1.append(np.min(eig))
        e2 = []
        for i in range(Psmooth.shape[0]):
            eig = np.linalg.eigvals(Psmooth[i])
            e2.append(np.min(eig))

        loglikelihoods, innovations = loglikelihood(y, self.A, self.C, self.R, mupred, Ppred)

        logll = np.sum(loglikelihoods)
        return mufilt, mupred, musmooth, Pfilt, Ppred, Psmooth, Pt1t_smooth, innovations, logll

    def M(self, **kwargs):
        pass

    def update_parameters(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


class StableStateSpaceML(StateSpaceML):

    def __init__(self, init_strategy='SSID', tol=1e-2, max_iter=100, 
                 lambda_A=1, rand_state=None, optimize_init_cond=True):

        super(StableStateSpaceML, self).__init__(init_strategy, tol, max_iter, 
                                                 rand_state, optimize_init_cond)
        self.lambda_A = lambda_A

        # Do not allow correlations between R and Q
        self.correlated_noise = False

    def E(self, y):
        _, _, musmooth, _, _, Psmooth, Pt1t_smooth, _, logll = super(StableStateSpaceML, self).E(y)

        return (musmooth, Psmooth, Pt1t_smooth), logll

    def M(self, y, musmooth, Psmooth, Ptt1_smooth):
        
        # Pi = np.block([[self.Q, self.S], [self.S.T, self.R]])        
        # Eyy = np.cov(y, bias=True)
        # Eyxt1 = 1/(y.shape[0] - 1) * (y[:-1].T @ musmooth[1:])        
        # Eyxt =  1/y.shape[0] * (y.T @ musmooth)
        Ext1xt1 = np.cov(musmooth[1:], rowvar=False, bias=True) + np.mean(Psmooth[1:], axis=0)
        Extxt = np.cov(musmooth[:-1], rowvar=False, bias=True) + np.mean(Psmooth[:-1], axis=0)
        # assert(Ptt1_smooth.shape[0] == musmooth.shape[0] - 1)
        Ext1xt = 1/(musmooth.shape[0] - 1) * (musmooth[1:, :].T @ musmooth[:-1, :]) + np.mean(Ptt1_smooth, axis=0)

        # Form the equivalent of the quantities Phi, Psi, Sigma (eq 23-25) from 
        # Robust maximum-likelihood estimation of multivariable dynamic systems
        # Here, we have no external inputs
        # phi = np.block([[Ext1xt1, Eyxt1.T], [Eyxt1, Eyy]])
        # psi = np.block([[Ext1xt], [Eyxt]])
        # sigma = Extxt

        # Gamma = np.block([[self.A], [self.C]])        

        A = _em_stable_transition_matrix(self.A, Extxt, Ext1xt1, Ext1xt, self.lambda_A, y.shape[0])        

        # C unchanged
        C = _em_observation_matrix(y, np.zeros(y.shape), musmooth, Psmooth)

        # This should be done robustly
        R = _em_observation_covariance(
            y, np.zeros(y.shape),
            self.C, musmooth,
            Psmooth
        )   

        # transition_covariance = _em_transition_covariance(
        #     self.transition_matrix, transition_offsets,
        #     smoothed_state_means, smoothed_state_covariances,
        #     pairwise_covariances
        # )
        # transition covariance is constrained to be 1 - A A^\top


        if self.optimize_init_cond:
            x0 = _em_initial_state_mean(musmooth)
            Sigma0 = _em_initial_state_covariance(x0, musmooth, Psmooth)
        else:
            x0 = self.x0
            Sigma0 = self.Sigma0        
        
        self.update_parameters(A = A, 
                               C = C, 
                               R = R,
                               initial_state_mean = x0,
                               initial_state_covariance = Sigma0)


    def update_parameters(self, **kwargs):
        super().update_parameters(**kwargs)

        # Transition covariance is constrained
        self.Q = np.eye(self.A.shape[0]) - self.A @ self.A.T
        # No correlation between noises
        self.S = np.zeros((self.Q.shape[0], self.R.shape[0]))

        # Check whether modified dare still has a PSD solution
        As = self.A - self.S @ np.linalg.inv(self.R) @ self.C
        Qs = self.Q - self.S @ np.linalg.inv(self.R) @ self.S.T

        # print('Riccati check: %r' % check_dare(As, self.C, Qs, self.R))


class DifferentiableKF(nn.Module):

    def __init__(self, A, C, K, x0, P0):
        super().__init__()
        self.A = nn.Parameter(torch.tensor(A, dtype=torch.float32))
        self.C = nn.Parameter(torch.tensor(C, dtype=torch.float32))
        self.K = nn.Parameter(torch.tensor(K, dtype=torch.float32))

        self.x0 = torch.tensor(x0, dtype=torch.float32)
        self.Psqrt0 = torch.cholesky(torch.tensor(P0)).float()        

    def forward(self, observations):
        observations = torch.tensor(observations, dtype=torch.float32)
        epsilon = observations[0]

        Psqrt = self.Psqrt0
        x = self.x0
        yvar = torch.ger(epsilon, epsilon) + torch.chain_matmul(self.C, Psqrt, torch.t(Psqrt), torch.t(self.C))

        for i in range(observations.shape[0]):
            Psqrt = torch.matmul(self.A - torch.matmul(self.K, self.C), Psqrt)
            x = torch.matmul(self.A - torch.matmul(self.K, self.C), x) + torch.matmul(self.K, observations[i - 1])
            ypred = torch.matmul(self.C, x)
            epsilon = observations[i] - ypred
            yvar += torch.ger(epsilon, epsilon) + torch.chain_matmul(self.C, Psqrt, torch.t(Psqrt), torch.t(self.C))
         
        yvar *= 1/observations.shape[0]
        loss = torch.slogdet(yvar)[1]
        return loss

def _em_ACK(y, A, C, K, x0, P0, optim='Adam'):

    mstepfilter = DifferentiableKF(A, C, K, x0, P0)

    loss_history = []

    #opt = torch.optim.LBFGS(params=mstepfilter.parameters())        
    if optim == 'Adam':
        opt = torch.optim.Adam(mstepfilter.parameters())
    elif optim == 'Adadelta':
        opt = torch.optim.Adadelta(mstepfilter.parameters())
    elif optim == 'RMSprop':
        opt = torch.optim.Adadelta(mstepfilter.parameters())
    elif optim == 'LBFGS':
        opt = torch.optim.LBFGS(mstepfilter.parameters())

    for i in tqdm(range(100)):
        opt.zero_grad()
        loss = mstepfilter(y)
        loss.backward()
        opt.step(lambda : mstepfilter.forward(y))
        loss_history.append(loss)

    # # Adjust for stability
    if max(np.abs(np.linalg.eigvals(A))) > 0.99:
        A = dstable_descent(A)
    # This may not guarantee positive realnesS. In this case, we should explore 
    # implementing this method: Sharma, FINDING THE NEAREST POSITIVE-REAL SYSTEM
    
    return A, C, K

class ARMAStateSpaceML(StateSpaceML):

    def __init__(self, init_strategy='SSID', tol=1e-2, max_iter=100, 
                 lambda_A=1, rand_state=None, optimize_init_cond=True):
        super(ARMAStateSpaceML, self).__init__(init_strategy=init_strategy, tol=tol, max_iter=max_iter, 
                                               rand_state=rand_state, optimize_init_cond=optimize_init_cond)
        # Parameter initialization needs to be modified since we deal with R and K, not R and Q. Q = K R K.T
        self.correlated_noise = False

    def init_params(self, y, state_dim, **init_kwargs):
        if self.init_strategy == 'manual':
            Ainit = init_kwargs['Ainit']
            Cinit = init_kwargs['Cinit']
            Kinit = init_kwargs['Kinit']
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
            Rinit = 1e-3 * np.eye(y.shape[1])
            Cinit = np.block([[np.eye(y.shape[1])], [np.zeros((Ainit.shape[0] - y.shape[1], y.shape[1]))]])
            
            ##### This needs to be checked. There is a proper way to map VAR to state space in innovations form
            Qinit = np.eye(Ainit.shape[0])
            # Obtain K from solution of Riccati equation
            Pinf = scipy.linalg.solve_discrete_are(Ainit.T, Cinit.T, Qinit, Rinit)
            Kinit = (Ainit @ Pinf @ Cinit.T) @ np.linalg.pinv(Rinit + Cinit @ Pinf @ Cinit.T) 
            # Modify Qinit apporpriately

            Sigma0 = scipy.linalg.solve_discrete_lyapunov(Ainit, Qinit)
            x0 = np.zeros(state_dim)

        elif self.init_strategy == 'SSID':
            # Fit stable subpsace identification
            ssid = SubspaceIdentification(estimator=IteratedStableEstimator, obs_regressor='OLS', **init_kwargs)
            A, C, Cbar, L0, Q, R, S = ssid.identify(y, order=state_dim, T=3)
            # Solve for Kalman gain
            Ainit = A
            Cinit = C

            Rinit = R
            Pinf = scipy.linalg.solve_discrete_are(A.T, C.T, np.zeros(A.shape), L0, s=Cbar.T)
            Kinit = (Cbar.T - Ainit @ Pinf @ Cinit.T) @ np.linalg.pinv(L0 - Cinit @ Pinf @ Cinit.T)
            Qinit = Kinit @ Rinit @ Kinit.T

            # Initial state mean set to PCA initialization, covariance set to solution of Lyapunov equation
            x0 = np.zeros(state_dim)
            Sigma0 = scipy.linalg.solve_discrete_lyapunov(A, Qinit)
        

        filter_params = {
            'A':Ainit,
            'C':Cinit,
            'Q':Qinit,
            'K':Kinit,
            'R':Rinit,
            'x0':x0,
            'S': np.zeros(Kinit.shape),
            'Sigma0':Sigma0,
        }

        for key, val in filter_params.items():
            setattr(self, key, val)     

    def E(self, y):

        # Need to be memory efficient - do not store anything here


        _, _, musmooth, _, Ppred, Psmooth, Pt1t_smooth, innovations, logll = super(ARMAStateSpaceML, self).E(y)
        return (musmooth, Psmooth, Ppred, innovations), logll


    def M(self, y, musmooth, Psmooth, Ppred, innovations):

        if self.optimize_init_cond:
            x0 = _em_initial_state_mean(musmooth)
            Sigma0 = _em_initial_state_covariance(x0, musmooth, Psmooth)
        else:
            x0 = self.x0
            Sigma0 = self.Sigma0        

        # A, C, K estimated jointly 
        A, C, K = _em_ACK(y, self.A, self.C, self.K, x0, Sigma0)

        # # Adjust for stability
        # if max(np.abs(np.linalg.eigvals(A))) > 0.99:
        #     x = 


        #     A = IteratedStableEstimator.solve_qp(A, x0, x1)



        # Observaton covariance - experiment with updating this before/after solving for A, C, K
        R = np.mean([np.outer(epsilon, epsilon) + self.C @ Ppred[idx] @ self.C.T for idx, epsilon in enumerate(innovations)], axis=0)
        self.update_parameters(A = A, 
                               C = C,
                               K = K, 
                               R = R,
                               initial_state_mean = x0,
                               initial_state_covariance = Sigma0)