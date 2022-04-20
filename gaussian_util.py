from re import I
from statistics import mean
import numpy as np
import scipy 
import torch
import pandas as pd
import pickle
import pdb
from tqdm import tqdm
from dca.cov_util import calc_cross_cov_mats_from_data, calc_cov_from_cross_cov_mats, form_lag_matrix

# Calculate mutual information from joint and marginal covariance  
def mutual_information(covjoint, covx, covy):
    use_torch = isinstance(covjoint, torch.Tensor)

    if use_torch:
        return 0.5 * (torch.slogdet(covx)[1] + torch.slogdet(covy)[1] - torch.slogdet(covjoint)[1])
    else:
        return 0.5 * (np.linalg.slogdet(covx)[1] + np.linalg.slogdet(covy)[1] - np.linalg.slogdet(covjoint)[1])

# KL(P|Q) where *cov2* corresponds to P
def gaussian_KL(cov1, cov2):
    trace = np.trace(np.linalg.solve(cov1, cov2))
    logdets = np.linalg.slogdet(cov1)[1] - np.linalg.slogdet(cov2)[1]
    kl = .5 * (trace + logdets - cov2.shape[0])
    return kl

# log Gaussian distribution
def logprob(x, cov):
     return -x.size/2 * np.log(2 * np.pi) - 1/2 * np.linalg.slogdet(cov)[1] - 1/2 * x.T @ np.linalg.inv(cov) @ x

# Gaussian entropy
def entropy(cov):
    d = cov.shape[0]
    return 0.5 * d * (np.log(2 * np.pi) + 1) + 0.5 * np.linalg.slogdet(cov)[1]

# Gaussian cross entropy - cov2, mu2 correspond to the distrubution over which expectation are taken
def cross_entropy(cov1, cov2, mu1, mu2):
    d = cov1.shape[0]
    cov1inv = np.linalg.inv(cov1)
    return 0.5 * (d * np.log(2 * np.pi) + np.linalg.slogdet(cov1)[1] + np.trace(cov1inv @ cov2) +\
                  mu2 @ cov1inv @ mu2 - mu1 @ cov1inv @ mu2 - mu2 @ cov1inv @ mu1 + mu1 @ cov1inv @ mu1)

# Sample from conditional distribution of 1 given 2
def conditional_samples(cov1, mu1, cov2, mu2, cross_cov, n_inner_samples, n_outer_samples):

    cond_cov = cov1 - cross_cov @ np.linalg.inv(cov2) @ cross_cov.T
    mu = lambda x2: mu1 + cross_cov @ np.linalg.inv(cov2) @ (x2 - mu2)

    # Draw outer samples (from distribution of x2)
    outer_samples = np.random.multivariate_normal(mean=mu2, cov=cov2, size=n_outer_samples)
    inner_samples = []
    for sample in outer_samples:
        inner_samples.append(np.random.multivariate_normal(mean=mu(sample), cov=cond_cov, size=n_inner_samples))

    return inner_samples, outer_samples

# Return the covariance of y_n conditioned on its own past and (optionally) the history and current value of x
def conditional_cov(covy, ydim, T, covx=None, covxy=None, xdim=None, return_covs=False):

    covypast = covy[:-ydim, :-ydim]
    covyi = covy[-ydim:, -ydim:]

    if covx is not None:
        # Assuming that covxy contains the cross-covariance between x^T and y_T, so we
        # remove it to calculate cov_given
        covxypast = covxy[:, :-ydim]
        cov_given = np.block([[covx, covxypast], [covxypast.T, covypast]])
        # Cross cov of y_i with x^i
        cross_cov = np.hstack([covxy[:, -ydim:].T, covy[-ydim:, :-ydim]])
    else:
        cov_given = covypast
        cross_cov = covy[-ydim:, :-ydim]

    if return_covs:
        return covyi - cross_cov @ np.linalg.inv(cov_given) @ cross_cov.T, covyi, cross_cov, cov_given
    else:
        return covyi - cross_cov @ np.linalg.inv(cov_given) @ cross_cov.T

# Evaluate the KL divergence in eq. 9 of https://arxiv.org/pdf/2003.04179.pdf using its definition as the difference 
# between the i^th CMI and DKL of Y between its block and marginal distribution
def DKL_cc(x, y, T):
    ydim = y.shape[1]
    xdim = x.shape[1]

    autocovs = np.array([separate_blocks(c, xdim) for c in calc_cross_cov_mats_from_data(np.hstack([x, y]), T)])
    covXT, covYT, covXYT, covYXT = joint_cov_from_cross_cov_mats(autocovs)

    XI = form_lag_matrix(x, T + 1)
    YI = form_lag_matrix(y, T + 1)
    YI1 = YI[:, :-ydim]
    Y_I = YI[:, -ydim:]
    I = gaussian_CMI(XI, YI1, Y_I)
    
    DKL_Y = gaussian_KL(scipy.linalg.block_diag(covYT[:-ydim, :-ydim], covYT[-ydim:, -ydim:]), covYT)
    return I + DKL_Y

# Estimate the entropy rate of x using a decomposition into the cross entropy and KL divergence
def entropy_rate(x, T):
    xdim = x.shape[1]
    XT = form_lag_matrix(x, T)
    
    covT = np.cov(XT, rowvar=False)
    #covT1 = np.cov(form_lag_matrix(x, T - 1), rowvar=False)
    covT1 = covT[:-xdim, :-xdim]
    covx = np.cov(x, rowvar=False)

    HT = entropy(covT)
    product_cov = scipy.linalg.block_diag(covT1, covx)
    cross_entropy = -np.mean([logprob(xi, product_cov) for xi in XT])

    DKL = gaussian_KL(product_cov, covT)
    # DKL = np.mean([logprob(xi, covT) - logprob(xi, product_cov) for xi in XT])
    return HT, cross_entropy - DKL, -np.mean([logprob(xi, covT) for xi in XT])

# Separate joint covariance matrix between x and y
def separate_blocks(P, dimx):
    PZ = P[0:dimx, 0:dimx]
    PY = P[dimx:, dimx:]
    PZY = P[0:dimx, dimx:]
    PYZ = P[dimx:, 0:dimx]

    return PZ, PY, PZY, PYZ

######## Utils for gausiian_model ##########
def joint_cov_from_cross_cov_mats(autocovs):
    covXT = calc_cov_from_cross_cov_mats(np.array([autocovs[k][0] for k in range(len(autocovs))]))
    covYT = calc_cov_from_cross_cov_mats(np.array([autocovs[k][1] for k in range(len(autocovs))]))
    
    T = len(autocovs)
    xdim = autocovs[0][0].shape[0]
    ydim = autocovs[0][1].shape[0]
    
    
    # Off-diagonals
    cross_cov_XYT = []
    for i in range(T):

        for j in range(T):
            if i > j:
                cross_cov_XYT.append(autocovs[abs(i - j)][2])
            else:
                cross_cov_XYT.append(autocovs[abs(i - j)][3].T)

    covXYT_tensor = np.reshape(np.stack(cross_cov_XYT), (T, T, xdim, ydim))
    covXYT = np.concatenate([np.concatenate([cov_ii_jj for cov_ii_jj in cov_ii], axis=1)
                            for cov_ii in covXYT_tensor])

    covYXT = covXYT.T
    return covXT, covYT, covXYT, covYXT


# Currently assumes full state observation (Cy, Cz = I), otherwise we have to solve 
# KF...
def state_space_cc_distr(T, AX, AY, AXY, Cy, Cz):

    ydim = AY.shape[0]
    xdim = AX.shape[0]
    # Assumes X -> Y

    # Solve Lyapunov equation for cascaded system
    A = np.block([[AX, np.zeros((AX.shape[0], AY.shape[1]))], [AXY, AY]])
    Pi = scipy.linalg.solve_discrete_lyapunov(A, np.eye(A.shape[0]))
    
    autocovs = np.array([separate_blocks(np.linalg.matrix_power(A, k) @ Pi, xdim) for k in range(T)])
    covXT, covYT, covXYT, covYXT = joint_cov_from_cross_cov_mats(autocovs)

    # For each i 1 to T, return the sequence of causally conditioned covariances
    cc_cov = []
    for i in range(1, T + 1):
        cc_cov.append(conditional_cov(covYT[0:i*ydim, 0:i*ydim], ydim, i, covx=covXT[0:i*xdim, 0:i*xdim],
                                      covxy=covXYT[0:i*xdim, 0:i*ydim], xdim=xdim))

    return cc_cov, covYT


# Calculate gaussian DI from covariance matrices using the difference between the block entropy and the 
# causally conditioned entropy
def gaussian_DI2(x, y, T):

    ydim = y.shape[1]
    xdim = x.shape[1]
    autocovs = np.array([separate_blocks(c, xdim) for c in calc_cross_cov_mats_from_data(np.hstack([x, y]), T)])
    covXT, covYT, covXYT, covYXT = joint_cov_from_cross_cov_mats(autocovs)
    cc_cov = []
    for i in range(1, T + 1):
        cc_cov.append(conditional_cov(covYT[0:i*ydim, 0:i*ydim], ydim, i, covx=covXT[0:i*xdim, 0:i*xdim],
                                        covxy=covXYT[0:i*xdim, 0:i*ydim], xdim=xdim))

    Hcc = entropy(scipy.linalg.block_diag(*cc_cov))
    Hy = entropy(covYT)
    return Hy - Hcc

def gaussian_CMI(x, y, z):
    xdim = x.shape[1]
    ydim = y.shape[1]

    if z is not None:
        zdim = z.shape[1]

        cov_joint = np.cov(np.hstack([x, y, z]), rowvar=False)

        covx = cov_joint[0:xdim, 0:xdim]
        covy = cov_joint[xdim:(xdim + ydim), xdim:(xdim + ydim)]
        covz = cov_joint[-zdim:, -zdim:]

        covyz = cov_joint[xdim:, xdim:]
        covxz = np.cov(np.hstack([x, z]), rowvar=False)

        # Evaluate CMI as the difference of 2 mutual informations
        I1 = mutual_information(cov_joint, covx, covyz)
        I2 = mutual_information(covxz, covx, covz)

        return I1 - I2

    else:
        # Return MI beetween X and Y
        cov_joint = np.cov(np.hstack([x, y]), rowvar=False)
        covx = cov_joint[0:xdim, 0:xdim]
        covy = cov_joint[xdim:(xdim + ydim), xdim:(xdim + ydim)]
        Hjoint = entropy(cov_joint)
        Hx = entropy(covx)
        Hy = entropy(covy)
        return Hx + Hy - Hjoint        

def gaussian_DI(x, y, T):

    xdim = x.shape[1]
    ydim = y.shape[1]

    DI = 0

    # Handle i = 1 term
    DI += gaussian_CMI(x, y, None)

    for i in range(1, T):
        XI = form_lag_matrix(x, i + 1)
        YI = form_lag_matrix(y, i + 1)
        YI1 = YI[:, :-ydim]
        Y_I = YI[:, -ydim:]
        DI += gaussian_CMI(XI, Y_I, YI1)

    return DI

# Trialized x and y 
def gaussian_DI_trialized(x, y, T):

    xdim = x[0].shape[1]
    ydim = y[0].shape[1]

    DI = 0

    # Handle i = 1 term
    xjoined = np.vstack(x)
    yjoined = np.vstack(y)

    DI += gaussian_CMI(xjoined, yjoined, None)

    for i in range(1, T):
        # Lag and then join
        XI = np.vstack([form_lag_matrix(xi, i + 1) for xi in x])
        YI = [form_lag_matrix(yi, i + 1) for yi in y]

        YI1 = np.vstack([yi[:, :-ydim] for yi in YI])
        Y_I = np.vstack([yi[:, -ydim:] for yi in YI])

    DI += gaussian_CMI(XI, Y_I, YI1)
 
    return DI

# Generate synthetic data for which we can evaluate the Directed Information exactly
def gaussian_model(T=3, N=int(1e4), dim_Y=10, dim_y=5, dim_Z=10, dim_z=5, AY=None, AZ=None, AYZ=None, seed=None):

    if seed is not None:
        state = np.random.RandomState(seed)
    else:
        state = np.random.RandomState(np.random.randint(1e4))

    if AY is None:
        AY = state.normal(scale = 1/np.sqrt(2 * dim_Y), size=(dim_Y, dim_Y))
        # Ensure stability
        while max(np.abs(np.linalg.eigvals(AY))) > 0.99:
            AY = state.normal(scale = 1/np.sqrt(2 * dim_Y), size=(dim_Y, dim_Y))

    if AZ is None:
        AZ = state.normal(scale = 1/np.sqrt(2 * dim_Z), size=(dim_Z, dim_Z))
        # Ensure stability
        while max(np.abs(np.linalg.eigvals(AZ))) > 0.99:
            AZ = state.normal(scale = 1/np.sqrt(2 * dim_Z), size=(dim_Z, dim_Z))

    if AYZ is None:
        AYZ = state.normal(size=(dim_Y, dim_Z))

    # We wait for the transients to 
    # dissipate by waiting 5x the slowest autocorrelation time
    eigvals = np.linalg.eigvals(AY)
    slow_mode = np.max(np.abs(np.real(eigvals)))  
    burnoff_timeY = int(np.abs(np.log(0.001)/np.log(slow_mode)))

    # y and z are obtained from random projections of Y and Z
    #Cy = scipy.stats.ortho_group.rvs(dim_Y, random_state=seed)[0:dim_y, :]
    #Cz = scipy.stats.ortho_group.rvs(dim_Z, random_state=seed)[0:dim_z, :]
    Cy = np.eye(AY.shape[0])
    Cz = np.eye(AZ.shape[0])

    eigvals = np.linalg.eigvals(AZ)
    slow_mode = np.max(np.abs(np.real(eigvals)))  
    burnoff_timeZ = int(np.abs(np.log(0.001)/np.log(slow_mode)))

    # AlwAYs throw awAY the first point
    burnoff_time = max(1, max(burnoff_timeZ, burnoff_timeY))

    # Generate white noise trajectories. Note that we use uncorrelated white noise
    # processes 
    w = state.normal(size=(burnoff_time + N, dim_Z))
    v = state.normal(size=(burnoff_time + N, dim_Y))
    
    # Random initial condition
    Y0 = state.normal(size=(dim_Y,))
    Z0 = state.normal(size=(dim_Z,))

    Z = np.zeros((burnoff_time + N, dim_Z))
    Y = np.zeros((burnoff_time + N, dim_Y))

    Z[0, :] = Z0
    Y[0, :] = Y0

    # Propagate forward 
    for t in range(1, burnoff_time + N):
        Z[t] = AZ @ Z[t - 1] + w[t]
        Y[t] = AY @ Y[t - 1] + AYZ @ Z[t - 1] + v[t]

    Z = Z[burnoff_time:]
    Y = Y[burnoff_time:]

    z = Z @ Cz.T
    y = Y @ Cy.T        
    
    # # Determine the causally conditioned distributions analytically
    # # Currently requires Cz = Cy = I
    # # Assumes Z -> Y
    # cov_cc, covYT = state_space_cc_distr(T, AZ, AY, AYZ, Cy, Cz)
        
    # # # # Calculate the entropy of the causally conditioned distributions
    # # Hcc = sum([entropy(cov_cc[i]) for i in range(len(cov_cc))])
    # # HY = entropy(covYT)

    # # # Compare to the value implied by the 

    # # DI_zy_analytic = HY - Hcc

    # Calculate DI from Z to Y
    DI_zy = gaussian_DI(z, y, T=T)
    # Calculate DI from Y to Z
    DI_yz = gaussian_DI(y, z, T=T)

    return y, z, DI_zy, DI_yz, AZ, AY, AYZ

# Implement the examples provided here:
#  
def AR1_test(n_samples=int(2*1e4), dim=1, condition=1):

    # Used to test conditional mutual information
    if condition == 1:
        # Simulate data across 10 trials
        n_trials = 10
        sigma_x = np.linspace(0, 4, 20)
        sigma_y = 1
        sigma_z = 1
        
        X = np.zeros((n_trials, sigma_x.size, n_samples, dim))
        Y = np.zeros((n_trials, sigma_x.size, n_samples, dim))
        Z = np.zeros((n_trials, sigma_x.size, n_samples, dim))

        A = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
        B = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])

        CMI_est = np.zeros((n_trials, sigma_x.size))
        CMI_true = np.zeros((n_trials, sigma_x.size))

        for trial in tqdm(range(n_trials)):
            for j, sx in enumerate(sigma_x):

                # generate data
                for k in range(1, n_samples):

                    # 't' = 0:
                    y0 = np.random.normal(scale=np.sqrt(sigma_y), size=(1, dim))
                    z0 = np.random.normal(scale=np.sqrt(sigma_z), size=(1, dim))

                    # 't' = 1:
                    y1 = z0 + np.random.normal(scale=np.sqrt(sigma_y), size=(1, dim))
                    z1 = np.random.normal(scale=np.sqrt(sigma_z), size=(1, dim))
                    x1 = y1 + y0 + np.random.normal(scale=np.sqrt(sx), size=(1, dim))

                    X[trial, j, k] = x1
                    Y[trial, j, k] = y1
                    Z[trial, j, k] = z1

                # Calculate the conditional mutual information
                CMI_est[trial, j] = gaussian_CMI(X[trial, j], Y[trial, j], Z[trial, j])

                # One can just calculate it exactly:
                Hxyz = 0.5 * np.linalg.slogdet(np.array([[sx + 2 * sigma_y + sigma_z, sigma_y + sigma_z, 0], [sigma_y + sigma_z, sigma_y + sigma_z, 0],
                                                        [0, 0, sigma_z]]))[1]
                Hx = 0.5 * np.linalg.slogdet(np.array([[sx + 2 * sigma_y + sigma_z]]))[1]
                Hyz = 0.5 * np.linalg.slogdet(np.array([[sigma_y + sigma_z, 0], [0, sigma_z]]))[1]
                Hxz = 0.5 * np.linalg.slogdet(np.array([[sx + 2 * sigma_y + sigma_z, 0], [0, sigma_z]]))[1]
                Hz = 0.5 * np.linalg.slogdet(np.array([[sigma_z]]))[1]

                CMI_true[trial, j] = (Hx + Hyz - Hxyz) - (Hx + Hz - Hxz)

        return X, Y, Z, CMI_est, CMI_true


    # Used to test directed information
    elif condition == 2:

        n_trials = 1

        T = 3
        sigma_x = 1
        sigma_y = 1
        A = np.array([[0.8, 0], [0.5, 0.8]])

        X = np.zeros((n_trials, n_samples, dim))
        Y = np.zeros((n_trials, n_samples, dim))

        DI_est = np.zeros((n_trials, 2))

        for trial in range(n_trials):
            # Initialize randomly
            X[trial, 0, :] = np.random.normal(scale=np.sqrt(sigma_x), size=(dim,))
            Y[trial, 0, :] = np.random.normal(scale=np.sqrt(sigma_y), size=(dim,))

            # generate data
            for k in range(1, n_samples):
                xx = A @ np.vstack([X[trial, k - 1, :], 
                                    Y[trial, k - 1, :], 
                                    ])
                xx += np.vstack([np.random.normal(scale=np.sqrt(sigma_x), size=(1, dim)), 
                                    np.random.normal(scale=np.sqrt(sigma_y), size=(1, dim))])

                X[trial, k] = xx[0]
                Y[trial, k] = xx[1]

            # Estimate the various directed informations
            DI_est[trial, 0] = gaussian_DI(X[trial], Y[trial], T=T)
            DI_est[trial, 1] = gaussian_DI(Y[trial], X[trial], T=T)
        
        return X, Y, DI_est

def DKL_test():
    N = [int(1e3), int(5e3), int(1e4), int(5e4)]
    dim_Y=10
    dim_Z = [2, 5, 10]
    coupling_scale = np.linspace(0, 10)
    seed = 1234


    data_dict_list = []

    AY = np.random.normal(scale=1/(2 * np.sqrt(dim_Y)), size=(dim_Y, dim_Y))
    while max(np.abs(np.linalg.eigvals(AY))) > 1:
        AY = np.random.normal(scale=1/(2 * np.sqrt(dim_Y)), size=(dim_Y, dim_Y))

    for i in range(len(dim_Z)):
        AZ = np.random.normal(scale=1/(2 * np.sqrt(dim_Z[i])), size=(dim_Z[i], dim_Z[i]))
        while max(np.abs(np.linalg.eigvals(AZ))) > 1:
            AZ = np.random.normal(scale=1/(2 * np.sqrt(dim_Z[i])), size=(dim_Z[i], dim_Z[i]))

        # Iterate over the strength of the coupling term
        for j in range(coupling_scale.size):
            AYZ = np.random.normal(scale=coupling_scale[j]/(np.sqrt(dim_Z[i]) * np.sqrt(dim_Y)), size=(dim_Y, dim_Z[i]))
            
            for k in range(len(N)):
                y, z, DI_zy, DI_yz, AZ, AY, AYZ = gaussian_model(T=3, N=N[k], dim_Y=dim_Y, dim_y=dim_Y, dim_Z=dim_Z[i], dim_z=dim_Z[i],
                                                                AY=AY, AZ=AZ, AYZ=AYZ, seed=seed)
                DKL = DKL_cc(z, y, T=3)

                data_dict = {}
                data_dict['seed'] = 1234
                data_dict['N'] = N[k]
                data_dict['dim_Z'] = dim_Z[i]
                data_dict['AY'] = AY
                data_dict['AZ'] = AZ
                data_dict['AYZ'] = AYZ
                data_dict['DI_zy'] = DI_zy
                data_dict['DI_yz'] = DI_yz
                data_dict['z'] = z
                data_dict['y'] = y
                data_dict['DKL_cc'] = DKL

                data_dict_list.append(data_dict)        
    
    df = pd.DataFrame(data_dict_list)
    with open('gaussian_test.dat', 'wb') as f:
        f.write(pickle.dumps(df))

if __name__ == '__main__':
    DKL_test()