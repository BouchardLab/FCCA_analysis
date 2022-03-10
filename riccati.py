import numpy as np
import scipy 
from scipy import optimize
from scipy.stats import ortho_group

import pdb
from tqdm import tqdm

# Check for existence of solution by calculating eigenvalues of the matrix pencil
# (see eq 52 in Van Dooren A Generalized Eigenvalue Approach for Solving Riccati Equations)
def check_gdare(A, C, Cbar, L0, tol=1e-3):
    
    AA = np.block([[np.eye(A.shape[0]), -Cbar.T @ np.linalg.inv(L0) @ Cbar], 
                  [np.zeros(A.shape), A.T - C.T @ np.linalg.inv(L0) @ Cbar]])
    BB = np.block([[A - Cbar.T @ np.linalg.inv(L0) @ C, np.zeros(A.shape)],
                   [-C.T @ np.linalg.inv(L0) @ C, np.eye(A.shape[0])]])
    
    eig = scipy.linalg.eig(AA, b=BB, left=False, right=False)
    if np.any(np.bitwise_and(np.abs(eig) > 1 - tol, np.abs(eig) < 1 + tol)):
        return False
    else:
        return True

def continuous_riccati(P, A, B, C, S, R, Q):
    pass

def continuous_generalized_riccati(P, A, C, Cbar, L0):
    pass

# Single riccati iteration
def discrete_riccati(P, A, B, C, R, Q=None, S=None):
    if Q is None:
        Q = np.eye(B.shape[1])
    if S is None:
        S = np.zeros((B.shape[1], C.shape[0]))

    return A @ P @ A.T - (A @ P @ C.T + B @ S) @ np.linalg.pinv(R + C @ P @ C.T) @\
           (A @ P @ C.T + B @ S).T + B @ Q @ B.T

# Single riccati iteration for spectral factorization problem
def discrete_generalized_riccati(P, A, C, Cbar, L0):
    return A @ P @ A.T + (Cbar.T - A @ P @ C.T) @ np.linalg.pinv(L0 - C @ P @ C.T) @ (Cbar.T - A @ P @ C.T).T
        

TYPE_DICT = {'discrete_generalized':discrete_generalized_riccati}


# Find the minimum solution to the riccati equation via fixed point iteration
def riccati_solve(*args, type='discrete_generalized', tol=1e-3, Pinit=None, max_iter=20):

    if Pinit is None:
        Pinit = np.zeros(args[0].shape)

    err = np.inf
    iter_ = 0

    p = Pinit
    while err > tol and iter_ < max_iter:

        pp = TYPE_DICT[type](p, *args)
        err = np.linalg.norm(p - pp)
        p = pp
        iter_ += 1
    return p

def riccati_array(Psqrt, F, H, Q):

    # Measurement update
    M = scipy.linalg.sqrtm()

    # Time update
\


# CKMS Array algorithm (see chapter 13 of Kailath). Implementation currently only works for initial condition 0
# Alphabet Translation:
# Rsqrt -> sqrt of L0 - C P Ct
# K -> Kalman gain
# H -> C
# F -> A
# L -> square root of P_{i + 1} - P_i
def CKMS_Array(Re, K, L, F, H):

    Rsqrt = scipy.linalg.sqrtm(Re)
    M = np.block([[Rsqrt, H @ L], [K @ np.linalg.pinv(Rsqrt), F @ L]])


    # QR Factorize
    Q_, R_ = scipy.linalg.qr(M)

    # Select out the blocks of the appropriate sizes
    R_ = R_.T
    Rsqrt = R_[0:Rsqrt.shape[0], 0:Rsqrt.shape[1]]
    Kbar = R_[Rsqrt.shape[0]:, 0:K.shape[1]]
    L = R_[Rsqrt.shape[0]:, K.shape[1]:]

    Re = Rsqrt @ Rsqrt
    K = Kbar @ Rsqrt

    return Re, K, L
