import numpy as np
import scipy 
from scipy import optimize
from scipy.stats import ortho_group

import pdb
from tqdm import tqdm

# Check for existence of solution by calculating eigenvalues of the matrix pencil
# (see eq 52 in Van Dooren A Generalized Eigenvalue Approach for Solving Riccati Equations)
def check_gdare(A, C, Cbar, L0, tol=1e-3):
    
    AA = np.block([[A - Cbar.T @ np.linalg.inv(L0) @ C, np.zeros(A.shape)],
                   [-C.T @ np.linalg.inv(L0) @ C, np.eye(A.shape[0])]])
    
    BB = np.block([[np.eye(A.shape[0]), -Cbar.T @ np.linalg.inv(L0) @ Cbar], 
                  [np.zeros(A.shape), A.T - C.T @ np.linalg.inv(L0) @ Cbar]])
    
    eig = scipy.linalg.eig(AA, b=BB, left=False, right=False)

    if np.any(np.bitwise_and(np.abs(eig) > 1 - tol, np.abs(eig) < 1 + tol)):
        return False
    else:
        return True
# Solve gdare using QZ decomposition
def solve_gdare(A, C, Cbar, L0):

    assert(check_gdare(A, C, Cbar, L0, tol=1e-4))

    AA = np.block([[A - Cbar.T @ np.linalg.inv(L0) @ C, np.zeros(A.shape)],
                   [-C.T @ np.linalg.inv(L0) @ C, np.eye(A.shape[0])]])

    BB = np.block([[np.eye(A.shape[0]), -Cbar.T @ np.linalg.inv(L0) @ Cbar], 
                  [np.zeros(A.shape), A.T - C.T @ np.linalg.inv(L0) @ Cbar]])

    AA, BB, alpha, beta, Q, Z = scipy.linalg.ordqz(AA, BB, sort='iuc')

    geig = np.divide(alpha, beta)
    iuc = np.where(np.abs(geig) < 1)
    # pdb.set_trace
    # assert(len(iuc) == AA.shape[0]/2)

    return AA, BB, alpha, beta, Q, Z


def continuous_riccati(P, A, B, C, S, R, Q):
    pass

def continuous_generalized_riccati(P, A, C, Cbar, L0):
    pass

# Single riccati iteration
def discrete_riccati(P, A, C, Q, R, G=None, S=None):
    if G is None:
        G = np.eye(A.shape[0])
    if S is None:
        S = np.zeros((G.shape[1], C.shape[0]))

    return A @ P @ A.T - (A @ P @ C.T + G @ S) @ np.linalg.pinv(R + C @ P @ C.T) @\
           (A @ P @ C.T + G @ S).T + G @ Q @ G.T

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

def sqrt_riccati(Psqrt, F, H, Qsqrt, Rsqrt):

    # Other form 
    pre = np.block([[Rsqrt, H @ Psqrt, np.zeros((H.shape[0], Qsqrt.shape[1]))],
                    [np.zeros((F.shape[0], Rsqrt.shape[1])), F @ Psqrt, Qsqrt]]).T
    _, post = scipy.linalg.qr(pre)
    post = post.T

    Psqrt = post[Rsqrt.shape[0]:, Rsqrt.shape[1]:Rsqrt.shape[1] + Psqrt.shape[1]]
    return Psqrt 

    # # Measurement update
    # # Pre-array
    # pre = np.block([[Rsqrt, H @ Psqrt], [np.zeros((Psqrt.shape[0], Rsqrt.shape[1])), Psqrt]]).T
    # _, post = scipy.linalg.qr(pre)
    # # Post-arry
    # post = post.T
    # #post_ = pre.T @ Q_

    # # Reisqrt = post[0:Rsqrt.shape[0], 0:Rsqrt.shape[1]]
    # # Kbar = post[Rsqrt.shape[0]:, 0:Rsqrt.shape[1]]
    # Piisqrt = post[Rsqrt.shape[0]:, Rsqrt.shape[1]:]

    # # Time update

    # pre = np.block([[F @ Piisqrt, Qsqrt]]).T
    # _, post = scipy.linalg.qr(pre)
    # post = post.T
    # Pi1sqrt = post[0:Psqrt.shape[0], 0:Psqrt.shape[1]]
    # return Pi1sqrt

# CKMS Array algorithm (see chapter 13 of Kailath). Implementation currently only works for initial condition 0
# Alphabet Translation:
# Rsqrt -> sqrt of L0 - C P Ct
# K -> Kalman gain
# H -> C
# F -> A
# L -> square root of P_{i + 1} - P_i
def CKMS_Array(Rsqrt, K, L, F, H):

    pre = np.block([[Rsqrt, H @ L], [K @ np.linalg.pinv(Rsqrt).T, F @ L]]).T

    # QR Factorize
    _, post = scipy.linalg.qr(pre)

    # Select out the blocks of the appropriate sizes
    post = post.T
    Rsqrt = post[0:Rsqrt.shape[0], 0:Rsqrt.shape[1]]
    Kbar = post[Rsqrt.shape[0]:, 0:K.shape[1]]
    L = post[Rsqrt.shape[0]:, K.shape[1]:]

    K = Kbar @ Rsqrt.T

    return Rsqrt, K, L

def CKMS(A, C, X, Xbar, R, Rbar):
    X = A @ X - Xbar @ np.linalg.pinv(Rbar) @ Xbar.T @ C.T
    Xbar = Xbar - A @ X @ np.linalg.pinv(R) @ X.T @ C.T
    R = R - C @ Xbar @ np.linalg.pinv(Rbar) @ Xbar.T @ C.T
    Rbar = Rbar - Xbar.T @ C.T @ np.linalg.pinv(R) @ C @ Xbar
    return X, Xbar, R, Rbar
