import numpy as np
import scipy
from copy import deepcopy
import pdb

# Apparently not needed
def calc_spectral_absicca(A, eps):
    pass

# See pg. 54 of matrix calculus book and the wikipedia page https://en.wikipedia.org/wiki/Commutation_matrix
def comm_mat(m,n):
    # determine permutation applied by K
    w = np.arange(m*n).reshape((m,n),order='F').T.ravel(order='F')

    # apply this permutation to the rows (i.e. to each column) of identity matrix and return result
    return np.eye(m*n)[w,:]

def gen_init_W(M, p, gamma, R, diag=0):

    Ainit = np.zeros((2 * M, 2 * M))

    w = R/np.sqrt(p * (1 - p) * (1 + gamma**2)/2)

    # Excitatory
    for j in range(M):
        for k in range(2 * M):
            if np.random.binomial(1, p):
                Ainit[j, k] = w/np.sqrt(2 * M)


    # Inhibitory
    for j in range(M):
        for k in range(2 * M):
            if np.random.binomial(1,p):
                Ainit[j + M, k] = -gamma * w/np.sqrt(2 * M)

    # Setting diagonals to 0 initially
    np.fill_diagonal(Ainit, diag)

    # Symmetrize within the E/I blocks
    
    # Ainit[0:M, 0:M] = (1/(2 - asymm)) * (Ainit[0:M, 0:M] + (1 - asymm) * Ainit[0:M, 0:M].T)
    # Ainit[M:2*M, M:2*M] = (1/(2 - asymm)) * (Ainit[M:2*M, M:2*M] + (1 - asymm) * Ainit[M:2*M, M:2*M].T)

    return Ainit


def gradient_checks():
    s = 0.1
    A = gen_init_W(100, 0.25, 3, 1, diag=-1)

    # Check whether for U = W = I, the gradient from the direct method matches the adjoint method
    P = scipy.linalg.solve_continuous_lyapunov(A - s * np.eye(A.shape[0]), -np.eye(A.shape[0]))
    Q = scipy.linalg.solve_continuous_lyapunov((A - s * np.eye(A.shape[0])).T, -np.eye(A.shape[0]))
    dfds1 = np.trace(scipy.linalg.solve_continuous_lyapunov(A - s * np.eye(A.shape[0]), 2 * P))
    dfds2 = -2 * np.trace(Q @ P)
    print(dfds1)
    print(dfds2)

    # Discrete time version
    A = gen_init_W(100, 0.25, 3, 1)
    P = scipy.linalg.solve_discrete_lyapunov(1/s * A, np.eye(A.shape[0]))
    Q = scipy.linalg.solve_discrete_lyapunov(1/s * A.T, np.eye(A.shape[0]))

    dfds1 = np.trace(scipy.linalg.solve_discrete_lyapunov(1/s * A, -2/s**3 * A @ P @ A.T))
    # Derive from adjoint method
    #dfds2 = 2/s**3 * Q.flatten() @ (np.kron(A, A)) @ P.flatten()
    print(dfds1)
    #print(dfds2)

def stabilize(A, max_iter=1000, eta=10):

    # See the supplementary note in the Hennequin paper. Don't actually have to calculate the spectral absicca, but rather just
    # keep a safe margin from it
    C = 1.5
    B = 0.2

    alpha = np.max(np.real(np.linalg.eigvals(A)))
    if alpha < 0:
        return A

    iter_ = 0

    while alpha > 0 and iter_ < max_iter:

        alpha_e = max(C * alpha, C * alpha + B)
        Q = scipy.linalg.solve_continuous_lyapunov((A - alpha_e * np.eye(A.shape[0])).T, -2 * np.eye(A.shape[0]))   
        P = scipy.linalg.solve_continuous_lyapunov(A - alpha_e * np.eye(A.shape[0]), -2 * np.eye(A.shape[0]))

        grad = Q @ P/np.trace(Q @ P)

        # Adjust inhibitory weights
        inh_idx = np.argwhere(A < 0)
        for idx in inh_idx:
            A[idx[0], idx[1]] -= eta * grad[idx[0], idx[1]]
            # Make sure no inhibitory weights got turned into excitatory weights
            if A[idx[0], idx[1]] > 0:
                A[idx[0], idx[1]] = 0
        
        alpha = np.max(np.real(np.linalg.eigvals(A)))
        iter_ += 1
    
    return A

def stabilize_discrete(A, max_iter=1000, C=1.01):
    # # See the supplementary note in the Hennequin paper. Don't actually have to calculate the spectral absicca, but rather just
    # # keep a safe margin from it

    n = A.shape[0]

    alpha = np.max(np.abs(np.linalg.eigvals(A)))
    if alpha < 1:
        return A

    iter_ = 0
    print('iter_:%d, alpha:%f' % (iter_, alpha) )    
    while alpha > 1 and iter_ < max_iter:

        # alpha_e = max(C * alpha, C * alpha + B)
        alpha_e = C * alpha

        # Need the gradient of the relaxed H2 norm with respect to A. To calculate this, we
        # first perform a bilinear transformation to solve a continuous time version of the Lyapunov
        # equation.

        P = scipy.linalg.solve_discrete_lyapunov(A/alpha_e, np.eye(n))
        K = comm_mat(n, n)

        # Note in the last term, we include a factor of alpha_e squared since we also pulled it out of the dA term
        jac = np.linalg.inv(np.eye(n**2)- np.kron(A/alpha_e, A/alpha_e)) @ (np.kron(A/alpha_e**2 @ P, np.eye(n)) + np.kron(np.eye(n), A @ P/alpha_e**2) @ K)

        grad = (np.eye(n).flatten().T @ jac).reshape((n, n), order='F')
        # Adjust inhibitory weights
        inh_idx = np.argwhere(A < 0)

        # Take the largest step size that still decreases the trace
        eta = np.logspace(-8, -3, 20)
        tr = np.zeros(eta.size)
        for i in range(eta.size):
            Ap = np.zeros(A.shape)
            for idx in inh_idx:
                Ap[idx[0], idx[1]] = A[idx[0], idx[1]] - eta[i] * grad[idx[0], idx[1]]

            tr[i] = np.trace(scipy.linalg.solve_discrete_lyapunov(Ap/alpha_e, np.eye(n)))        

        eta = eta[np.argmin(tr)]

        for idx in inh_idx:
            A[idx[0], idx[1]] -= eta * grad[idx[0], idx[1]]
            # Make sure no inhibitory weights got turned into excitatory weights
            if A[idx[0], idx[1]] > 0:
                A[idx[0], idx[1]] = 0
        
        alpha = np.max(np.abs(np.linalg.eigvals(A)))
        iter_ += 1
        print('iter_:%d, alpha:%f' % (iter_, alpha) )    


    return A

def gradient_check2():
    #A = gen_init_W(10, 0.9, 3, 1)
    A = np.random.normal(size=(10, 10))
    alpha = np.max(np.abs(np.linalg.eigvals(A)))
    alpha_e = 1.1 * alpha
    assert(np.max(np.abs(np.linalg.eigvals(A/alpha_e))) < 1)

    # Test our expression for the gradients of P with respect to A
    # Do this by looking at linear response
    n = A.shape[0]

    P = scipy.linalg.solve_discrete_lyapunov(A/alpha_e, np.eye(n))

    K = comm_mat(n, n)
    jac = np.linalg.inv(np.eye(n**2)- np.kron(A/alpha_e, A/alpha_e)) @ (np.kron(A/alpha_e**2 @ P, np.eye(n)) + np.kron(np.eye(n), A @ P/alpha_e**2) @ K)

    # For a function that maps nxq -> mxp, the Jacobian has the shape mp x nq, which in our case is just n^2 x n^2
    dtrpdA = (np.eye(n).flatten().T @ jac).reshape((n, n), order='F')
    print(dtrpdA)

    # Empirical test
    dtrpdA2 = np.zeros(A.shape)
    eps = 1e-6
    for i in range(A.shape[0]):
        for j in range(A.shape[0]):
            eij = np.zeros(A.shape)
            eij[i, j] = 1
            delta_A = A/alpha_e + eps * eij 
            Pdelta = scipy.linalg.solve_discrete_lyapunov(delta_A, np.eye(n))    
            dtrpdA2[i, j] = (np.trace(Pdelta) - np.trace(P))/eps
    print(dtrpdA2)

