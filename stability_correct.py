import numpy as np
import pdb
from subspaces import IteratedStableEstimator
from dstableFGM import dstable_descent
from dca.cov_util import form_lag_matrix

def check_stability(A):
    if max(np.abs(np.linalg.eigvals(A)) < 0.999):
        return True
    else:
        return False

def solve_qp(A, x0, x1, interp_iter=20):
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
    for i in range(interp_iter):    
        A_ = gamma * A1 + (1 - gamma) * A0
        if check_stability(A_):
            # Bring A_ closer to A0 (gamma -> 0)
            gamma = gamma - 0.5**(i + 1)
        else:
            # Bring A_ closer to A1 (gamma -> 1)
            gamma = gamma + 0.5**(i + 1)

    return A_


def stability_correct_VAR(A, order, data, correction_method='QP', **correction_kwargs):

    if max(np.abs(np.linalg.eigvals(A))) < 0.99:
        print('A already stable')
        return A

    if correction_method == 'QP':
        ydim = data.shape[1]
        yT = form_lag_matrix(data, order)
        y0 = yT[:, -ydim:]
        y1 = yT[:, :-ydim]
        A = solve_qp(A, y0, y1, **correction_kwargs)
    elif correction_method == 'proj':
        if 'n_init' in correction_kwargs.keys():
            n_init = correction_kwargs.pop('n_init')
        else:
            n_init = 20

        # Try n_init random initializations. Take the best result
        A_ = []
        losses = []
        for i in range(n_init):
            A, final_loss = dstable_descent(A, **correction_kwargs)
            losses.append(final_loss)
            A_.append(A)
        A = A_[np.argmin(losses)]

    return A
