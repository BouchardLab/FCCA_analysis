import numpy as np
import sys
import pickle

from scipy.stats import unitary_group, ortho_group

if __name__ == '__main__':

    seed = int(sys.argv[1])
    random = np.random.RandomState(seed)

    # Generate A matrices 
    dim = 100


    # 50 random U matrices. Then modulate the spread of the diagonals of P for each U

    U = []
    for i in range(50):
        U.append(unitary_group.rvs(100, random_state=seed))

    Puniform = []
    for j in range(50):
        Puniform.append(np.diag(np.linspace(0.775 - j/50 * 0.6, 0.8 + j/50 * 0.6, 100)))
    # for i in range(50):

    Pclustered = []
    for j in range(50):

        # Smaller cluster
        smaller_cluster = np.linspace(0.775 - j/50 * 0.6, 0.8 - j/50 * 0.5, dim//2)
        larger_cluster = np.linspace(0.775 + j/50 * 0.5, 0.8 + j/50 * 0.6)
        sigma = np.append(smaller_cluster, larger_cluster)

        # Larger cluster
        Pclustered.append(np.diag(sigma))

    # Check to make sure all eigenvalues are < 1
    A = []
    lambda_max = np.zeros((50, 50, 2))
    for i in range(len(U)):
        for j in range(len(Puniform)):

            A_ = U[i] @ Puniform[j]
            lambda_max[i, j, 0] = np.max(np.abs(np.linalg.eigvals(A_)))     
            A.append(A_)

            A_ = U[i] @ Pclustered[j]
            lambda_max[i, j, 1] = np.max(np.abs(np.linalg.eigvals(A_)))
            A.append(A_)     


    # Does the polar decomposition return U, P?

    assert(np.all(lambda_max < 1))

    # Modulate ranks of the 'B' matrices.
    B = []
    bdims = [2, 5, 10, 25, 50, 100]
    for k in range(50):
        BB = ortho_group.rvs(dim)
        for d in bdims:
            B.append(BB[:, 0:d])
        

    with open('LDS_db.dat', 'wb') as f:
        f.write(pickle.dumps(A))
        f.write(pickle.dumps(B))