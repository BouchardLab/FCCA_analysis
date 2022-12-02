import numpy as np 
import scipy
import pdb

# Calculate loadings 
# U: matrix of shape (n_neurons * order, dim)
# d: order of model
def calc_loadings(U, d=1):
    # Sum over components
    U = np.sum(np.power(np.abs(U), 2), axis=-1)

    # Reshape and then sum over neurons
    U = np.reshape(U, (d, -1))
    loadings = np.sum(U, axis=0)
    loadings /= np.max(loadings)
    return loadings

# Calculate loadings when 2 rectangular matrices are cascaded 
# (e.g. dimreduc + decoding)
# U1: matrix of shape (n_neurons * d1, dim)
# U2: matrix of shape (dim * d2, dim2)
def calc_cascaded_loadings(U1, U2, d1=1, d2=1):

    # Simply multiply the matrices and then calculate the leverage score. 
    # Only sensible case if is d2 > dim, in which case we reshape and multiply
    # several times
    if d2 * U2.shape[0] > U1.shape[1]:
        U2 = np.reshape(U2, (d2, U1.shape[1], -1))
        V = np.array([U1 @ U2_ for U2_ in U2])

        # Sum over final index, and reshaping of d2//d1
        U = np.sum(np.sum(np.power(np.abs(V), 2), axis=-1), axis=0)
        # Sum over d1
        U = np.reshape(U, (d1, -1))
        loadings = np.sum(U, axis=0)
        loadings /= np.max(loadings)
        return loadings
    elif d2 * U2.shape[0] == U1.shape[1]:
        return calc_loadings(U1 @ U2, d=d1)
    else:
        raise ValueError("Don't know what to do with this case")

def filter_by_dict(df, root_key, dict_filter):

    col = df[root_key].values

    filtered_idxs = []

    for i, c in enumerate(col):
        match = True
        for key, val in dict_filter.items():
            if c[key] != val:
                match = False
        if match:
            filtered_idxs.append(i)

    return df.iloc[filtered_idxs]

# Shortcut to apply multiple filters to pandas dataframe
def apply_df_filters(dtfrm, invert=False, reset_index=True, **kwargs):

    filtered_df = dtfrm

    for key, value in kwargs.items():

        # If the value is the dict
        if type(value) == dict:

            filtered_df = filter_by_dict(filtered_df, key, value)

        else:
            if type(value) == list:
                matching_idxs = []
                for v in value:
                    df_ = apply_df_filters(filtered_df, reset_index=False, **{key:v})
                    if invert:
                        matching_idxs.extend(list(np.setdiff1d(np.arange(filtered_df.shape[0]), list(df_.index))))
                    else:
                        matchings_idxs = matching_idxs.extend(list(df_.index))

                filtered_df = filtered_df.iloc[matching_idxs]
        
            elif type(value) == str:
                filtered_df = filtered_df.loc[[value in s for s in filtered_df[key].values]]
            else:
                if invert:
                    filtered_df = filtered_df.loc[filtered_df[key] != value]
                else:
                    filtered_df = filtered_df.loc[filtered_df[key] == value]
        
        if reset_index:
            filtered_df.reset_index(inplace=True, drop=True)

        # if filtered_df.shape[0] == 0:
        #     print('Key %s reduced size to 0!' % key)

    return filtered_df


def gram_schmidt(v0):
    """
    Gives a orthonormal matrix, using modified Gram Schmidt Procedure
    :param A: a matrix of column vectors
    :return: a matrix of orthonormal column vectors
    """
    # assuming A is a square matrix
    dim = v0.size
    Q = np.zeros((dim, dim), dtype=v0.dtype)
    Q[:, 0] = v0/np.linalg.norm(v0, ord=2)
    for j in range(1, dim):
        # Generate vector with random real and imaginary components
        q = np.random.normal(size=dim) + 1j * np.random.normal(size=dim)
        for i in range(0, j):
            rij = np.vdot(Q[:,i], q)
            q = q - rij*Q[:,i]
        rjj = np.linalg.norm(q, ord=2)
        if np.isclose(rjj,0.0):
            raise ValueError("invalid input matrix")
        else:
            Q[:,j] = q/rjj
                
    return Q


# Assumes eigenvectors are all distinct
# Allows one to calculate the schur decomposition with one's own prescribed eigenvalue ordering
def schur(A, eig_order):
    
    eigvalA, eigvecA = np.linalg.eig(A)
    
    AN = A
        
    # Chain of U matrices
    U = []
    S = []    
    for i, eig_idx in enumerate(eig_order):
        eigvals, eigvecs = np.linalg.eig(AN)
        
        # Find the eigenvector in the reduced matrix corresponding to the eigenvalue of choice
        try:
            eig_idxN = np.where(np.isclose(np.real(eigvals), np.real(eigvalA[eig_idx])))[0][0]
        except:
            pdb.set_trace()
            
        b1 = eigvecs[:, eig_idxN]

        # Form an orthonormal basis for the null space
        try:
            bn = scipy.linalg.orth(scipy.linalg.null_space(b1[np.newaxis, :]))
            V = np.append(b1[:, np.newaxis], bn, axis=1)
        except:
            V = gram_schmidt(b1)
                    
        # Needs to be a unitary matrix
        try:
            assert(np.allclose(np.real(np.conjugate(V).T), np.real(np.linalg.inv(V))))
            assert(np.allclose(np.imag(np.conjugate(V).T), np.imag(np.linalg.inv(V))))
        except:
            V = gram_schmidt(b1)
            assert(np.allclose(np.real(np.conjugate(V).T), np.real(np.linalg.inv(V))))
            assert(np.allclose(np.imag(np.conjugate(V).T), np.imag(np.linalg.inv(V))))
        
        U.append(V)
        
        # Transform the submatrix and extract its lower diagonal block
        B = np.conjugate(V).T @ AN @ V

        S.append(B)
        AN = B[1:, 1:]
        
    # Chain together the transformation matrices to obtain the final transformation
    Q = U[0]
    
    n = A.shape[0]

    for i in range(1, len(U)):
        # Append the necessary size identity block to the transformations
        UU = np.block([[np.eye(i), np.zeros((i, n - i))], [np.zeros((n - i, i)), U[i]]])
        Q = Q @ UU
    
    # Extract the upper triangular blocks from the transformed matrices
    T = np.zeros((n, n))
    for i in range(len(S)):
        
        # Append the necessary number of zeros
        if i > 0:
            row = np.append(np.zeros(i), S[i][0, :])
        else:
            row = S[i][0, :]
        
        T[i, :] = row
        
    return T, Q
