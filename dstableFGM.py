from copy import deepcopy
import pdb
import torch
import numpy as np
from scipy.linalg import polar

def graddstableform(A, S, U, B, return_grad=False):

    # Throw away any imaginary components that leak in
    S = np.real(S)
    U = np.real(U)
    B = np.real(B)

    n = A.shape[0]
    At = torch.tensor(A, requires_grad=True)
    St = torch.tensor(S, requires_grad=True)
    Ut = torch.tensor(U, requires_grad=True)
    Bt = torch.tensor(B, requires_grad=True)

    loss = torch.pow(torch.norm(At - torch.chain_matmul(torch.inverse(St), Ut, Bt, St)), 2)

    # Gradient
    if return_grad:
        loss.backward()
        gradS = St.grad.detach().numpy()
        gradU = Ut.grad.detach().numpy()
        gradB = Bt.grad.detach().numpy()

        return loss.detach().numpy(), gradS, gradU, gradB
    else:
        return loss.detach().numpy()

def projectPSD(Q, epsilon=1e-6, delta=np.inf):

    Q = (Q + Q.T)/2
    e, V = np.linalg.eig(Q)
    Qp = np.diag(np.minimum(delta * np.ones(e.size), np.maximum(e, epsilon * np.ones(e.size))))
    Qp = V @ Qp @ V.T
    return Qp

# Just implement a simple projected gradient descent
def dstable_descent(A, maxiter=100, inneriter=20, step_reduc=1.5, tol=1e-3, posdef=1e-6, astab=1e-6, init='default'):
    n = A.shape[0]
    if not astab >= 0 and astab <= 1:
        raise ValueError('astab must be between - and 1')    
    
    if init == 'default':
        S = np.eye(n)
        U, B = polar(A)
        B = projectPSD(B, 0, 1 - astab)

    elif init == 'random':
        S = np.diag(np.random.uniform(1e-6, 1 - 1e-6, size=(n,)))
        U, B = polar(S @ A @ np.linalg.inv(S))
        B = projectPSD(B, 0, 1 - astab)
    
    if np.linalg.cond(S) > 1e12:
        print('The initial S is ill-conditioned')

    eS = np.linalg.eigvals(S)
    e = np.zeros(maxiter + 1)

    L=(max(eS) / min(eS)) ** 2
    e[0]=graddstableform(A,S,U,B)
    print(e[0])
    step=1 / L

    i = 0
    delta_e = np.inf
    while i < maxiter and delta_e > tol: 
        # Throw away any imaginary components that leak in
        S = np.real(S)
        U = np.real(U)
        B = np.real(B)

        _, gS, gU, gB = graddstableform(A, S, U, B, return_grad=True)

        j = 0
        e_ = []
        while j < inneriter:        
            Sn=S - gS * step
            Un=U - gU * step
            Bn=B - gB * step

            Sn=projectPSD(Sn, posdef)
            Un, _ = polar(Un)
            Bn=projectPSD(Bn,0,1 - astab)

            e_.append(graddstableform(A, Sn, Un, Bn))
            if e_[-1] < e[i]:
                break
            else:
                step /= step_reduc
                j += 1
        if j == inneriter:
            print('Descent failed')
            return np.linalg.inv(S) @ U @ B @ S
        else:
            i += 1

            e[i] = e_[-1]

            if i > 10:
                delta_e = np.mean(np.abs(np.diff(e))[i - 5:i])
            print(e[i])
            S = Sn
            U = Un
            B = Bn
            step *= 2

    return np.linalg.inv(S) @ U @ B @ S        


    

def dstableFGM(A=None, maxiter=int(1e3), posdef=1e-6, astab=1e-6, display=1,
               alpha0=0.5, lsparam=1.5, lsitermax=50, gradient=0, init='default'):

    n = A.shape[0]    
    nA2 = np.linalg.norm(A)**2
    
    if not astab >= 0 and astab <= 1:
        raise ValueError('astab must be between - and 1')    
    
    if init == 'default':
        S = np.eye(n)
        U, B = polar(A)
        B = projectPSD(B, 0, 1 - astab)

    elif init == 'random':
        S = np.diag(np.random.uniform(1e-6, 1 - 1e-6, size=(n,)))
        U, B = polar(S @ A @ np.linalg.inv(S))
        B = projectPSD(B, 0, 1 - astab)

    if not (alpha0 > 0 and alpha0 < 1):
        raise ValueError('alpha0 has to be in (0, 1).')

    if not lsparam > 1:
        raise ValueError('lsparam has to be larger than 1 for convergence.')


    if np.linalg.cond(S) > 1e12:
        print('The initial S is ill-conditioned')
    

    eS = np.linalg.eigvals(S)

    e = np.zeros(maxiter)
    alpha = np.zeros(maxiter)
    alpha[0] = alpha0
    beta = np.zeros(maxiter)

    L=(max(eS) / min(eS)) ** 2
    e[0]=graddstableform(A,S,U,B)
    print(e[0])
    step=1 / L
    i=0

    Ys = deepcopy(S)
    Yu = deepcopy(U)
    Yb = deepcopy(B)

    restarti=1

    if display:
        print('Display of iteration number and error:')
        if n < 10:
            ndisplay=1000
        elif n < 50:
            ndisplay=100
        elif n < 100:
            ndisplay=10
        elif n < 500:
            ndisplay = 5
        else:
            ndisplay = 1

    # Main loop
    while i < maxiter: 
        # Compute gradient
        __,gS,gU,gB=graddstableform(A,Ys,Yu,Yb,return_grad=True)
        e[i + 1] =+ np.inf
        inneriter=1

        e_ = []
        while e[i + 1] > e[i] and ((i == 0 and inneriter < 100) or inneriter < lsitermax):

            # For i == 1, we always have a descent direction
            Sn=Ys - gS * step
            Un=Yu - gU * step
            Bn=Yb - gB * step

            Sn=projectPSD(Sn, posdef)
            Un, _ = polar(Un)
            Bn=projectPSD(Bn,0,1 - astab)

            e_.append(graddstableform(A, Sn, Un, Bn))

            e[i + 1]= e_[-1]

            step=step / lsparam
            inneriter=inneriter + 1

        if i == 0:
            inneriter0=deepcopy(inneriter)

        # Conjugate with FGM weights, if decrease was achieved 
        # otherwise restart FGM
        alpha[i + 1]= (np.sqrt(alpha[i]**4 + 4 * alpha[i]**2) - alpha[i]**2)/2

        beta[i]= alpha[i] * (1 - alpha[i]) / (alpha[i] ** 2 + alpha[i + 1])
        if inneriter >= lsitermax:
            if restarti == 1:
                # Restart FGM if not a descent direction
                restarti=0
                alpha[i + 1]=alpha0
                Ys=deepcopy(S)
                Yu=deepcopy(U)
                Yb=deepcopy(B)
                e[i + 1]=e[i]
                if display:
                    print('Descent could not be achieved: restart. (Iteration %3.0f) \n',i)

                # Reinitialize step length
                eS=np.linalg.eigvals(S)
                L=(max(eS) / min(eS)) ** 2
                # step were necessary to obtain decrease
                step=1 / L / (lsparam ** inneriter0)
            else:
                if restarti == 0:
                    if display:
                        print('The algorithm has converged.')
                    e[i + 1]=e[i]
                    break
        else:
            restarti=1
            if gradient == 1:
                beta[i]=0

            Ys=Sn + (beta[i] * (Sn - S))
            Yu=Un + (beta[i] *(Un - U))
            Yb=Bn + (beta[i] * (Bn - B))
            S=deepcopy(Sn)
            U=deepcopy(Un)
            B=deepcopy(Bn)
            
        A_ = np.linalg.inv(Ys) @ Yu @ Yb @ Ys
        loss = np.linalg.norm(A - A_)

        # Display
        if display:
            if i % ndisplay == 0:
                print('%2.0f:%2.3f - ',i ,e[i + 1])
            if i % (ndisplay * 10) == 0:
                print('\n')

        if e[i] < (1e-12 * nA2) or (i > 100 and (e[i - 100] - e[i]) < 1e-06 * e[i - 100]):
            pdb.set_trace()
            if display:
                print('The algorithm has converged.')
            break

        step *= 2

    return np.linalg.inv(S) @ U @ B @ S
