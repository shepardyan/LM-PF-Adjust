from scipy.sparse import vstack, hstack, eye, csr_matrix
from scipy.sparse.linalg import spsolve, norm, lsmr, lgmres
from scipy.optimize import minimize
from scipy.linalg import solve
import numpy as np
from scipy.io import savemat
import matlab.engine
# 启动MATLAB引擎
eng = matlab.engine.start_matlab()

def solve_reactive_step(Jp, Jq, gamma, delta, Fq, pk):
    A = vstack([
        hstack([Jq.T @ Jq + gamma * eye(Jq.shape[1]), Jp.T]),
        hstack([Jp, -delta * eye(Jp.shape[0])])
    ], format='csr')
    rhs_partial = (Jq.T @ (Fq + Jq @ pk) + gamma * pk).reshape(-1, 1)
    b = vstack([
        rhs_partial,
        csr_matrix((A.shape[0] - rhs_partial.shape[0], 1))
    ], format='csr')
    # A_matlab = matlab.double(A.toarray().tolist())
    # b_matlab = matlab.double((-b.toarray()).tolist())

    # X = eng.hsl_ma86_densewrapper(A_matlab, b_matlab)
    # res = np.array(list(map(np.float64, X))).flatten()
    res = spsolve(A, -b)
    q = res[:pk.shape[0]]
    return q