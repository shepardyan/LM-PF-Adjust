from scipy.sparse.linalg import spsolve, norm
from scipy.sparse import vstack
import numpy as np

def get_lagrange_multiplier(Jp, Jq, Fq):
    return -spsolve(Jp @ Jp.T, Jp @ Jq.T @ Fq)

def get_lagrange_derivative(lambda_k, Fp, Fq, Jp, Jq):
    deri = vstack([(Jq.T @ Fq.reshape(-1) + Jp.T @ lambda_k.reshape(-1)).reshape(-1, 1), Fp.reshape(-1, 1)])
    return deri
