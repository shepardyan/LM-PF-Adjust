import sparseqr
import numpy as np
from scipy.sparse import vstack, eye, csr_matrix, csr_array

def solve_active_step(Jp, gamma, Fp):
    # Setup LHS term
    A = vstack([Jp, np.sqrt(gamma) * eye(Jp.shape[1])], format='coo')
    # Setup RHS term
    b = vstack([Fp.reshape(-1, 1), csr_matrix((A.shape[0] - Fp.shape[0], 1), dtype=Fp.dtype)])
    # solve
    p = -sparseqr.solve(A, b, tolerance=0)
    return p.toarray().flatten()