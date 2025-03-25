from active_subproblem import solve_active_step
from reactive_subproblem import solve_reactive_step
from lagrange import get_lagrange_multiplier, get_lagrange_derivative
from scipy.sparse.linalg import norm, spsolve
from scipy.sparse import eye
from pf_function import *
from pypower.api import makeYbus, makeSbus, bustypes, ext2int
from pypower.pfsoln import pfsoln
from pypower.idx_bus import *
import numpy as np
from copy import deepcopy

def lm_adjust(ppc: dict, k_max=10000, epsilon=1e-2, verbose=True, options:dict=None):
    # extract options
    if options is None:
        options = {
            "sigma0": 1e-4,
            "sigma1": 0.5,
            "sigma2": 0.9,
            "C": 10.0,
            "alpha_max": 1e8,
            "alpha_0": 1e-4,
            "beta_min": 1e-8,
            "beta_0": 0.1,
            "r0": 1.0,
            "rho_epsilon": 1e-6
        }
    sigma0 = options["sigma0"]
    sigma1 = options["sigma1"]
    sigma2 = options["sigma2"]
    C = options["C"]
    alpha_max = options['alpha_max']
    alpha_0 = options['alpha_0']
    beta_min = options['beta_min']
    beta_0 = options['beta_0']
    r0 = options['r0']
    rho_epsilon = options['rho_epsilon']

    # result_record
    Fp_array = np.zeros(k_max)
    Fq_array = np.zeros(k_max)

    # parameters
    ppc = ext2int(ppc)
    baseMVA, bus, gen, branch = ppc['baseMVA'], deepcopy(ppc['bus']), deepcopy(ppc['gen']), deepcopy(ppc['branch'])
    ref, pv, pq = bustypes(bus, gen)
    Ybus, Yf, Yt = makeYbus(baseMVA, bus, branch)

    # initialization
    V  = bus[:, VM] * np.exp(1j * np.pi / 180. * bus[:, VA])
    k = 0
    xk = V_to_x(V, pv, pq)
    alpha_k = alpha_0
    beta_k = beta_0

    while k < k_max:  # alg.step 3
        bus, V = x_to_V(bus, xk, pv, pq)
        Sbus = makeSbus(baseMVA, bus, gen)
        Fp, Fq = extract_function(Ybus, Sbus, V, pv, pq)
        Jp, Jq = extract_jacobian(Ybus, V, pv, pq)

        Fp_array[k] = np.linalg.norm(Fp)
        Fq_array[k] = np.linalg.norm(Fq)

        # alg.step 4
        lambda_k = get_lagrange_multiplier(Jp, Jq, Fq)

        # alg.step 5
        deri = get_lagrange_derivative(lambda_k, Fp, Fq, Jp, Jq)
        norm_deri = np.linalg.norm(deri.toarray().flatten(), ord=2)
        delta_k = alpha_k * norm_deri 
        gamma_k = beta_k * norm_deri
        if verbose:
            print(f"Step {k} norm {norm_deri}, delta {delta_k} gamma {gamma_k} \n |- active {np.linalg.norm(Fp, ord=2)}, reactive {np.linalg.norm(Fq, ord=2)}", end=" ")
        if norm_deri  < epsilon:
            return xk, Fp_array[:k], Fq_array[:k], Fq
        
        # alg.step 6
        pk = solve_active_step(Jp, gamma_k, Fp)

        # alg.step 7
        qk = solve_reactive_step(Jp, Jq, gamma_k, delta_k, Fq, pk)
        
        # alg.step 8
        sk = pk + qk
        pkc = np.linalg.norm(Fp, ord=2) ** 2 - np.linalg.norm(Fp + Jp @ sk, ord=2) ** 2
        nkc = np.linalg.norm(Fp, ord=2) ** 2 - np.linalg.norm(Fp + Jp @ pk, ord=2) ** 2

        # alg.step 9
        while pkc < sigma0 * nkc:
            if verbose:
                print(f"\n |----- qk {np.linalg.norm(qk, ord=2)}, constraint {np.linalg.norm(Jp @ qk, ord=2)}, pkc {pkc}, nkc {nkc}", end="")
            # alg.step 10
            alpha_k /= C
            delta_k = alpha_k * norm_deri 
            # alg.step 11
            qk = solve_reactive_step(Jp, Jq, gamma_k, delta_k, Fq, pk)  
            # alg.step 12
            sk = pk + qk
            pkc = np.linalg.norm(Fp, ord=2) ** 2 - np.linalg.norm(Fp + Jp @ sk, ord=2) ** 2
            nkc = np.linalg.norm(Fp, ord=2) ** 2 - np.linalg.norm(Fp + Jp @ pk, ord=2) ** 2

        # alg.step 13
        if pkc < sigma1 * nkc:
            alpha_k /= C 
        elif pkc > sigma2 * nkc:
            alpha_k = min(C * alpha_k, alpha_max) 

        # alg.step 14
        xk_hat = xk + sk
        _, V_hat = x_to_V(bus, xk_hat, pv, pq)
        Fp_hat, Fq_hat = extract_function(Ybus, Sbus, V_hat, pv, pq)
        
        akl = (0.5 * np.linalg.norm(Fq, ord=2) ** 2 + lambda_k.T @ Fp - 0.5 * np.linalg.norm(Fq_hat, ord=2) ** 2 - lambda_k.T @ Fp_hat).item()  # ()
        pkl = (0.5 * np.linalg.norm(Fq, ord=2) ** 2 + lambda_k.T @ Fp - 0.5 * np.linalg.norm(Fq + Jq @ sk, ord=2) ** 2 - lambda_k.T @ (Fp + Jp @ sk)).item() 
        akc = np.linalg.norm(Fp, ord=2) ** 2 - np.linalg.norm(Fp_hat, ord=2) ** 2
        pkc = np.linalg.norm(Fp, ord=2) ** 2 - np.linalg.norm(Fp + Jp @ sk, ord=2) ** 2
        ak_func = lambda rk: akl + rk * akc 
        pk_func = lambda rk: pkl + rk * pkc
        # alg.step 15
        if pk_func(r0) >= 0.5 * r0 * pkc:
            rk = r0 
        else:
            rk = -2 * pk_func(r0) / pkc + 2 * r0 + rho_epsilon

        ak = ak_func(rk)
        pk = pk_func(rk)

        # alg.step 16
        if ak < sigma1 * pk: 
            beta_k *= C 
        elif ak > sigma2 * pk:
            beta_k = max(beta_k / C, beta_min)
        
        # alg.step 17
        if ak > sigma0 * pk:
            xk += sk
            if verbose:
                print(f"\n|-Update norm {np.linalg.norm(sk, ord=2)}\n")
        else:
            if verbose:
                print("\n|-Update norm 0.0\n")
        k += 1
        # bus, gen, branch = pfsoln(baseMVA, bus, gen, branch, Ybus, Yf, Yt, V, ref, pv, pq)

    return xk, Fp_array[:k], Fq_array[:k], Fq

