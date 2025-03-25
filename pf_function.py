from pypower.api import dSbus_dV, makeSbus, newtonpf, runpf
from numpy import array, angle, exp, linalg, conj, r_, Inf
from scipy.sparse import hstack, vstack
from pypower.idx_bus import *
from copy import deepcopy
import numpy as np


def extract_function(Ybus, Sbus, V, pv, pq):
    mis = V * conj(Ybus * V) - Sbus
    Fp = r_[ mis[pv].real,
             mis[pq].real]
    Fq = mis[pq].imag
    return Fp, Fq
    

def extract_jacobian(Ybus, V, pv, pq):
    pvpq = r_[pv, pq]
    dS_dVm, dS_dVa = dSbus_dV(Ybus, V)
    J11 = dS_dVa[array([pvpq]).T, pvpq].real
    J12 = dS_dVm[array([pvpq]).T, pq].real
    J21 = dS_dVa[array([pq]).T, pvpq].imag
    J22 = dS_dVm[array([pq]).T, pq].imag
    Jp, Jq = hstack([J11, J12], format="csr"), hstack([J21, J22], format="csr")
    return Jp, Jq

def x_to_V(bus_ppc, x, pv, pq):
    bus = deepcopy(bus_ppc)
    pvpq = r_[pv, pq]
    bus[pvpq, VA] = x[:len(pvpq)] * 180. / np.pi
    bus[pq, VM] = x[len(pvpq):]
    return bus, bus[:, VM] * np.exp(1j * np.pi / 180. * bus[:, VA])

def V_to_x(V, pv, pq):
    pvpq = r_[pv, pq]
    xa = angle(V[pvpq])
    xm = abs(V[pq])
    return r_[xa, xm]