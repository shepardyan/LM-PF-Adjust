from pypower.api import case300
from pypower.api import runpf
from pypower.idx_bus import *
from pypower.idx_gen import *
from lm_adjust import lm_adjust
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    ppc = case300()

    # Increase Load and generation
    ppc['bus'][:, PD] *= 1.5
    ppc['bus'][:, QD] *= 1.5

    ppc['gen'][:, PG] *= 1.5
    ppc['gen'][:, QG] *= 1.5

    # ppc['bus'][:, VA] = 0.0
    # ppc['bus'][:, VM] = 1.0

    x, Fp, Fq, Fq_current = lm_adjust(ppc, verbose=False)
    plt.plot(Fp, label="Active Subproblem")
    plt.plot(Fq, label="Reactive Subproblem")
    plt.yscale('log')
    plt.legend()
    plt.xlabel("# Iter")
    plt.ylabel("2-Norm of Residual")
    plt.savefig("residual.png", dpi=600)
    plt.close()

    sorted_Fq_index = list(np.argsort(Fq_current)[-11:])
    sorted_Fq_index.reverse()
    bus_index = ppc['bus'][sorted_Fq_index, BUS_I].astype(int)
    bars = plt.bar(range(len(sorted_Fq_index)), Fq_current[sorted_Fq_index])
    plt.xticks(range(len(sorted_Fq_index)), labels=bus_index)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    plt.tight_layout()
    plt.xlabel("# Bus")
    plt.ylabel("2-Norm of Residual")
    plt.savefig("reactive.png", dpi=600)
    plt.close()



