from utils import Calc_I, functional, functional_irrational
from params import freqs, recoverable_params, recoverable_params_indexes,ParmLocal, Lparms, Rparms, NSteps, Nf
import matplotlib
matplotlib.rcParams.update({'font.size': 25})

RL_reference = Calc_I(freqs, recoverable_params, recoverable_params_indexes, ParmLocal, Lparms, Rparms, NSteps, Nf)
reference = RL_reference[:,5:].ravel()

import numpy as np
import matplotlib.pyplot as plt

N = 2**10
test_delta = 1
delta = np.linspace(-test_delta, test_delta, N)

for test_axis in range(5):
    testprs = np.zeros((N, recoverable_params.shape[0]))
    testprs[:] = recoverable_params
    testprs[:,test_axis] += delta * testprs[:,test_axis]
    
    II = np.zeros((N, reference.shape[0]))
    
    for i, prs in enumerate(testprs):
        II[i] = Calc_I(freqs, prs, recoverable_params_indexes, ParmLocal, Lparms, Rparms, NSteps, Nf)[:,5:].ravel()
    
    plt.figure(figsize=(16, 14))
    plt.title('С учетом поляризации в обычном функционале')
    plt.plot(delta, functional_irrational(II, reference), 'r', linewidth = 6)
    plt.xlabel(f'Относительный сдвиг {test_axis+1} параметра')
    plt.ylabel('Функционал иррациональный (красный)')
    plt.twinx()
    plt.plot(delta, functional(II, reference), 'b', linewidth = 6)
    plt.ylabel('Функционал обычный (синий)')
    plt.tight_layout()
    plt.pause(1)
    plt.savefig(f'test_functional{test_axis}.png')