#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 16:47:25 2022

@author: edombek
"""

from utils import Calc_I, functional, functional_irrational
from params import freqs, refprs, prs_indexes, ParmLocal, Lparms, Rparms, NSteps, Nf

RL_reference = Calc_I(freqs, refprs, prs_indexes, ParmLocal, Lparms, Rparms, NSteps, Nf)
reference = RL_reference[:,5:].ravel()

import numpy as np
import matplotlib.pyplot as plt

N = 2**10
test_delta = 0.5
delta = np.linspace(-test_delta, test_delta, N)

for test_axis in range(5):
    testprs = np.zeros((N, refprs.shape[0]))
    testprs[:] = refprs
    testprs[:,test_axis] += delta * testprs[:,test_axis]
    
    II = np.zeros((N, reference.shape[0]))
    
    for i, prs in enumerate(testprs):
        II[i] = Calc_I(freqs, prs, prs_indexes, ParmLocal, Lparms, Rparms, NSteps, Nf)[:,5:].ravel()
    
    plt.figure(figsize=(8,6))
    plt.plot(delta, functional_irrational(II, reference), 'r')
    plt.xlabel(f'Относительный сдвиг {test_axis} параметра')
    plt.ylabel('Функционал иррациональный (красный)')
    plt.twinx()
    plt.plot(delta, functional(II, reference), 'b')
    plt.ylabel('Функционал в интенсивностях и параметрах стокса (синий)')
    plt.tight_layout()
    plt.pause(1)
    plt.savefig(f'test_functional{test_axis}.png')