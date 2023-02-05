#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  6 22:01:45 2022

@author: edombek
"""

from utils import Calc_I, functional_irrational
from params import freqs, refprs, prs_indexes, ParmLocal, Lparms, Rparms, NSteps, Nf
import numpy as np
import matplotlib.pyplot as plt

RL_reference = Calc_I(freqs, refprs, prs_indexes, ParmLocal, Lparms, Rparms, NSteps, Nf)
reference = RL_reference[:,5:].ravel()

prs = refprs
d=0.99
n=16
dp = np.linspace(-d,d,n)
im = np.zeros((n,n))

fig,axs = plt.subplots(len(prs), len(prs), sharex=True, sharey=True, figsize = (10,10))

for j in range(len(prs)):
    for i in range(len(prs)):
        prs = refprs.copy()
        #if i <= j or i == j: continue
        for xi, x in enumerate(dp):
            for yi, y in enumerate(dp):
                prs[i] = x * refprs[i] + refprs[i]
                prs[j] = y * refprs[j] + refprs[j]
                RL = Calc_I(freqs, prs, prs_indexes, ParmLocal, Lparms, Rparms, NSteps, Nf)
                im[xi,yi] = functional_irrational(RL[:,5:].ravel()[None,:], reference)
        axs[i,j].imshow(im, origin='lower', extent=(-d,d,-d,d))
        plt.pause(1)
plt.tight_layout()