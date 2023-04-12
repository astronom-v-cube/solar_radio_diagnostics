import os, shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np

from generatingModels import generatingModels
from params import (Lparms, Nf, NSteps, ParmLocal, Rparms, freqs,
                    indexes_of_recoverable_parameters, recoverable_parameters)
from utils import Calc_I, functional, functional_irrational

RL_reference = Calc_I(freqs, recoverable_parameters, indexes_of_recoverable_parameters, ParmLocal, Lparms, Rparms, NSteps, Nf)
# снижение размерности массива с тем же количеством элементов
reference = RL_reference[:,5:].ravel()

# значение интенсивности для каждой из точек
def func(variable_parameter):
    y = np.zeros((variable_parameter.shape[1], len(freqs)*2))
    for i in range(variable_parameter.shape[1]):
        y[i] = Calc_I(freqs, variable_parameter[:,i], indexes_of_recoverable_parameters, ParmLocal, Lparms, Rparms, NSteps, Nf)[:,5:].ravel()
    return y

# аналог func для одной точки и с возвратом индекса
def sub(variable_parameter, i):
    print(variable_parameter)
    return Calc_I(freqs, variable_parameter, indexes_of_recoverable_parameters, ParmLocal, Lparms, Rparms, NSteps, Nf)[:,5:].ravel(), i

def func_multythread(variable_parameter):
    y = np.zeros((variable_parameter.shape[1], len(freqs)*2))
    with ThreadPoolExecutor(max_workers=11) as executor:
        # переменная - список задач для многопотока
        futures = []
        for i in range(variable_parameter.shape[1]):
            futures.append(executor.submit(sub, variable_parameter[:,i], i))
        for future in as_completed(futures):
            ret, i = future.result()
            y[i] = ret
    return y

def minimizer(y):
    return functional_irrational(y, reference)
    # return functional(y, reference)

try:
    os.mkdir('dats')
except:
    shutil.rmtree('dats')
    os.mkdir('dats')
    

# привязка к количеству точек
n = len(indexes_of_recoverable_parameters)

gen = generatingModels(func_multythread, minimizer, dimensions = n, fname = 'dats/dat')

#чтобы не комментировать каждый раз лишние параметры
try:
    gen.x0[0]=1e11   # n_0 - тепловая электронная плотность, см^{-3}
    gen.x0[1]=700   # B - магнитное поле, G
    gen.x0[2]=250 #угол между В и лучом зрения
    gen.x0[3]=1e7  # n_b - нетепловая электронная плотность, см^{-3}
    gen.x0[4]=1.0 # \delta_1
except: pass

# ширина генерации``
gen.sigma = gen.x0/4

# сумарно будет насчитано points*nchildren*ngen
gen.Generating(ngenerations=25, nchildren=15, sigmacoeff=4, points=2**10, do_plot = True, recoverable_parameters = recoverable_parameters)
print(gen.x0)

# gen.plot(recoverable_parameters)
# plt.show()
