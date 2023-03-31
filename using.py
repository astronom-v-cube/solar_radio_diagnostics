from utils import Calc_I, functional, functional_irrational
from params import freqs, recoverable_params, recoverable_params_indexes, ParmLocal, Lparms, Rparms, NSteps, Nf
from  generatingModels import generatingModels
import matplotlib.pyplot as plt
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil
import numpy as np
import os
from tqdm import tqdm
import time


RL_reference = Calc_I(freqs, recoverable_params, recoverable_params_indexes, ParmLocal, Lparms, Rparms, NSteps, Nf)
reference = RL_reference[:,5:].ravel()

def func(prs):
    y = np.zeros((prs.shape[1], len(freqs)*2))
    for i in range(prs.shape[1]):
        y[i] = Calc_I(freqs, prs[:,i], recoverable_params_indexes, ParmLocal, Lparms, Rparms, NSteps, Nf)[:,5:].ravel()
    return y

def sub(prs, i):
    return Calc_I(freqs, prs, recoverable_params_indexes, ParmLocal, Lparms, Rparms, NSteps, Nf)[:,5:].ravel(), i

# функция для многопоточной работы
def func_multythread(prs):
    # определяем число потоков и осталяем один свободным для возможности работать
    num_of_cpu = multiprocessing.cpu_count()
    y = np.zeros((prs.shape[1], len(freqs)*2))
    with ThreadPoolExecutor(max_workers = num_of_cpu - 1) as executor:
        futures = []
        for i in range(prs.shape[1]):
            futures.append(executor.submit(sub, prs[:,i], i))
        for future in tqdm(as_completed(futures), total=len(futures), desc='Расчет спектров'):
            ret, i = future.result()
            y[i] = ret
    return y

def minimizer(y):
    return functional_irrational(y, reference)
    # return functional(y, reference)

# удаляем все остатки с прошлого раза, если они есть
try:
    os.mkdir('dats')
except:
    shutil.rmtree('dats')
    os.mkdir('dats')

# привязка к количеству точек
n = len(recoverable_params_indexes)
gen = generatingModels(func_multythread, minimizer, dimensions = n, fname = 'dats/dat')

#чтоб не комментить каждый раз лишние параметры
# try:
#     gen.x0[0] = 2e7   # T_0, K
#     gen.x0[1] = 5e9   # n_0 - тепловая электронная плотность, см^{-3}
#     gen.x0[2] = 320   # B - магнитное поле, G
#     gen.x0[3] = 170    # угол между В и лучом зрения
#     # gen.x0[4] = 4e5   # n_b - нетепловая электронная плотность, см^{-3}
# except: pass

# ширина генерации
# gen.sigma = gen.x0 * 0.85

# сумарно будет насчитано points * nchildren * ngen
# for i in [0.5, 1, 2, 3, 4, 6, 10]:
#     try:
#         os.mkdir('dats')
#     except:
#         shutil.rmtree('dats')
#         os.mkdir('dats')
#     gen = generatingModels(func_multythread, minimizer, dimensions = n, fname = 'dats/dat')
#     gen.Generating(ngenerations=0, nchildren=50, sigmacoeff=i, points=2**12, method='new_random_first_gen', do_plot = True, refx = recoverable_params)
# print(gen.x0)

start = time.time()
gen.Generating(ngenerations=10, nchildren=35, sigmacoeff=8, points=2**12, method='new_random_first_gen', do_plot = True, refx = recoverable_params)
print(gen.x0)
end = time.time()
print(f"Время выполнения - {(end-start)/60} min")

# gen.plot(recoverable_params)
# сохранение графика
# plt.savefig('grade_standart.png')