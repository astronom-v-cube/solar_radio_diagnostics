import matplotlib.pyplot as plt
import numpy as np
from params import (Lparms, Nf, NSteps, ParmLocal, Rparms, freqs,
                    limits_of_gen_ParmLocal, names_of_ParmLocal,
                    recoverable_params, recoverable_params_indexes, reference)
from utils import Calc_I
from generatingModels import generatingModels

from utils import Calc_I, functional, functional_irrational
import multiprocessing
from params import freqs, recoverable_params, recoverable_params_indexes, ParmLocal, Lparms, Rparms, NSteps, Nf, reference
from  generatingModels import generatingModels
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil
import numpy as np
import os
from tqdm import tqdm
import time

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

n = len(recoverable_params_indexes)
gen = generatingModels(func_multythread, minimizer, dimensions = n, fname = 'dats/dat')
gen.gen = 15

print(recoverable_params)

model_RL_reference = gen.get('y', gen.getmins(1))[0] #Calc_I(freqs, recoverable_params, recoverable_params_indexes, ParmLocal, Lparms, Rparms, NSteps, Nf)
model_reference = model_RL_reference
# print(freqs, recoverable_params, recoverable_params_indexes, ParmLocal, Lparms, Rparms, NSteps, Nf)
# Создание фигуры и подграфиков
fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex = True, sharey = True)

# Построение первого графика
axes[0].plot(freqs, model_reference[1::2], color='red', label='Модель')
axes[0].plot(freqs, reference[1::2], color='blue', label='Практика')
axes[0].set_title('Модель LCP')
axes[0].set_xlabel('freq')
axes[0].set_ylabel('intensity')
axes[0].set_yscale('log')
axes[0].legend()
#axes[0].set_ylim(np.min(reference) - 0.1, np.max(reference) + 0.1)
axes[0].grid(which='minor', color = 'k', linewidth = 0.2)
axes[0].grid(which='major', color = 'k', linewidth = 1)

# Построение второго графика
axes[1].plot(freqs, model_reference[::2], color='red', label='Модель')
axes[1].plot(freqs, reference[::2], color='blue', label='Практика')
axes[1].set_title('Модель RCP')
axes[1].legend()
#axes[1].set_ylim(np.min(reference) - 0.1, np.max(reference) + 0.1)
axes[1].grid(which='minor', color = 'k', linewidth = 0.2)
axes[1].grid(which='major', color = 'k', linewidth = 1)
fig.tight_layout()
# # Построение третьего графика
# axes[1, 0].plot(freqs, reference[1::2], color='blue')
# axes[1, 0].set_title('Спектр LCP')
# axes[1, 0].set_xlabel('freq')
# axes[1, 0].set_ylabel('intensity')
# axes[1, 0].set_yscale('log')
# axes[1, 0].set_ylim(np.min(reference) - 0.1, np.max(reference) + 0.1)
# axes[1, 0].grid(which='minor', color = 'k', linewidth = 0.2)
# axes[1, 0].grid(which='major', color = 'k', linewidth = 1)

# # Построение четвертого графика
# axes[1, 1].plot(freqs, reference[::2], color='purple')
# axes[1, 1].set_title('Спектр RCP')
# axes[1, 1].set_xlabel('freq')
# axes[1, 1].set_ylabel('intensity')
# axes[1, 1].set_yscale('log')
# axes[1, 1].set_ylim(np.min(reference) - 0.1, np.max(reference) + 0.1)
# axes[1, 1].grid(which='minor', color = 'k', linewidth = 0.2)
# axes[1, 1].grid(which='major', color = 'k', linewidth = 1)

axes[0].xaxis.set_ticks(freqs)
axes[1].xaxis.set_ticks(freqs)
# axes[1, 0].xaxis.set_ticks(freqs)
# axes[1, 1].xaxis.set_ticks(freqs)


# Отображение фигуры с графиками
plt.tight_layout()
plt.show()

