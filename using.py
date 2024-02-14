from utils import Calc_I, functional_irrational, functional
import multiprocessing
from params import recoverable_params, recoverable_params_indexes, ParmLocal, Lparms, Rparms, NSteps, Nf, reference, freqs, space_freqs
from  generatingModels import generatingModels
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil
import numpy as np
import os
from tqdm import tqdm
import time

# функция для однопоточной работы
def func(prs):
    y = np.zeros((prs.shape[1], len(freqs)*2))
    for i in range(prs.shape[1]):
        y[i] = Calc_I(prs, recoverable_params_indexes).ravel()
    return y

def sub(prs, i):
    # n_b = prs[3] / 0.03 ** (prs[4] - 1)
    # prs[3] = n_b
    return Calc_I(prs, recoverable_params_indexes).ravel(), i

# функция для многопоточной работы
def func_multythread(prs):
    # определяем число потоков и осталяем один свободным для возможности работать
    num_of_cpu = multiprocessing.cpu_count()
    y = np.zeros((prs.shape[1], Nf*2))
    with ThreadPoolExecutor(max_workers = num_of_cpu - 2) as executor:
        futures = []
        for i in range(prs.shape[1]):
            futures.append(executor.submit(sub, prs[:,i], i))
        for future in tqdm(as_completed(futures), total=len(futures), desc='Расчет спектров'):
            ret, i = future.result()
            y[i] = ret
    return y

def minimizer(y):
    mLIs = y[:,:Nf]
    mRIs = y[:,Nf:]
    LIs = np.array([np.interp(freqs, space_freqs, mLI) for mLI in mLIs])
    RIs = np.array([np.interp(freqs, space_freqs, mRI) for mRI in mRIs])
    return functional_irrational(RIs, LIs, reference)
    # return functional(RIs, LIs, reference)

if __name__ == "__main__":

    # удаляем все остатки с прошлого раза, если они есть
    try:
        os.mkdir('dats')
        os.mkdir('graphs')
    except:
        shutil.rmtree('graphs')
        os.mkdir('graphs')
        # os.system('rm -rf dats/*')
        shutil.rmtree('dats')
        os.mkdir('dats')
    
    # привязка к количеству точек
    n = len(recoverable_params_indexes)
    gen = generatingModels(func_multythread, minimizer, dimensions = n, fname = 'dats/dat')
    
    # начальная (центральная) точка генерации (параметры указываются по порядку)
    try:
        gen.x0[0]=1e7
        gen.x0[1]=1e10
        gen.x0[2]=350
        gen.x0[3]=85
        gen.x0[4]=8e6
        gen.x0[5]=0.3
        gen.x0[6]=4
        gen.x0[7]=8
        
        # gen.x0[0]=4.68e+07
        # gen.x0[1]=2.5e+9
        # gen.x0[2]=75
        # gen.x0[3]=120
        # gen.x0[4]=4.07535875e+10
        # gen.x0[5]=5
        
    except: pass 

    # ширина генерации
    gen.sigma = [gen.x0[0]*10, gen.x0[1]*10, gen.x0[2], gen.x0[3], gen.x0[4]*10, gen.x0[5], gen.x0[6], gen.x0[7]]
    
    # gen.sigma = [gen.x0[1]*5, gen.x0[2]*5, 100, 30, gen.x0[4]*5, 4]
    
    start = time.time()
    
    gen.Generating(ngenerations=50, nchildren=150, sigmacoeff=3, points=2**10, method='log_gaussian', do_plot = True, refx = recoverable_params)
    
    end = time.time()
    print(f"Время выполнения - {(end-start)/60} min")