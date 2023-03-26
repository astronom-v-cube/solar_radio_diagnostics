import numpy as np
from gyrosynchrotron import GScodes
from params import libname

GET_MW = GScodes.initGET_MW(libname)

dummy = np.array(0, dtype='double')

# функция подсчета левой и правой интенсивности на всех частотах
def Calc_I(freqs, prs, prs_indexes, ParmLocal, Lparms, Rparms, NSteps, Nf):

    RL=np.zeros((len(freqs), 7, Nf), dtype='double') # массив ввода/вывода

    for i, freq in enumerate(freqs):

        Rparms[1]=freq  # начальная частота для вычисления спектра, Гц

        Parms=np.zeros((24, NSteps), dtype='double', order='F') # 2D массив входных параметров - для нескольких вокселей (трехмерных пикселей)
        Parms.T[:] = ParmLocal
        Parms.T[:, prs_indexes] = prs

        RL_=np.zeros((7, Nf), dtype='double', order='F') # массив ввода/вывода

        res=GET_MW(Lparms, Rparms, Parms, dummy, dummy, dummy, RL_)
        RL[i] = RL_
        
    return RL #[:,[5,6]]

# функционал из работы А. Моргачева
def functional(Irl, Irefrl): # I1L, I1R, I2L, I2R
    I = Irl[:,0::2] + Irl[:,1::2]
    V = Irl[:,0::2] - Irl[:,1::2]
    Iref = Irefrl[0::2] + Irefrl[1::2]
    Vref = Irefrl[0::2] - Irefrl[1::2]
    # return np.sqrt(((I-Iref)**2/Iref**2 + (V-Vref)**2/Vref**2).sum(1))

    # еще вот так его писали, это без учета поляризации
    return np.sqrt(((I-Iref)**2/Iref**2).sum(1))

# новый иррациональный функционал
def functional_irrational(Irl, Irefrl): # I1L, I1R, I2L, I2R
    I = Irl[:,0::2] + Irl[:,1::2] * 1j
    Iref = Irefrl[0::2] + Irefrl[1::2] * 1j
    return np.abs(I - Iref).sum(1)
    