import numpy as np
from gyrosynchrotron import GScodes
from params import libname

# функция подсчета левой и правой интенсивности на всех частотах
def Calc_I(prs, prs_indexes):
    from params import ParmLocal, Lparms, Rparms, NSteps, Nf
    RL=np.zeros((7, Nf), dtype='double') # массив ввода/вывода
    GET_MW = GScodes.initGET_MW(libname)
    dummy = np.array(0, dtype='double')

    Parms=np.zeros((24, NSteps), dtype='double', order='F') # 2D массив входных параметров - для нескольких вокселей (трехмерных пикселей)
    for k in range(NSteps):
        Parms[:, k]=ParmLocal # most of the parameters are the same in all voxels
        Parms[prs_indexes, k]=prs
        Parms[4, k]=Parms[4, k] - (Parms[4, k] / 5 * k / (NSteps-1))
        Parms[3, k]=Parms[3, k] + (Parms[3, k] * k / (NSteps-1))

    RL=np.zeros((7, Nf), dtype='double', order='F') # массив ввода/вывода

    GET_MW(Lparms, Rparms, Parms, dummy, dummy, dummy, RL)

    return RL[5:]

# функционал из работы А. Моргачева
# тут порядок поляризаций надо проверять
def functional(RI, LI, Irefrl): # I1L, I1R, I2L, I2R
    I = (LI + RI)
    V = (RI - LI)
    # I = Irl[:,0::2] + Irl[:,1::2]
    # V = Irl[:,0::2] - Irl[:,1::2]
    Iref = Irefrl[0::2] + Irefrl[1::2]
    Vref = Irefrl[1::2] - Irefrl[0::2]
    # return np.sqrt(((I-Iref)**2 + (V-Vref)**2).sum(1))

    # это без учета поляризации
    return np.sqrt(((I-Iref)**2).sum(1))

# новый иррациональный функционал
def functional_irrational(RI, LI, Irefrl): # I1L, I1R, I2L, I2R
    I = LI + RI * 1j
    Iref = Irefrl[0::2] + Irefrl[1::2] * 1j
    return np.abs(I - Iref).sum(1)

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from params import mfreqs, recoverable_params_indexes, recoverable_params, recoverable_params_analise
    import matplotlib
    matplotlib.rcParams.update({'font.size': 18})
    
    IL, IR = Calc_I(recoverable_params, recoverable_params_indexes)
    IL1, IR1 = Calc_I(recoverable_params_analise, recoverable_params_indexes)
    
    plt.figure()
    plt.plot(mfreqs, IL, '--', label = 'L', linewidth = 4)
    plt.plot(mfreqs, IR, '--', label = 'R', linewidth = 4)
    plt.plot(mfreqs, IL1, label = 'L сдвиг', linewidth = 4)
    plt.plot(mfreqs, IR1, label = 'R сдвиг', linewidth = 4)
    plt.grid(True, which="both", linestyle='--')
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(2.8e9, 11.8e9)
    # plt.ylim(5, 30)
    plt.xlabel('Frequency, GHz')
    plt.ylabel('Intensity')
    plt.tight_layout()
    plt.show()