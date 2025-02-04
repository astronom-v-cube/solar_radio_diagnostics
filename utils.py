import numpy as np
from gyrosynchrotron.examples import GScodes
from params import libname, freqs, LLL, RRR, ParmLocal

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

    return RL[0, :], RL[5:]

# функционал из работы А. Моргачева
# тут порядок поляризаций надо проверять
def functional(RI, LI, Irefrl): # I1L, I1R, I2L, I2R
    I = (LI + RI)
    V = (RI - LI)
    # I = Irl[:,0::2] + Irl[:,1::2]
    # V = Irl[:,0::2] - Irl[:,1::2]
    Iref = Irefrl[0::2] + Irefrl[1::2]
    Vref = Irefrl[1::2] - Irefrl[0::2]
    return np.sqrt(((I-Iref)**2 + (V-Vref)**2).sum(1))

    # это без учета поляризации
    # return np.sqrt(((I-Iref)**2).sum(1))

# новый иррациональный функционал
def functional_irrational(RI, LI, Irefrl): # I1L, I1R, I2L, I2R
    I = LI + RI * 1j
    Iref = Irefrl[0::2] + Irefrl[1::2] * 1j
    return np.abs(I - Iref).sum(1)

if __name__ == "__main__":

    # LLL = [1.97919838, 2.75288695, 3.75450244, 5.02385174, 6.59928829, 8.51507938, 10.7985903, 13.46752041, 16.52745128, 19.96996229, 32.28230423, 36.87252314, 41.58833726, 46.34747492, 55.65630678, 64.15217439, 71.31168107, 76.80789629, 80.53609625, 82.59588164, 83.24399839, 82.8357478, 81.77048889, 80.45127467, 79.26298224, 78.56925539, 78.72691289, 80.11720096, 83.196275] # 1.3
    # RRR = [2.37592814, 3.20470044, 4.23941591, 5.50372602, 7.01629004, 8.78872666, 10.82380172, 13.11405179, 15.64101412, 18.37518071, 27.38050373, 30.46775354, 33.497846, 36.41149639, 41.67622875, 45.91262169, 48.92282947, 50.6720923, 51.26794444, 50.91957516, 49.89160073, 48.46362617, 46.90208792, 45.44627067, 44.30738615, 43.67851011, 43.75375288, 44.75696325, 46.98353881] # 1.3

    import matplotlib.pyplot as plt
    from params import space_freqs, recoverable_params_indexes, recoverable_params_350, recoverable_params_200, Rparms
    import matplotlib
    matplotlib.rcParams.update({'font.size': 18})

    FREQ, (IL, IR) = Calc_I(recoverable_params_350, recoverable_params_indexes)
    # print(len(FREQ))
    # print(", ".join((IL+IR).astype(str)))
    FREQ1, (IL1, IR1) = Calc_I(recoverable_params_200, recoverable_params_indexes)

    fig = plt.figure(figsize=(12, 9))

    plt.plot(FREQ*1e9, IL, '-', label = 'L 350 Гс', linewidth = 4, c='r')
    plt.plot(FREQ*1e9, IR, '--', label = 'R 350 Гс', linewidth = 4, c='r')
    # print(FREQ1)
    plt.plot(FREQ1*1e9, IL1, '-', label = 'L 200 Гс', linewidth = 4, c='b')
    plt.plot(FREQ1*1e9, IR1, '--', label = 'R 200 Гс', linewidth = 4, c='b')
    print(IL1)
    print(IR1)
    # plt.plot(space_freqs, LLL, 'D')
    # plt.plot(space_freqs, RRR, 'D')
    plt.title(fr'$\alpha_c = {ParmLocal[15]}$, $\Delta \mu_c = {ParmLocal[16]}$')
    plt.grid(True, which="both", linestyle='--')
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(2.8*1e9, 23.8*1e9)
    plt.ylim(0.1, 150)
    plt.xlabel('Frequency, Hz')
    plt.ylabel('Intensity')

    # fig.canvas.manager.window.showMaximized()
    fig.tight_layout()
    # plt.savefig(f'alpha_c = {ParmLocal[15]}, Delta mu_c = {ParmLocal[16]}.png', dpi=500)
    plt.show()