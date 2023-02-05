import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from generatingModels import generatingModels
from gyrosynchrotron import \
    GScodes  # библиотека инициализации - находится либо в текущем каталоге, либо в системном пути
from params import (Lparms, Nf, NSteps, ParmLocal, Rparms, freqs,
                    indexes_of_recoverable_parameters, recoverable_parameters)
from utils import Calc_I, functional, functional_irrational

matplotlib.rcParams.update({'font.size': 18})

freq=[4*1e9, 6*1e9, 8*1e9, 10*1e9, 12*1e9]
dop_freq = [5*1e9, 7*1e9, 9*1e9, 11*1e9]
# freqs=np.linspace(4*1e9, 12*1e9, 1024)

libname='gyrosynchrotron/Binaries/MWTransferArr64.dll' # имя исполняемой библиотеки - находится там, где Python может ее найти

GET_MW=GScodes.initGET_MW(libname) # загрузка библиотеки

Nf=1    # количество частот
NSteps=1  # количество узлов вдоль линии прямой видимости
 
Lparms=np.zeros(11, dtype='int32') # массив измерений и т.д.
Lparms[0]=NSteps
Lparms[1]=Nf
 
Rparms=np.zeros(5, dtype='double') # массив глобальных параметров с плавающей запятой
Rparms[0]=1e20 # площадь, см^2
Rparms[1]=1e9  # начальная частота для вычисления спектра, Гц
Rparms[2]=0.002 # логарифмический шаг по частоте
Rparms[3]=12   # f^C
Rparms[4]=12   # f^WH
 
L=1e10 # общая глубина источника, см
 
ParmLocal=np.zeros(24, dtype='double') # массив параметров вокселя - для одного вокселя
ParmLocal[0]=L/NSteps  # глубина вокселя, см
ParmLocal[1]=1e7   # T_0, K
ParmLocal[2]=3e9   # n_0 - тепловая электронная плотность, см^{-3}
ParmLocal[3]=180   # B - магнитное поле, G
ParmLocal[4]=80    #угол между В и лучом зрения
ParmLocal[5]= 0 + 4
ParmLocal[6]=3     # распределение по энергии (выбирается ЗАКОН LAW)
ParmLocal[7]=1e6   # n_b - нетепловая электронная плотность, см^{-3}
ParmLocal[9]=0.03   # E_min, MeV
ParmLocal[10]=10.0 # E_max, MeV
ParmLocal[12]=4.0  # \delta_1
ParmLocal[14]=3    # # распределение по питч-углу (выбирается GLC)
ParmLocal[15]=70   # граница конуса потерь, градусы
ParmLocal[16]=1  # \Delta\mu
ParmLocal[22]=-1  # \Delta\mu
 
def GS_codes_for_array_freq(freq, ParmLocal, Lparms, Rparms):

    f_model = []
    I_R_model = []
    I_L_model = []

    for one_freq in freq:

        Rparms[1]=one_freq  # начальная частота для вычисления спектра, Гц

        Parms=np.zeros((24, NSteps), dtype='double', order='F') # 2D массив входных параметров - для нескольких вокселей (трехмерных пикселей)
        for i in range(NSteps):
            Parms[:, i]=ParmLocal

        RL=np.zeros((7, Nf), dtype='double', order='F') # массив ввода/вывода
        dummy=np.array(0, dtype='double')

        res=GET_MW(Lparms, Rparms, Parms, dummy, dummy, dummy, RL)
    
        # получение результатов
        f_model.append(RL[0][0])
        I_L_model.append(RL[5][0])
        I_R_model.append(RL[6][0])

    return f_model, I_L_model, I_R_model

# f_model, I_L_model, I_R_model = GS_codes_for_array_freq(freqs, ParmLocal, Lparms, Rparms)
f_model1, I_L_model1, I_R_model1 = GS_codes_for_array_freq(freq, ParmLocal, Lparms, Rparms)
# f_model2, I_L_model2, I_R_model2 = GS_codes_for_array_freq(dop_freq, ParmLocal, Lparms, Rparms)

# f_model = np.array(f_model)
# I_L_model = np.array(I_L_model)
# I_R_model = np.array(I_R_model)

I_L_observed = [5474.28978, 7428.58723, 4557.02686, 2897.84035, 1994.35538]
I_R_observed = [5033.84767, 12171.6216, 7676.13776, 4560.20727, 2961.69482]

f_observed = f_model1
I_L_observed = np.array(I_L_observed)
I_R_observed = np.array(I_R_observed)

f_model1 = np.array(f_model1)
I_L_model1 = np.array(I_L_model1)
I_R_model1 = np.array(I_R_model1)

# f_model2 = np.array(f_model2)
# I_L_model2 = np.array(I_L_model2)
# I_R_model2 = np.array(I_R_model2)


# print(I_L_model, I_R_model)

# построение результатов
plt.figure(1)
# plt.plot(f_model, I_L_model+I_R_model)
plt.scatter(f_model1, I_L_model1+I_R_model1, s = 25, c='blue', marker="D", alpha = 1)
# plt.scatter(f_model2, I_L_model2+I_R_model2, s = 40, c='red', marker="x", alpha = 1)
# plt.scatter(f_model, I_L_model+I_R_model, s = 35, c='blue', marker="o", alpha = 1, label='Модельные данные')
plt.plot(f_observed, I_L_observed+I_R_observed, linestyle='dashdot')
plt.scatter(f_observed, I_L_observed+I_R_observed, s = 40, c='red', marker="x", alpha = 1, label='Восстановленные данные')
plt.xscale('log')
plt.yscale('log')
plt.grid(True, which="both", linestyle='--')
plt.xlabel('Частота, GHz', fontsize=20)
plt.ylabel('Интенсивность, sfu', fontsize=20)
# plt.legend() 

plt.figure(2)
# plt.plot(f_model, (I_L_model-I_R_model)/(I_L_model+I_R_model))
plt.scatter(f_model1, (I_L_model1-I_R_model1)/(I_L_model1+I_R_model1), s = 25, c='blue', marker="D", alpha = 1)
# plt.scatter(f_model2, (I_L_model2-I_R_model2)/(I_L_model2+I_R_model2), s = 40, c='red', marker="x", alpha = 1)
# plt.scatter(f_model, (I_L_model-I_R_model)/(I_L_model+I_R_model), s = 35, c='blue', marker="o", alpha = 1, label='Модельные данные')
plt.plot(f_observed, (I_L_observed-I_R_observed)/(I_L_observed+I_R_observed), linestyle='dashdot')
plt.scatter(f_observed, (I_L_observed-I_R_observed)/(I_L_observed+I_R_observed), s = 40, c='red', marker="x", alpha = 1, label='Восстановленные данные')
plt.xscale('log')
plt.grid(True, which="both", linestyle='--')
plt.xlabel('Частота, GHz', fontsize=20)
plt.ylabel('Степень круговой поляризации', fontsize=20)
# plt.legend() 

plt.show()

# RL = Calc_model_I(freqs, ParmLocal, Lparms, Rparms, NSteps, Nf)
# print(RL)
