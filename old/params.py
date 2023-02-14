import numpy as np

libname = 'gyrosynchrotron/Binaries/MWTransferArr64.dll' # имя исполняемой библиотеки - находится там, где Python может ее найти

freqs = [4*1e9, 6*1e9, 8*1e9, 10*1e9, 12*1e9]

Nf = 1     # number of frequencies
NSteps = 1  # number of nodes along the line-of-sight
 
Lparms = np.zeros(11, dtype = 'int32') # массив измерений и т.д.
Lparms[0] = NSteps
Lparms[1] = Nf
 
Rparms = np.zeros(5, dtype = 'double') # массив глобальных параметров с плавающей запятой
Rparms[0] = 1e20 # площадь, см^2
Rparms[1] = 1e9  # начальная частота для вычисления спектра, Гц
Rparms[2] = 0.002 # логарифмический шаг по частоте
Rparms[3] = 12   # f^C
Rparms[4] = 12   # f^WH
 
L = 1e10 # общая глубина источника, см
 
ParmLocal = np.zeros(24, dtype = 'double') # массив параметров вокселя - для одного вокселя
ParmLocal[0] = L/NSteps  # глубина вокселя, см
ParmLocal[1] = 1e7   # T_0, K
ParmLocal[2] = 3e9   # n_0 - тепловая электронная плотность, см^{-3}
ParmLocal[3] = 180   # B - магнитное поле, G
ParmLocal[4] = 80    # угол между В и лучом зрения
ParmLocal[5] =  0 + 4
ParmLocal[6] = 3     # распределение по энергии (выбирается ЗАКОН LAW)
ParmLocal[7] = 1e6   # n_b - нетепловая электронная плотность, см^{-3}
ParmLocal[9] = 0.03   # E_min, MeV
ParmLocal[10] = 10.0 # E_max, MeV
ParmLocal[12] = 4.0  # \delta_1
ParmLocal[14] = 3    # # распределение по питч-углу (выбирается GLC)
ParmLocal[15] = 70   # граница конуса потерь, градусы
ParmLocal[16] = 1  # \Delta\mu
ParmLocal[22] = -1  # \Delta\mu

indexes_of_recoverable_parameters = [2, 3, 4, 7, 12] 
# indexes_of_recoverable_parameters = [3, 4, 12] 
recoverable_parameters = np.zeros(len(indexes_of_recoverable_parameters), dtype = 'double')
recoverable_parameters = ParmLocal[indexes_of_recoverable_parameters]