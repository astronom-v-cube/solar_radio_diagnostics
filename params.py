import numpy as np

libname = 'gyrosynchrotron/Binaries/MWTransferArr64.dll' # имя исполняемой библиотеки - находится там, где Python сможет ее найти

# список частот, на которых происходит восстановление
freqs=[3*1e9, 4*1e9, 5*1e9, 6*1e9, 7*1e9, 8*1e9, 9*1e9, 10*1e9, 11*1e9, 12*1e9]
# freqs=[4*1e9, 6*1e9, 8*1e9, 10*1e9, 12*1e9]


Nf=1     # number of frequencies
NSteps=1  # number of nodes along the line-of-sight

Lparms=np.zeros(11, dtype='int32') # массив измерений и т.д.
Lparms[0]=NSteps
Lparms[1]=Nf
 
Rparms=np.zeros(5, dtype='double') # массив глобальных параметров с плавающей запятой
Rparms[0]=1e20 # площадь, см^2
Rparms[1]=1e9  # начальная частота для вычисления спектра, Гц
Rparms[2]=0.002 # логарифмический шаг по частоте
Rparms[3]=12   # f^C
Rparms[4]=12   # f^WH
 
L=1e9 # общая глубина источника, см

ParmLocal=np.zeros(24, dtype='double') # массив параметров вокселя - для одного вокселя
ParmLocal[0]=L/NSteps  # глубина вокселя, см
ParmLocal[1]=1e7   # T_0, K
ParmLocal[2]=1e8   # n_0 - тепловая электронная плотность, см^{-3}
ParmLocal[3]=165   # B - магнитное поле, G
ParmLocal[4]=75    #угол между В и лучом зрения
ParmLocal[5]= 0 + 4
ParmLocal[6]=3     # распределение по энергии (выбирается ЗАКОН LAW)
ParmLocal[7]=1e8   # n_b - нетепловая электронная плотность, см^{-3}
ParmLocal[9]=0.03   # E_min, MeV
ParmLocal[10]=10.0 # E_max, MeV
ParmLocal[12]=6  # \delta_1
ParmLocal[14]=3    # # распределение по питч-углу (выбирается GLC)
ParmLocal[15]=70   # граница конуса потерь, градусы
ParmLocal[16]=1  # \Delta\mu
ParmLocal[22]=-1  # \Delta\mu

limits_of_gen_ParmLocal = np.zeros(24, dtype=[('min', float), ('max', float), ('axes_scale', 'U10')])
limits_of_gen_ParmLocal[1]=(1e6, 1e8, 'log')   # T_0, K
limits_of_gen_ParmLocal[2]=(1e8, 1e11, 'log')  # n_0 - тепловая электронная плотность, см^{-3}
limits_of_gen_ParmLocal[3]=(20, 1000, 'linear')   # B - магнитное поле, G
limits_of_gen_ParmLocal[4]=(0, 360, 'linear')   #угол между В и лучом зрения
# limits_of_gen_ParmLocal[5]= 0 + 4
# limits_of_gen_ParmLocal[6]=3     # распределение по энергии (выбирается ЗАКОН LAW)
limits_of_gen_ParmLocal[7]=(1e6, 1e10, 'log')   # n_b - нетепловая электронная плотность, см^{-3}
# limits_of_gen_ParmLocal[9]=0.03   # E_min, MeV
# limits_of_gen_ParmLocal[10]=10.0 # E_max, MeV
limits_of_gen_ParmLocal[12]=(2.0, 7.0, 'linear')  # \delta_1
# limits_of_gen_ParmLocal[14]=3    # # распределение по питч-углу (выбирается GLC)
limits_of_gen_ParmLocal[15]=(0, 360, 'linear')   # граница конуса потерь, градусы
# limits_of_gen_ParmLocal[16]=1  # \Delta\mu
# limits_of_gen_ParmLocal[22]=-1  # \Delta\mu

names_of_ParmLocal = [''] * 24
names_of_ParmLocal[1]=r'$T_0, K$'
names_of_ParmLocal[2]=r'$n_0, sm^{-3}$'
names_of_ParmLocal[3]=r'$B, G$'
names_of_ParmLocal[4]=r'$\theta, grad$'
# names_of_ParmLocal[5]= 0 + 4
# names_of_ParmLocal[6]=3     # распределение по энергии (выбирается ЗАКОН LAW)
names_of_ParmLocal[7]=r'$n_b, sm^{-3}$'
# names_of_ParmLocal[9]=r'$E_min, MeV$'
# names_of_ParmLocal[10]=r'$E_max, MeV$'
names_of_ParmLocal[12]=r'$\delta_1$'
# names_of_ParmLocal[14]=3    # # распределение по питч-углу (выбирается GLC)
names_of_ParmLocal[15]=r'$theta_2$'   # граница конуса потерь, градусы
# names_of_ParmLocal[16]=1  # \Delta\mu
# names_of_ParmLocal[22]=-1  # \Delta\mu

# индексы восстанавливаемых параметров
# [12, 2], [12, 3], [12, 4], [12, 7], [2, 3], [2, 4], [2, 7], [3, 4], [4, 7], [3, 7]
recoverable_params_indexes = [2, 7] 
recoverable_params=np.zeros(len(recoverable_params_indexes), dtype='double')
recoverable_params = ParmLocal[recoverable_params_indexes]