import numpy as np
import matplotlib.pyplot as plt
from gyrosynchrotron import GScodes # библиотека инициализации - находится либо в текущем каталоге, либо в системном пути

import idlsave

name = idlsave.read("Emission_sourse1_4p_(f=17_34)_l=1e8.sav")
print(name.emiss_i)

libname='gyrosynchrotron/Binaries/MWTransferArr64.dll' # имя исполняемой библиотеки - находится там, где Python может ее найти

GET_MW=GScodes.initGET_MW(libname) # загрузка библиотеки

Nf=100    # количество частот
NSteps=30  # количество узлов вдоль линии прямой видимости
 
Lparms=np.zeros(11, dtype='int32') # массив измерений и т.д.
Lparms[0]=NSteps
Lparms[1]=Nf
 
Rparms=np.zeros(5, dtype='double') # массив глобальных параметров с плавающей запятой
Rparms[0]=1e20 # площадь, см^2
Rparms[1]=1e9  # начальная частота для вычисления спектра, Гц
Rparms[2]=0.02 # логарифмический шаг по частоте
Rparms[3]=12   # f^C
Rparms[4]=12   # f^WH
 
L=1e10 # общая глубина источника, см
 
ParmLocal=np.zeros(24, dtype='double') # массив параметров вокселя - для одного вокселя
ParmLocal[0]=L/NSteps  # глубина вокселя, см
ParmLocal[1]=3e7   # T_0, K
ParmLocal[2]=3e9   # n_0 - тепловая электронная плотность, см^{-3}
ParmLocal[3]=180   # B - магнитное поле, G
ParmLocal[6]=3     # распределение по энергии (выбирается ЗАКОН LAW)
ParmLocal[7]=1e6   # n_b - нетепловая электронная плотность, см^{-3}
ParmLocal[9]=0.03   # E_min, MeV
ParmLocal[10]=10.0 # E_max, MeV
ParmLocal[12]=4.0  # \delta_1
ParmLocal[14]=3    # # распределение по питч-углу (выбирается GLC)
ParmLocal[15]=70   # граница конуса потерь, градусы
ParmLocal[16]=0.2  # \Delta\mu
 
Parms=np.zeros((24, NSteps), dtype='double', order='F') # 2D массив входных параметров - для нескольких вокселей (трехмерных пикселей)
for i in range(NSteps):
    Parms[:, i]=ParmLocal # большинство параметров одинаковы во всех вокселях 
    Parms[4, i]=50.0+30.0*i/(NSteps-1) # угол обзора варьируется от 50 до 80 градусов вдоль оси

RL=np.zeros((7, Nf), dtype='double', order='F') # массив ввода/вывода
dummy=np.array(0, dtype='double')

print(ParmLocal)

# вычисление выбросов для аналитического распределения (массив -> выкл.),
# неиспользуемые параметры могут быть установлены на любое значение
res=GET_MW(Lparms, Rparms, Parms, dummy, dummy, dummy, RL)
 
# получение результатов
f=RL[0]
I_L=RL[5]
I_R=RL[6]

# построение результатов
plt.figure(1)
plt.plot(f, I_L+I_R)
plt.xscale('log')
plt.yscale('log')
plt.grid(True)
plt.title('Total intensity (analytical)')
plt.xlabel('Frequency, GHz')
plt.ylabel('Intensity, sfu')

plt.figure(2)
plt.plot(f, (I_L-I_R)/(I_L+I_R))
plt.xscale('log')
plt.grid(True)
plt.title('Circular polarization degree (analytical)')
plt.xlabel('Frequency, GHz')
plt.ylabel('Polarization degree')

plt.show()