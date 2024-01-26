import numpy as np
import matplotlib.pyplot as plt
import GScodes # initialization library - located either in the current directory or in the system path

libname='D:\Документы\solar_radio_diagnostics\gyrosynchrotron\Binaries\MWTransferArr64.dll' # name of the executable library - located where Python can find it

GET_MW=GScodes.initGET_MW(libname) # load the library

Nf=150    # number of frequencies
NSteps=10  # number of nodes along the line-of-sight
 
Lparms=np.zeros(11, dtype='int32') # array of dimensions etc.
Lparms[0]=NSteps
Lparms[1]=Nf
 
Rparms=np.zeros(5, dtype='double') # array of global floating-point parameters
Rparms[0]=(1.75 * 1e8 * 1.75 * 1e8) * 9  # площадь, см^2
Rparms[1]=1e9  # starting frequency to calculate spectrum, Hz
Rparms[2]=0.02 # logarithmic step in frequency
# Rparms[3]=12   #f^C
# Rparms[4]=12   #f^WH
 
L=3e8 # total source depth, cm
d = [1.02514285e+04, 4.80215384e+10, 4.46419762e+02, 9.34343702e+01, 5.77325778e+07, 5.79590544e+00]

ParmLocal=np.zeros(24, dtype='double') # массив параметров вокселя - для одного вокселя
ParmLocal[0]=L/NSteps  # глубина вокселя, см
ParmLocal[1]=d[0]  # T_0, K
ParmLocal[2]=d[1]   # n_0 - тепловая электронная плотность, см^{-3}
ParmLocal[3]=d[2]  # B - магнитное поле, G
ParmLocal[4]=d[3]    #угол между В и лучом зрения
ParmLocal[6]=3     # распределение по энергии (выбирается ЗАКОН LAW)
ParmLocal[7]=d[4]  # n_b - нетепловая электронная плотность, см^{-3}
ParmLocal[9]=0.03   # E_min, MeV
ParmLocal[10]=100.0 # E_max, MeV
ParmLocal[12]=d[5]  # \delta_1
ParmLocal[14]=0    # распределение по питч-углу (выбирается GLC), 0 - изотропное
ParmLocal[15]=10   # граница конуса потерь, градусы
ParmLocal[16]=0.2  # \Delta\mu
 
Parms=np.zeros((24, NSteps), dtype='double', order='F') # 2D array of input parameters - for multiple voxels
for i in range(NSteps):
    Parms[:, i]=ParmLocal # most of the parameters are the same in all voxels
    Parms[4, i]=d[3] - (d[3] / 5 * i / (NSteps-1))
    Parms[3, i]=d[2] + (d[2]*i/(NSteps-1))
 
RL=np.zeros((7, Nf), dtype='double', order='F') # input/output array
dummy=np.array(0, dtype='double')

# calculating the emission for analytical distribution (array -> off),
# the unused parameters can be set to any value
res=GET_MW(Lparms, Rparms, Parms, dummy, dummy, dummy, RL)
 
# retrieving the results
f=RL[0]
I_L=RL[5]
I_R=RL[6]

# plotting the results
plt.figure(1)
plt.plot(f, I_L)
plt.xscale('log')
plt.yscale('log')
plt.title('I_L')
plt.xlabel('Frequency, GHz')
plt.ylabel('Intensity, sfu')

plt.figure(2)
plt.plot(f, I_R)
plt.xscale('log')
plt.title('I_R')
plt.xlabel('Frequency, GHz')
plt.ylabel('Polarization degree')

plt.show()
