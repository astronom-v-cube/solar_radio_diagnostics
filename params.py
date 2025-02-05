import numpy as np
import platform
import matplotlib.pyplot as plt

libname = 'gyrosynchrotron/MWTransferArr.so' if platform.system() == 'Linux' else 'gyrosynchrotron/Binaries/MWTransferArr64.dll' # имя исполняемой библиотеки - находится там, где Python сможет ее найти

# список частот, на которых происходит восстановление
# freqs=[5.8e9, 6.2e9, 6.6e9, 7e9, 7.4e9, 7.8e9, 8.2e9, 8.6e9, 9e9, 9.4e9, 9.8e9]
# freqs = np.array([2.8e9, 3e9, 3.2e9, 3.4e9, 3.6e9, 3.8e9, 4e9, 4.2e9, 4.4e9, 4.6e9, 5.2e9, 5.4e9, 5.6e9, 5.8e9, 6.2e9, 6.6e9, 7e9, 7.4e9, 7.8e9, 8.2e9, 8.6e9, 9e9, 9.4e9, 9.8e9, 10.2e9, 10.6e9, 11e9, 11.4e9, 11.8e9])
# freqs = np.array([5.8e9, 6.2e9, 6.6e9, 7e9, 7.4e9, 7.8e9, 8.2e9, 8.6e9, 9e9, 9.4e9, 10.2e9, 10.6e9, 11e9, 11.4e9, 11.8e9]) #


# freqs = np.array([3e9, 5e9, 7e9, 9e9, 11e9, 13e9, 15e9, 17e9, 19e9, 21e9, 23e9])
freqs = np.array([3.0, 3.197166180597847, 3.4072905287862083, 3.63122468203554, 3.869876249185399, 4.124212488998135, 4.3952642304747025, 4.684130050821713, 4.991980728003084, 5.32006398592256, 5.669709551469396, 6.042334543923514, 6.439449218563458, 6.862663087756107, 7.313691444337006, 7.794362313720699, 8.306623862918068, 8.852552296489563, 9.434360271436745, 10.054405865137827, 10.715202132674433, 11.41942729228554, 12.169935580230458, 12.969768819055751, 13.822168746152506, 14.73059015257178, 15.698714885350057, 16.730466770096402, 17.83002751432277, 19.00185365597395, 20.250694625849825, 21.58161199712721, 22.999999999999957]) * 1e9
# freqs = np.array([2.8e9, 3e9, 3.2e9, 3.4e9, 3.6e9, 3.8e9, 4e9, 4.2e9, 4.4e9, 4.6e9, 5.2e9, 5.4e9, 5.6e9, 5.8e9, 6.2e9, 6.6e9, 7e9, 7.4e9, 7.8e9, 8.2e9, 8.6e9, 9e9, 9.4e9, 9.8e9, 10.2e9, 10.6e9, 11e9, 11.4e9, 11.8e9])

############## модели без анизотропии

# LLL = [1.9709710764342239, 2.2740463001015256, 2.625010316785983, 3.032545913342126, 3.5073923342654303, 4.0625120081796995, 4.713923666543442, 5.481483708175706, 6.3899353641449705, 7.470487929837572, 8.762634060876152, 10.316876790285784, 12.199153140167187, 14.499895244093052, 17.351400633832917, 20.943285818813045, 25.38680131580179, 30.206599009670526, 34.08477343709394, 35.6877051379472, 34.64568966673013, 31.558904970486157, 27.383136328775233, 22.935332457585453, 18.73017082795293, 15.017337430929238, 11.87395211879989, 9.283892218541164, 7.18911427677168, 5.518107615899294, 4.199954009667041, 3.170320782353565, 2.3734799504879804] # 200

# RRR = [1.556929258120866, 1.8715553163686924, 2.2205867264302634, 2.6153483061010157, 3.0672645636414853, 3.589005149786618, 4.1953670717958085, 4.904156609425995, 5.737217682517727, 6.721724346019482, 7.891860828236289, 9.291035106471146, 10.974802690452838, 13.014707934114249, 15.503262242193243, 18.560246469164845, 22.340625334025486, 27.050000450865788, 33.002703372635374, 40.68532657352549, 49.93599536787024, 57.95943102972578, 60.85734222266705, 57.64007058227319, 50.309685202460386, 41.46562032471018, 32.89312602700511, 25.436754789425716, 19.32893485409958, 14.500143027552038, 10.766919283960107, 7.924443852780835, 5.784998404796549] # 200

# LLL = [2.915770773609363, 3.8041295097157586, 4.9195198406876495, 6.329632586319533, 8.120967957032374, 10.385800492728881, 13.168626093115785, 16.382238846961556, 19.768769880081933, 22.963708126026866, 25.621558162303412, 27.514865310518413, 28.562814598534693, 28.80478267694133, 28.35319345098007, 27.35122106009976, 25.944062675475067, 24.262927408002138, 22.418691798520907, 20.500546763367225, 18.577516001178743, 16.70119550747163, 14.908530742305707, 13.224485725407726, 11.664670412009764, 10.237083864584472, 8.944148303826488, 7.783850450372258, 6.751123310029251, 5.838660644882706, 5.037735641729696, 4.338812804877318, 3.732012493190137] # 350

# RRR = [1.198562427709363, 1.9420749336130168, 2.834950578661557, 3.9309438645575843, 5.295571041314085, 7.010362527833894, 9.176534556192909, 11.918107859859267, 15.386487106150119, 19.773574687100375, 25.329949384141624, 32.32286935532914, 40.74932030311075, 49.87112022986816, 58.23241540360779, 64.36227305764828, 67.45256466037286, 67.496548920382, 65.02810652230595, 60.78750642498065, 55.48966521736681, 49.71587847203986, 43.88889982936635, 38.28940387317084, 33.086383503058464, 28.367775936835503, 24.165972117661358, 20.477227279546106, 17.27562499969296, 14.522710839964251, 12.174035649156036, 10.183408587603184, 8.505625021068994] # 350

# # с анизотропией 200
# LLL = [2.02877103, 2.54714337, 3.16701915, 3.91168779, 4.80210902, 5.82748522, 6.88862184, 7.78329107, 8.29799246, 8.32989829, 7.92227181, 7.20926512, 6.34120781, 5.43861978, 4.5794791, 3.80505507, 3.13123562, 2.5587661, 2.08046564, 1.68572735, 1.36304075, 1.10129884, 0.89039946, 0.72146286, 0.58685712, 0.48011602, 0.39583701, 0.32954417, 0.27756481, 0.23690979, 0.20516726, 0.18040159, 0.16107682]

# RRR = [1.12934407, 1.78531436, 2.54800751, 3.45682098, 4.55664518, 5.90106885, 7.55482027, 9.59554928, 12.11471642, 15.21719934, 19.0192557, 23.6444183, 29.20059381, 35.56377652, 41.75344512, 45.88831695, 46.53177944, 43.71908, 38.57858596, 32.44785988, 26.33972971, 20.832901, 16.16619745, 12.36957418, 9.36663711, 7.03912099, 5.26217808, 3.92124899, 2.91854013, 2.17413084, 1.62458416, 1.22059621, 0.9244623]

# с анизотропией 350
LLL = [1.72653413534138, 1.9837132527801775, 2.284652725523993, 2.6359760131673715, 3.0459004403763505, 3.5245501130688854, 4.0843550497614896, 4.74067138225886, 5.512495583839226, 6.423544138182175, 7.50359148630419, 8.790287838019587, 10.331564403601357, 12.188828300714379, 14.441053103067116, 17.189968549074017, 20.559074547249907, 24.607479600134305, 28.967633302680767, 32.509518193662196, 33.909053714159406, 32.68355019153312, 29.396147673896515, 25.05676633152527, 20.541775436341798, 16.384547034726587, 12.81761015911023, 9.885569956711938, 7.53978398161299, 5.696538470623351, 4.2668529422034975, 3.1693636967004504, 2.334562124885125]

RRR = [1.554165201500327, 1.8678821155440475, 2.2155993144456265, 2.6084574151582225, 3.057662483159293, 3.575603309001738, 4.176702346091059, 4.87825692123531, 5.701410726398334, 6.672364730531086, 7.823938847029022, 9.19761317189546, 10.846201608288272, 12.837332179720917, 15.257912234825005, 18.21971640485092, 21.866107259618712, 26.379622831385745, 31.989679356283006, 38.97892572920737, 47.685918274211176, 58.48738925183948, 71.40080887921384, 84.12863873503555, 91.33979157297728, 89.50117756240725, 80.00615666943759, 66.72099274017984, 52.98813282270045, 40.67856293733206, 30.483583271630586, 22.433246793786893, 16.270104207389846]

reference = []
for i in range(len(LLL)):
    reference.append(LLL[i])
    reference.append(RRR[i])
reference = np.array(reference)

Nf=33     # number of frequencies
NSteps=2  # number of nodes along the line-of-sight
space_freqs = np.logspace(np.log10(freqs[0]), np.log10(freqs[-1]), Nf)
# print(", ".join(space_freqs.astype(str)))
# plt.scatter(space_freqs, space_freqs)
# plt.show()
# space_freqs = freqs

Lparms=np.zeros(11, dtype='int32') # массив измерений и т.д.
Lparms[0]=NSteps
Lparms[1]=Nf

Rparms=np.zeros(5, dtype='double') # массив глобальных параметров с плавающей запятой
Rparms[0]=(1.75 * 1e8 * 1.75 * 1e8) * 9  # площадь, см^2
Rparms[1]=freqs[0]
Rparms[2]=(np.log10(freqs[-1])-np.log10(freqs[0]))/(Nf-1)
# Rparms[3]=12   #f^C
# Rparms[4]=12   #f^WH

L=3e8 # общая глубина источника, см
# индексы восстанавливаемых параметров
recoverable_params_indexes = [1, 2, 3, 4, 7, 11, 12, 13, 15, 16] # [1, 2, 3, 4, 7, 11, 12, 13]
# recoverable_params = np.array([1.9e6, 1.2e+11, 131, 122, 4.5e+10, 6.44], dtype='double')
# recoverable_params = np.array([1e7, 1e10, 350, 85, 8e6, 0.3, 4, 8], dtype='double') # первая модель
recoverable_params_200 = np.array([1e7, 5e10, 200, 65, 8e6, 0.5, 3, 5, 90, 0.2], dtype='double') # вторая модель [1e7, 5e10, 200, 65, 8e6, 0.5, 3, 5]  [1.32755297e+07, 1.19274978e+10, 8.40535603e+01, 5.85318963e+02, 3.78930031e+10, 4.97058000e+00]
recoverable_params_350 = np.array([1e7, 1e10, 350, 85, 8e6, 0.3, 4, 8, 90, 0.2], dtype='double') # первая [1e7, 1e10, 350, 85, 8e6, 0.3, 4, 8]   [1.32755297e+07, 1.19274978e+10, 8.40535603e+01, 5.85318963e+02, 3.78930031e+10, 4.97058000e+00]


ParmLocal=np.zeros(24, dtype='double') # массив параметров вокселя - для одного вокселя
ParmLocal[0]=L/NSteps  # глубина вокселя, см
#ParmLocal[1]=d[0]  # T_0, K
#ParmLocal[2]=d[1]   # n_0 - тепловая электронная плотность, см^{-3}
#ParmLocal[3]=d[2]  # B - магнитное поле, G
#ParmLocal[4]=d[3]    #угол между В и лучом зрения
ParmLocal[6]=4     # распределение по энергии (выбирается ЗАКОН LAW)
#ParmLocal[7]=d[4]  # n_b - нетепловая электронная плотность, см^{-3}
ParmLocal[9]=0.03   # E_min, MeV
ParmLocal[10]=10 # E_max, MeV
# ParmLocal[11]=0.3 # E_break, MeV
#ParmLocal[12]=d[5]  # \delta_1
#ParmLocal[13]=d[5]  # \delta_2
ParmLocal[14]=4    # распределение по питч-углу, 0 - изотропное, 3 - GLC
ParmLocal[15]=90   # граница конуса потерь, градусы
ParmLocal[16]=0.2  # \Delta\mu

limits_of_gen_ParmLocal = np.zeros(24, dtype=[('min', float), ('max', float), ('axes_scale', 'U10')])
limits_of_gen_ParmLocal[1]=(1e6, 1e8, 'log')   # T_0, K
limits_of_gen_ParmLocal[2]=(1e8, 1e11, 'log')  # n_0 - тепловая электронная плотность, см^{-3}
limits_of_gen_ParmLocal[3]=(20, 1000, 'linear')   # B - магнитное поле, G
limits_of_gen_ParmLocal[4]=(0, 360, 'linear')   #угол между В и лучом зрения
# limits_of_gen_ParmLocal[5]= 0 + 4
# limits_of_gen_ParmLocal[6]=3     # распределение по энергии (выбирается ЗАКОН LAW)
limits_of_gen_ParmLocal[7]=(1e3, 1e10, 'log')   # n_b - нетепловая электронная плотность, см^{-3}
# limits_of_gen_ParmLocal[9]=0.03   # E_min, MeV
# limits_of_gen_ParmLocal[10]=10.0 # E_max, MeV
limits_of_gen_ParmLocal[12]=(2.0, 7.0, 'linear')  # \delta_1
limits_of_gen_ParmLocal[13]=(2.0, 7.0, 'linear')  # \delta_1
limits_of_gen_ParmLocal[15]=(0, 180, 'linear')   # граница конуса потерь, градусы
limits_of_gen_ParmLocal[16]=(0, 7, 'linear')  # \Delta\mu
# limits_of_gen_ParmLocal[22]=-1  # \Delta\mu

names_of_ParmLocal = [''] * 24
names_of_ParmLocal[1]=r'$T_0, K$'
names_of_ParmLocal[2]=r'$n_0, sm^{-3}$'
names_of_ParmLocal[3]=r'$B, G$'
names_of_ParmLocal[4]=r'$\theta, grad$'
names_of_ParmLocal[7]=r'$n_b, sm^{-3}$'
names_of_ParmLocal[9]=r'$E_{min}, MeV$'
names_of_ParmLocal[10]=r'$E_{max}, MeV$'
names_of_ParmLocal[11]=r'$E_{break}, MeV$'
names_of_ParmLocal[12]=r'$\delta_1$'
names_of_ParmLocal[13]=r'$\delta_2$'
names_of_ParmLocal[15]=r'$theta_2$'   # граница конуса потерь, градусы
names_of_ParmLocal[16]=r'$\Delta \mu$'  # \Delta\mu
# names_of_ParmLocal[22]=-1  # \Delta\mu