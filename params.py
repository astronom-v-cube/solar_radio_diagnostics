import numpy as np
import platform

libname = 'gyrosynchrotron/MWTransferArr.so' if platform.system() == 'Linux' else 'gyrosynchrotron/Binaries/MWTransferArr64.dll' # имя исполняемой библиотеки - находится там, где Python сможет ее найти

# список частот, на которых происходит восстановление
# freqs=[5.8e9, 6.2e9, 6.6e9, 7e9, 7.4e9, 7.8e9, 8.2e9, 8.6e9, 9e9, 9.4e9, 9.8e9]
# freqs = np.array([2.8e9, 3e9, 3.2e9, 3.4e9, 3.6e9, 3.8e9, 4e9, 4.2e9, 4.4e9, 4.6e9, 5.2e9, 5.4e9, 5.6e9, 5.8e9, 6.2e9, 6.6e9, 7e9, 7.4e9, 7.8e9, 8.2e9, 8.6e9, 9e9, 9.4e9, 9.8e9, 10.2e9, 10.6e9, 11e9, 11.4e9, 11.8e9])
freqs = np.array([5.8e9, 6.2e9, 6.6e9, 7e9, 7.4e9, 7.8e9, 8.2e9, 8.6e9, 9e9, 9.4e9, 10.2e9, 10.6e9, 11e9, 11.4e9, 11.8e9])

############################################################  16.07.23 orig ########################################################
# LLL = [31.21496501, 37.57817085, 44.77493987, 52.81528043, 61.68903641, 71.36393919, 81.7844406 , 92.8714485, 104.5230378, 116.6161538, 154.05870298, 166.37444063, 178.31871827, 189.72118448, 210.26828538, 226.90776355, 238.85343033, 245.7022113 , 247.44062927, 244.40283043, 237.19396425, 226.59594196, 213.47185483, 198.68177437, 183.01776863, 167.1610966, 151.66061244, 136.92884572, 123.2510268] # 1
# RRR = [37.05742648,  43.77823652,  51.16024112,  59.15841689,  67.70620563, 76.7161341, 86.08145014, 95.67872968, 105.37134354, 115.01361538, 142.14464165, 150.11324496, 157.33263036, 163.69981702, 173.56777449, 179.31984509, 180.91418217, 178.62757762, 172.98270359, 164.65759493, 154.39470748, 142.92271047, 130.89857269, 118.87213141, 107.27123955, 96.40316622, 86.46705855, 77.57255579, 69.76059598] # 1

# LLL = [4.23417044, 5.12514972, 6.16378047, 7.36589161, 8.74734408, 10.32369082, 12.10979614, 14.11942164, 16.36478805, 18.85612434, 27.86902017, 31.39133075,  35.16593523,  39.18268603,  47.88039323,  57.31675495,  67.25850608,  77.41692735,  87.46395065,  97.05279581, 105.84105668, 113.51383439, 119.80454283, 124.51136627, 127.50795665, 128.74770711, 128.26170395, 126.15113041, 122.57538952] # 2
# RRR = [4.11548072, 4.95530014, 5.91847366, 7.01329314, 8.24686604, 9.62482232, 11.15104306, 12.82742288, 14.65367776, 16.62720779, 23.36962452, 25.85876894, 28.44724452, 31.11938264, 36.64513755, 42.2884001, 47.89640609, 53.32349821, 58.44207542, 63.15113765, 67.38178887, 71.09956819, 74.30391846, 77.02541689, 79.32155937, 81.27192253, 82.9734573, 84.53653865, 86.08225155] # 2

# LLL = [2.15765142, 2.73232579, 3.42572213, 4.25315546, 5.22974933, 6.36989101, 7.68662445, 9.19100236, 10.89142402, 12.79298965, 19.69418265, 22.36649447, 25.19872564, 28.16767167, 34.3997211, 40.79228183, 47.03144759, 52.79029925, 57.76216931, 61.69139947, 64.39700474, 65.78619892, 65.85673006, 64.68892557, 62.42984843, 59.27278773, 55.43541706, 51.13948381, 46.59406255] 3
# RRR = [1.52118491, 2.01320507, 2.62386105, 3.36905332,  4.26340509,  5.31929898,  6.54589021,  7.94816678,  9.52613191, 11.27418096, 17.39310968, 19.64694673, 21.95678855, 24.2865551, 28.85358979, 33.04732276, 36.60282544, 39.32549931, 41.11067363, 41.94643536, 41.90202899, 41.10682921, 39.72567116, 37.93553854, 35.90694171, 33.79147418, 31.71551925, 29.77912177, 28.05864692] # 3

# LLL = [0.86618888, 1.27694591, 1.83702429, 2.58079956, 3.54327707, 4.7575339, 6.25171381, 8.04583616, 10.14875246, 12.55561507, 21.31875984, 24.5848464, 27.90803877, 31.20770788, 37.41080962, 42.5979835, 46.33973127, 48.44024752, 48.93993701, 48.06627924, 46.15869136, 43.59314385, 40.72400207, 37.85001929, 35.20310659, 32.95421296, 31.2299869 , 30.13556942, 29.78170658] # 1.1
# RRR = [1.10011895, 1.56897899, 2.18383246, 2.96873248, 3.94454781, 5.12651993, 6.52187084, 8.12775698, 9.92985827, 11.90183889, 18.41059028, 20.59595667, 22.6891992, 24.63225442, 27.86871511, 30.00676287, 30.93211285, 30.71051426, 29.54272002, 27.70119974, 25.47007437, 23.10171411, 20.79406411, 18.68592309, 16.86430951, 15.378013, 14.25306807, 13.50802309, 13.16889555] # 1.1 

# LLL = [2.37559319, 3.10550275, 4.00692625, 5.10423735, 6.42112251, 7.97943481, 9.7979425, 11.89103675, 14.26747548, 16.92924641, 26.52727019, 30.18841812, 34.02152106, 37.9798434, 46.05664236, 53.95375336, 61.19350508, 67.34529658, 72.07613279, 75.18339698, 76.60597894, 76.41482766, 74.78781618, 71.97576718, 68.26655453, 63.95284813, 59.30700812, 54.56452182, 49.91566924] # 1.2
# RRR = [2.73947157, 3.49064331, 4.38970898, 5.4499038, 6.68183988, 8.09261134, 9.68496292, 11.45658476, 13.39959228, 15.50024157, 22.52447315, 25.00668418, 27.49947475, 29.96337845, 34.64518597, 38.74918938, 42.02426387, 44.30018467, 45.50184936, 45.64790077, 44.83642562, 43.22248456, 40.99276425, 38.34189513, 35.45349048, 32.48730138, 29.57247347, 26.80594981, 24.25461839] # 1.2

# LLL = [1.97919838, 2.75288695, 3.75450244, 5.02385174, 6.59928829, 8.51507938, 10.7985903, 13.46752041, 16.52745128, 19.96996229, 32.28230423, 36.87252314, 41.58833726, 46.34747492, 55.65630678, 64.15217439, 71.31168107, 76.80789629, 80.53609625, 82.59588164, 83.24399839, 82.8357478, 81.77048889, 80.45127467, 79.26298224, 78.56925539, 78.72691289, 80.11720096, 83.196275] # 1.3
# RRR = [2.37592814, 3.20470044, 4.23941591, 5.50372602, 7.01629004, 8.78872666, 10.82380172, 13.11405179, 15.64101412, 18.37518071, 27.38050373, 30.46775354, 33.497846, 36.41149639, 41.67622875, 45.91262169, 48.92282947, 50.6720923, 51.26794444, 50.91957516, 49.89160073, 48.46362617, 46.90208792, 45.44627067, 44.30738615, 43.67851011, 43.75375288, 44.75696325, 46.98353881] # 1.3 

# LLL = [1.84003942, 2.49610478, 3.32714917, 4.35993426, 5.61966272, 7.12832732, 8.90300306, 10.95421725, 13.28454084, 15.88753611, 25.12463178, 28.56493556, 32.10951594, 35.70472295, 42.82342647, 49.48754943, 55.33051752, 60.1012114, 63.68642778, 66.10756868, 67.49810373, 68.0712165, 68.0867282, 67.82407711, 67.56522939, 67.58899889, 68.17695943, 69.63114882, 72.30515649] # 1.4
# RRR = [1.70726457, 2.29831004, 3.03625267, 3.93865393, 5.01992764, 6.28993153, 7.75268806, 9.40536684, 11.23764505, 13.23153191, 19.89835237, 22.22715998, 24.54039824, 26.79594729, 30.97777256, 34.50643017, 37.21272657, 39.03888518, 40.03046356, 40.31289332, 40.06178521, 39.47502803, 38.75192288, 38.08163354, 37.64109295, 37.60157751, 38.14344646, 39.47995634, 41.8937372] # 1.4

# LLL = [2.51505793, 3.34855318, 4.38539822, 5.65237215, 7.17380535, 8.97000753, 11.0557359, 13.43882764, 16.11911679, 19.0877387, 29.50305503, 33.36462574, 37.34867159, 41.40577766, 49.53728188, 57.37437192, 64.6005572, 71.00764944, 76.51429243, 81.16458238, 85.11284682, 88.60255514, 91.94708206, 95.51866339, 99.7505058, 105.15660722, 112.37523639, 122.24617221, 135.94039363] # 2.1
# RRR = [2.40361818, 3.19048182, 4.15781233, 5.32316956, 6.69964081, 8.29444887, 10.10781626, 12.13221017, 14.35206152, 16.74400784, 24.6212083, 27.34812175, 30.05462982, 32.69941561, 37.65714319, 41.98378483, 45.54855374, 48.33462235, 50.42718771, 51.99071565, 53.24433629, 54.44268839, 55.8671063, 57.83026851, 60.69719263, 64.9274448, 71.14858668, 80.28130975, 93.7572313] # 2.1

# LLL = [2.42377074, 3.06212474, 3.82651429, 4.73099166, 5.78880167, 7.01184109, 8.41010907, 9.99117626, 11.75970171, 13.71702545, 20.67984128, 23.33131801, 26.12231175, 29.03243537, 35.11581957, 41.37627513, 47.59740861, 53.57382524, 59.13076935, 64.13868065, 68.52140019, 72.25805123, 75.3796582, 77.96219917, 80.11799855, 81.98724428, 83.7310947, 85.52747012, 87.5703235] # 2.2
# RRR = [2.09458219, 2.63511872, 3.27511076, 4.02278919, 4.88490905, 5.8663243, 6.96961007, 8.19475873, 9.53897237, 10.99656862, 15.9510604, 17.75130871, 19.59853574, 21.47443544, 25.23727507, 28.89429474, 32.31843118, 35.41413036, 38.12506866, 40.43635802, 42.37213829, 43.99013369, 45.37496878, 46.63190492, 47.88233691, 49.26205069, 50.92302643, 53.03958297, 55.81999242] # 2.2

# LLL = [1.53340416, 1.88352398, 2.29207414, 2.76404309, 3.30396361, 3.91575627, 4.60258328, 5.36671976, 6.20944943, 7.13099019, 10.35409219, 11.5711251, 12.8519785, 14.19093092, 17.01746151, 19.99643133, 23.07327688, 26.19934032, 29.33735568, 32.4658174, 35.58211158, 38.70457715, 41.87389944, 45.15441161, 48.63600636, 52.43747819, 56.71227831, 61.65794304, 67.53094773] # 2.3
# RRR = [0.82902546, 1.0493141, 1.3120257, 1.62122781, 1.98050662, 2.39279109, 2.86019193, 3.38386603, 3.96391574, 4.59933048, 6.81107589, 7.63622149, 8.496277, 9.38494727, 11.22173446, 13.09448455, 14.95693251, 16.77443132, 18.52804772, 20.21685222, 21.85872729, 23.49032928, 25.16701296, 26.96360861, 28.97700982, 31.3316864, 34.18960333, 37.76677309, 42.3600691] # 2.3

# LLL = [1.23281513, 1.52654455, 1.86330915, 2.24393045, 2.66842611, 3.13604069, 3.64531825, 4.19420961, 4.78020514, 5.40048286, 7.43729485, 8.1654786, 8.91424555, 9.68176573, 11.26819037, 12.92019065, 14.64135885, 16.44251542, 18.34004302, 20.35366347, 22.50370473, 24.80777451, 27.27666206, 29.90926005, 32.68636728, 35.56344432, 38.46279984, 41.26632157, 43.81071968] # 2.4
# RRR = [0.63179544, 0.83568434, 1.07863739, 1.36090948, 1.68123987, 2.03692905, 2.42404116, 2.83770351, 3.27246526, 3.72267494, 5.11367153, 5.57662241, 6.0343668, 6.48549585, 7.36715746, 8.22873219, 9.08868962, 9.97314002, 10.91238205, 11.93786556, 13.07927062, 14.36111319, 15.79814914, 17.38887149, 19.10666995, 20.88892653, 22.62567965, 24.1516152, 25.24772082] # 2.4

# LLL = [1.16666301, 1.62016158, 2.19814911, 2.91739797, 3.79235841, 4.83409729, 6.04940314, 7.44014315, 9.00292948, 10.72911855, 16.73146882, 18.93637325, 21.20223069, 23.50294169, 28.10705263, 32.55973144, 36.70543056, 40.42799769, 43.64774009, 46.31279108, 48.38811138, 49.84488472, 50.65230717, 50.77312623, 50.16383851, 48.78010914, 46.58753294, 43.57708805, 39.78337992] # 3.1
# RRR = [0.85420938, 1.24821049, 1.76591235, 2.42377393, 3.23377191, 4.20186843, 5.32698678, 6.60063972, 8.00725469, 9.52513747, 14.47019446, 16.14977242, 17.79773757, 19.39006639, 22.33262368, 24.8715192, 26.96638774, 28.6258335, 29.88592149, 30.7898642, 31.37199988, 31.64695073, 31.6038453, 31.20547214, 30.39274846, 29.09540608, 27.24972793, 24.82290007, 21.84067202] # 3.1 

# LLL = [0.9798001, 1.26915954, 1.61742552, 2.02962552, 2.50976918, 3.06061745, 3.68350575, 4.37823216, 5.14301661, 5.97453091, 8.81534716, 9.85398996, 10.92455666, 12.01795026, 14.23619722, 16.43611675, 18.55059804, 20.52095002, 22.29773557, 23.83999926, 25.11353488, 26.08885797, 26.73947713, 27.04094763, 26.97106639, 26.51142859, 25.65039507, 24.38728997, 22.73734482] # 3.2
# RRR = [0.64015296, 0.87832012, 1.17063722, 1.51880839, 1.92207292, 2.37718116, 2.87861697, 3.41902751, 3.98979556, 4.58167779, 6.39485917, 6.98649562, 7.56241788, 8.11927248, 9.16951005, 10.13656132, 11.03376217, 11.88082755, 12.69668234, 13.49315186, 14.26934308, 15.00646901, 15.66326459, 16.17295116, 16.4438005, 16.3663747, 15.83070664, 14.75484015, 13.12117448] # 3.2


############################################################  20.01.22 orig ########################################################

LLL = [1.0958675556840003, 1.2237082501306886, 1.3490403345571198, 1.4682472951213281, 1.5776138568530031, 1.6735143589928816, 1.752610470222964, 1.8120433731902006, 1.8496050551159955, 1.8638744140273389, 1.8212669924071023, 1.7660091274766463, 1.690594683943531, 1.597766393407647, 1.4907825254190796] # 0
RRR = [1.0921411898656246, 1.185627584211647, 1.275715291841387, 1.360489475253075, 1.4380453096598813, 1.5065581940980053, 1.5643546316537893, 1.609979963779846, 1.6422591439192313, 1.6603470103575837, 1.6524224686777504, 1.626620131306555, 1.587037367861314, 1.534702218223986, 1.4709470767505093] # 0
# # без 9800

# LLL = [4.594135145539357, 5.491922808938786, 6.432052712937501, 7.400493800240241, 8.387627943393655, 9.390017293376772, 10.411752184435509, 11.465496618031022, 12.57346440380642, 13.768654485379507, 18.418832597269937, 20.606806415952764, 23.33677534414567, 26.824645054290098] # 1
# RRR = [4.634151879635671, 5.59481989883073, 6.60683445751358, 7.653582577181726, 8.723137506050964, 9.810470354964531, 10.919145984071465, 12.062624599069808, 13.265444774119915, 14.564690128070001, 19.659016569694145, 22.083332030383318, 25.132151916181776, 29.062272913286023] # 1
# без 9800 и 10200

# LLL = [4.708406779952612, 5.457017358225958, 6.275725180950054, 7.164803929627035, 8.124209265389252, 9.15374646187844, 10.253266156272947, 11.422883419713461, 12.663215781945993, 13.97563685467744, 16.827638339444793, 18.37622531526292, 20.015531767492654, 21.755057430260933, 23.606967317809794] # 2
# RRR = [4.760678282942935, 5.589247584899397, 6.497409285655245, 7.4846941376285105, 8.550656090166887, 9.695293552291364, 10.919511799531977, 12.225617637574539, 13.617842816001176, 15.102899741930617, 18.394428498861046, 20.232494111369487, 22.228258190039075, 24.411750475883718, 26.82097024341064] # 2
# # без 9800

# LLL = [4.928331220442883, 5.730028106553658, 6.613938311108712, 7.582855392848465, 8.63967443591383, 9.787610397495213, 11.030453938282491, 12.372863903781022, 13.820697077048253, 15.38137800633978, 18.88136929383641, 20.84739891605904, 22.980884715116385, 25.304677788390762, 27.84689662437283] # 3
# RRR = [5.057065587072756, 5.92538982574642, 6.884053891492233, 7.93554712515378, 9.082618119833452, 10.328638720699619, 11.678021168477857, 13.136687090488705, 14.712591098436722, 16.416307315303374, 20.266662446407825, 22.454079671069458, 24.85286322780122, 27.499324906531477, 30.438857908174175] # 3
# # без 9800

# LLL = [5.220831679860928, 5.951654468491959, 6.742668295874359, 7.5958013653168805, 8.51363001383869, 9.49963604352552, 10.558490466110865, 11.696367896608203, 12.921299159498364, 14.243573715555707, 17.235505167609755, 18.941727411119942, 20.81993533821214, 22.90103179889008, 25.223087918980116] # 4
# RRR = [5.411323982404983, 6.220355744044132, 7.088328020888424, 8.014817967829424, 9.000552435236783, 10.04788581781619, 11.161295650981055, 12.347905705282638, 13.618056822722767, 14.985957613613074, 18.09602865187031, 19.893969232233847, 21.90406036142621, 24.176710316284765, 26.775867752460563] # 4
# # без 9800

# LLL = [4.895479164212093, 5.564175533417262, 6.280102383421348, 7.043850990593871, 7.856848496976055, 8.72164947514024, 9.64224488327027, 10.624395108497053, 11.675998893490576, 12.807515824603367, 15.36804694286398, 16.83589065008155, 18.463053639253996, 20.283282153267603, 22.338676886205885] # 5
# RRR = [5.094110562759132, 5.810836792936024, 6.575432090652177, 7.387725266499095, 8.248662026008615, 9.160688765394069, 10.128149545497369, 11.157707009759358, 12.25880634263517, 13.444210744956658, 16.13961996759077, 17.698448438568455, 19.441646029140454, 21.412744122626126, 23.6667484206384] # 5
# # без 9800

# LLL = [4.533849956274302, 5.168960151675097, 5.828321051664542, 6.508317656940345, 7.207086915369946, 7.924963498682887, 8.66486948071611, 9.432672891382364, 10.237555460306798, 11.092444280474108, 13.026289302494146, 15.440493177242939, 16.925865583799133, 18.672161929543936] # 6
# RRR = [4.733414925397241, 5.4369645292662785, 6.159218995505049, 6.894247412361184, 7.639149438282767, 8.394694948510672, 9.165822747807285, 9.962067079587948, 10.79801162935728, 11.693897740205108, 13.780761235117113, 16.547409336156853, 18.345270427363143, 20.54794984193829] # 6
# # без 9800 и 10600

reference = []
for i in range(len(LLL)):
    reference.append(LLL[i])
    reference.append(RRR[i])
reference = np.array(reference) / 2

Nf=len(freqs)*3       # number of frequencies
NSteps=10  # number of nodes along the line-of-sight
mfreqs = np.logspace(np.log10(freqs[0]), np.log10(freqs[-1]), Nf)

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
recoverable_params_indexes = [1, 2, 3, 4, 7, 11, 12, 13] 
# recoverable_params = np.array([1.9e6, 1.2e+11, 131, 122, 4.5e+10, 6.44], dtype='double')
recoverable_params = np.array([8.37543741e+06, 1.09374358e+10, 2.88018342e+02, 5.62577876e+02, 3.81317220e+10, 7.87934081e-01, 9.13969702e+00, 2.83425886e+00], dtype='double')
recoverable_params_analise = np.array([2.72287046e+07, 1.14966104e+10, 5.38487648e+02, 5.02921245e+02, 3.35113322e+10, 1.36936580e-01, 8.76063324e+00, 1.87562247e+00], dtype='double')

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
ParmLocal[14]=0    # распределение по питч-углу (выбирается GLC), 0 - изотропное
ParmLocal[15]=10   # граница конуса потерь, градусы
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