import numpy as np
import matplotlib.pyplot as plt
import time, os, shutil
from tqdm import tqdm
import corner
from params import freqs, space_freqs, ParmLocal, Lparms, Rparms, NSteps, Nf, recoverable_params_350, recoverable_params_indexes, limits_of_gen_ParmLocal, names_of_ParmLocal, reference
import matplotlib
from matplotlib.ticker import ScalarFormatter
matplotlib.rcParams.update({'font.size': 25})

import logging
logging.basicConfig(filename = 'diagnostics_logs.log',  filemode='a', level = logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
def logprint(msg):
    print(msg)
    logging.info(msg)

class generatingModels:
    # calcfunc(x), x=[x0,x1,x2,...,x(dim-1)]
    def __init__(self, calcfunc, minimize, dimensions = 1, fname = 'dats'):
        # подсчет излучения
        self.func = calcfunc
        # минимизация
        self.minif = minimize
        # количество осей
        self.ndim = dimensions
        # начальная точка
        self.x0 = np.zeros(dimensions)
        # больше сигма - шире точки
        self.sigma = np.ones(dimensions)
        # имя файла
        self.fname = fname
        # номер генерации, с которого начинаем (счетчик)
        self.gen = 0
        # массив для хранения найденных значений
        self.x0s = []
        self.x = np.zeros((0,dimensions))
        
###################
# реализация гененирования чисто int чисел не доделана!!!!!!!!!!!! КОСТЫЛЬ
    def float_and_int_generation(self, points, desc, method):
        
        if method=='old_gaussian':
            x = np.random.normal(self.x0, self.sigma, (points, self.ndim))
        
        elif method=='log_gaussian':
        # Создаем логарифмические распределения по нужным осям
            dist_lognorm = []
            for i, index in enumerate(recoverable_params_indexes):
                if index in [2, 7]:
                    # Создаем логнормальное распределение
                    dist = np.random.lognormal(mean=np.log(self.x0[i]), sigma=np.log(self.sigma[i]), size=points)
                else:
                    # Создаем нормальное распределение
                    dist = np.random.normal(self.x0[i], self.sigma[i], size=points)
                dist_lognorm.append(dist)

            # Объединяем все распределения в один
            x = np.column_stack(dist_lognorm)
            # first_vals = x[:, 0]
            # plt.hist(first_vals, bins=2048, log=True, range=(0,1e10))
            # plt.show()
            
        elif method=='gaussian':
            # (начальная точка, ширина, (количество точек, размерность))
            x = np.random.normal(self.x0, self.sigma, (points, self.ndim))
            # чтобы генерация была в нужной области и не было 3к градусов сжимаем полученный гаусс до пределов генерации
            for i, index in enumerate(recoverable_params_indexes):
                a = limits_of_gen_ParmLocal[index][0]
                b = limits_of_gen_ParmLocal[index][1]
                x[:, i] = (b - a) * (x[:, i] - x[:, i].min()) / (x[:, i].max() - x[:, i].min()) + a

        elif method == 'random':
            # (точка слева, точка справа, (количество точек, размерность))
            x = np.random.uniform(self.x0 - self.sigma, self.x0 + self.sigma, (points, self.ndim))

        elif method=='new_random_first_gen':
            x = []
            # для всех точек по их количеству
            for i in tqdm(range(points), desc=desc):
                # массив второго уровня хранящий одну точку
                one_point = np.zeros(len(recoverable_params_indexes))
                # проходимся по индексам и берем границы генерации 
                for k, index in enumerate(recoverable_params_indexes):
                    if index in []:
                        if index == 12:
                            one_point[k] = np.random.randint(2, 8)
                    else:
                        one_point[k] = np.random.uniform(limits_of_gen_ParmLocal[index][0], limits_of_gen_ParmLocal[index][1])
                x.append(one_point)
        
        # elif method=='log_new_random_first_gen':
        #     x = []
        #     # для всех точек по их количеству
        #     for i in tqdm(range(points), desc=desc):
        #         # массив второго уровня хранящий одну точку
        #         one_point = np.zeros(len(recoverable_params_indexes))
        #         # проходимся по индексам и берем границы генерации 
        #         for k, index in enumerate(recoverable_params_indexes):
        #             if index in [2, 7]:
        #                 one_point[k] = np.random.uniform(limits_of_gen_ParmLocal[index][0], limits_of_gen_ParmLocal[index][1])
        #                 one_point[k] = np.exp(one_point[k])
        #             else:
        #                 one_point[k] = np.random.uniform(limits_of_gen_ParmLocal[index][0], limits_of_gen_ParmLocal[index][1])

        #         x.append(one_point)
        # x = np.array(x)
        return x

    def generate(self, points, method):

        filex = open(f'{self.fname}_gen_{self.gen}_x.txt', 'a')
        filey = open(f'{self.fname}_gen_{self.gen}_y.txt', 'a')
        filer = open(f'{self.fname}_gen_{self.gen}_r.txt', 'a')
        
        #while self.x.shape[0] < points:
        # генерация координат
        if method == 'gaussian':
            x = self.float_and_int_generation(points, desc = 'Генерация массива координат', method=method)

        elif method == 'old_gaussian':
            x = self.float_and_int_generation(points, desc = 'Генерация массива координат', method=method)

        elif method == 'log_gaussian':
            x = self.float_and_int_generation(points, desc = 'Генерация массива координат', method=method)

        elif method == 'random':
            x = self.float_and_int_generation(points, desc = 'Генерация массива координат', method=method)

        elif method == 'new_random_first_gen':
            x = self.float_and_int_generation(points, desc = 'Генерация массива координат', method=method)

        # вбрасываем немного случайных координат вне зависимости от области генерации после 0 поколения в случае линейной генерации и в любом случае, если по гауссу
        # if method == 'gaussian' or 'old gaussian':
        #     x_additional = self.float_and_int_generation(points//3, desc = 'Генерация дополнительного массива координат', method='new_random_first_gen')
        #     x = np.concatenate((x, x_additional), axis=0)
            
        # elif self.gen:
        #     x_additional = self.float_and_int_generation(points//3, desc = 'Генерация дополнительного массива координат', method='new_random_first_gen')
        #     x = np.concatenate((x, x_additional), axis=0)

        # строка для вставки любого набора параметров
        # np.append(x, [])

        # удаление нулевых или отрицательных координат
        # ищем отрицательную координату в каждом наборе координат
        mask = x.min(1) > 0
        # удаление из массива точек с отрицательными координатами
        x = x[mask]
        self.x = x
        
        # чистка угла
        mask = x[:, 3] < 180
        x = x[mask]
        self.x = x
        
        # транспонируем - так удобнее
        y = self.func(x.T)
        self.y = y
        # расчет отклонения от истинных параметров - функционал
        r = self.minif(y)
        self.r = r

        # создание маски для проверки на nan у функционалов
        mask = ~np.isnan(r)
        
        # применение маски
        x, y, r = x[mask], y[mask], r[mask]

        # координаты
        np.savetxt(filex, x)
        filex.close()
        # интенсивности
        np.savetxt(filey, y)
        filey.close()
        # значения функционалов
        np.savetxt(filer, r)
        filer.close() 
        
    # n - количество точек, находит индексы n минимальных
    def getmins(self, n):
        r = np.loadtxt(f'{self.fname}_gen_{self.gen}_r.txt')
        return r.argsort()[:n]
    
    def get(self, ax, indexes = None):
        # если индексы не заданы возвращает полностью массив
        if isinstance(indexes, type(None)): return np.loadtxt(f'{self.fname}_gen_{self.gen}_{ax}.txt')
        # если индексы заданы возвращает по ним
        return np.loadtxt(f'{self.fname}_gen_{self.gen}_{ax}.txt')[indexes]
    
    def corner_plot(self, x, r, number_of_gen):
        # # получение координат истинной точки для отображения
        # truths = []
        # for i in recoverable_params_indexes:
        #     truths.append(ParmLocal[i])
        # получение интервалов генерации точек для отображения
        ranges = []
        axes_scales = []
        for i in recoverable_params_indexes:
            ranges.append([limits_of_gen_ParmLocal[i][0], limits_of_gen_ParmLocal[i][1]])
            axes_scales.append(limits_of_gen_ParmLocal[i][2])
            
        print(axes_scales)
        # получение подписей для графиков
        titles = []
        for i in recoverable_params_indexes:
            titles.append(names_of_ParmLocal[i]) 

        corner_figure = plt.figure(figsize=(25, 25))
        print(ranges)
        corner.corner(data = x, titles = titles, fig = corner_figure, title_fmt = None, show_titles = True, truth_color = 'red', range = ranges) 
        corner_figure.tight_layout()
        # , plot_datapoints=False  
        corner_figure.savefig(f'graphs/corner_plot_{number_of_gen}_gen.png')
        plt.close()

    def plot_error_rate(self, refx, number_of_gen, ngenerations, nchildren, sigmacoeff, points, method):
        # расчет относительной ошибки
        deltarefrerence = np.array(self.x0s) - refx
        # отрисовка
        plt.figure(figsize = (30, 16))
        plt.cla()
        plt.yscale('log')
        plt.grid(True, which="both", linestyle='--')
        # получение подписей для графиков
        legends_list = []
        for i in recoverable_params_indexes:
            legends_list.append(names_of_ParmLocal[i])
        for i, err in enumerate(deltarefrerence.T):
            plt.plot(np.abs(err/refx[i]), label = f"Параметр {i+1} - {legends_list[i]}", linewidth = 6)
            print(err/refx[i])
        plt.xlabel("Поколение", fontsize=28)
        plt.ylabel("Относительная ошибка", fontsize=28)
        plt.legend()
        # plt.ylim(1e-4, 1e1)
        plt.tight_layout()
        plt.savefig(f'error_rate_{number_of_gen}_gen_$_freqs = {len(freqs)}_$_ngenerations = {ngenerations}, nchildren = {nchildren}, sigmacoeff = {sigmacoeff}, point = {points}, method = {method}.png')
        plt.close()

    def functional_plot(self, functional_array, number_of_gen):
        plt.figure(figsize = (30, 16))
        plt.cla()
        plt.plot(range(len(functional_array)), functional_array, linewidth = 10)
        plt.grid(True, which="both", linestyle='--')
        plt.yscale('log')
        plt.xlabel("Поколение", fontsize=32)
        plt.ylabel("Значение функционала", fontsize=32)
        plt.tight_layout()
        plt.savefig(f'graphs/functional_plot_{number_of_gen}.png')
        plt.close()

    def plot_spectrum(self, L, R, number_of_gen):
        # подсчет спектра по модели
        reference_spectrum = reference
        # отрисовка
        fig1, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(25, 12))
        axs[0].grid(True, which="both", linestyle='--')
        axs[1].grid(True, which="both", linestyle='--')
        for i, intensivity_L in enumerate(L):
            axs[0].plot(space_freqs/1e9, intensivity_L, label = f"Поколение {i+1}", linewidth = 0.75*(i+1))
        for i, intensivity_R in enumerate(R):
            axs[1].plot(space_freqs/1e9, intensivity_R, label = f"Поколение {i+1}", linewidth = 0.75*(i+1))
        axs[0].plot(freqs/1e9, reference_spectrum[0::2],'D', label = f"Реальный спектр", linewidth = 8, color = 'darkblue', markersize=14)
        axs[1].plot(freqs/1e9, reference_spectrum[1::2],'D', label = f"Реальный спектр", linewidth = 8, color = 'darkblue', markersize=14)
        axs[0].set_xlabel("Частота, ГГц", fontsize=32)
        axs[1].set_xlabel("Частота, ГГц", fontsize=32)
        axs[0].set_title("Частотный спектр L - поляризации", fontsize=32)
        axs[1].set_title("Частотный спектр R - поляризации", fontsize=32)
        axs[0].set_ylabel(r"Интенсивность, $sfu$", fontsize=32)
        axs[0].loglog()
        axs[1].loglog()
        axs[0].set_ylim(reference_spectrum.min() - reference_spectrum.min() * 0.2, reference_spectrum.max() + reference_spectrum.max() * 0.2)
        axs[1].set_ylim(reference_spectrum.min() - reference_spectrum.min() * 0.2, reference_spectrum.max() + reference_spectrum.max() * 0.2)
        # axs[0].legend(fontsize=18, ncol=2)
        axs[1].legend(fontsize=18, ncol=3)
        plt.tight_layout()
        plt.savefig(f'graphs/RL_spectrum_{number_of_gen}_gen.png')
        plt.close()
        
        reference_intensity = reference_spectrum[0::2] + reference_spectrum[1::2]
        fig2, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(15, 15))
        ax.grid(True, which="both", linestyle='--')
        for i, intensivity_L in enumerate(L):
            ax.plot(space_freqs/1e9, L[i] + R[i], label = f"Поколение {i+1}", linewidth = 0.75*(i+1))
        ax.plot(freqs/1e9, reference_intensity, 'D', label = f"Реальный спектр", linewidth = 8, color = 'darkblue', markersize=14)
        ax.set_xlabel("Частота, ГГц", fontsize=32)
        ax.set_title("Частотный спектр интенсивности излучения", fontsize=32)
        ax.set_ylabel(r"Интенсивность, $sfu$", fontsize=32)
        ax.loglog()
        ax.set_ylim(reference_intensity.min() - reference_intensity.min() * 0.2, reference_intensity.max() + reference_intensity.max() * 0.2)
        ax.legend(fontsize=18, ncol=2)
        plt.tight_layout()
        plt.savefig(f'graphs/I_spectrum_{number_of_gen}_gen.png')
        plt.close()
        
    def analise_plot(self, functional, params, number_of_gen, gen_points = None):
        functional = np.array(functional)
        params = np.array(params)
        # Ограничение количества отображаемых точек
        if gen_points == None:
            num_points = len(functional)
        else:
            num_points = gen_points  # желаемое количество точек

        params = params[:num_points]
        functional = functional[:num_points]
        # Получение количества подэлементов в каждом элементе params
        num_subplots = len(params[0])

        # Вычисление количества строк и столбцов в субплотах
        num_cols = min(num_subplots, 4)  # Максимальное количество столбцов 
        num_rows = -(-num_subplots // num_cols)  # Округление вверх для количества строк

        # Создание субплотов в виде решетки
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(25, 5*num_rows))

        # Построение графиков для каждого подэлемента в params
        for i, ax in enumerate(axs.flat):
            if i < num_subplots:
                # График значения величины из params
                ax.plot(params[:, i], color = 'k', linewidth = 3)
                ax.set_ylabel(names_of_ParmLocal[recoverable_params_indexes[i]])

                # Создание второй координатной сетки для functional
                ax2 = ax.twinx()
                # График из functional
                ax2.plot(functional, color='r', label='Functional', linewidth = 3)
                ax2.set_ylabel('Functional', color='r')
                ax2.tick_params(axis='y', labelcolor='r')

        # Сохранение графиков
        plt.tight_layout()
        plt.savefig(f'graphs/analise_graph_{number_of_gen}_gen.png')
        plt.close()
    
    ######################################
    # функция генерации
    # (количество поколений, количество потомков, коэфиициент изменения ширины генерации, количество точек на ребенка)
    def Generating(self, ngenerations, nchildren, sigmacoeff, points, method, do_plot = False, continuation = False, continuation_gen = None, refx = None):
        
        spectrum_L, spectrum_R = [], []
        # массив хранения функционалов
        functional_array = [] 
        array_rec_params = []
        
        if continuation == True:
            backup = np.load('diagnostic_backup.npz')
            functional_array = list(backup['functional'])
            spectrum_L, spectrum_R = list(backup['spectrum_L']), list(backup['spectrum_R'])
            array_rec_params = list(backup['params'])
            if continuation_gen != None:
                self.gen = continuation_gen
            else:
                self.gen = backup['ngens']

        # если еще нету нулевого поколения - генерируем точки и значения к ним        
        if not self.gen:
            
            # удаляем все остатки с прошлого раза, если они есть
            try:
                os.mkdir('dats')
                os.mkdir('graphs')
            except:
                shutil.rmtree('graphs')
                os.mkdir('graphs')
                # os.system('rm -rf dats/*')
                shutil.rmtree('dats')
                os.mkdir('dats')
            
            self.x0s.append(self.x0)
            self.generate(points * nchildren * 10, method)
            try:
                self.corner_plot(self.x, self.r, 0)
            except Exception as err: 
                logprint(f'Ошибка построения corner: {err}')
            y_min = self.get('y', self.getmins(1))[0]
            spectrum_L.append(y_min[:Nf])
            spectrum_R.append(y_min[Nf:])
            self.plot_spectrum(spectrum_L, spectrum_R, self.gen)
            
            best = self.get('x', self.getmins(1))[0]
            logprint([", ".join(best.astype(str))])
            array_rec_params.append(best)
            functional_array.append(self.get('r', self.getmins(1)))
            self.analise_plot(functional_array, array_rec_params, self.gen)
            
        # переменная счетчик
        gen = self.gen
    
        # цикл генерации n поколений
        # поколоние с которого начали генерировать - текущее поколение, нужно для генерации + поколений к расчитанным

        while self.gen-gen < ngenerations:
            
            start = time.time()

            self.x = np.zeros((0, self.ndim))
            # imins - массив индексов детей по возрастанию
            imins = self.getmins(nchildren)
            # все координаты в прошлой генерации
            all_x = self.get('x')
            # координаты потомков
            xmins = all_x[imins]
            # генерация облака в зависимости от окружающих точек
            # deltas - относительный сдвиг до ближайшей точки для каждого потомка, вектор с началом в одной точке и концом в другой
            deltas = np.zeros((nchildren, self.ndim))
            for i, x in enumerate(xmins):
                # считаем относительное положение (по модулю) ближайших к минимуму точек
                xr = (((all_x - x) / self.sigma) ** 2).sum(1)
                # ну тут первая точка - сама точка, поэтому и расстояние до нее 0, поэтому берем вторую точку
                deltas[i] = np.abs(all_x[xr.argsort()[1]]-x) 
            
            #если типо сошелся то вызываем исключение, ибо нефиг)
            1/int(deltas[0,0])

            # если включена отрисовка графика и есть более двух точек - рисуем
            if do_plot:
                # self.plot_error_rate(refx, self.gen, ngenerations, nchildren, sigmacoeff, points, method)
                self.functional_plot(functional_array, self.gen)

            # увеличиваем счетчик поколений
            self.gen += 1
            # записываем искаемые параметры на этом поколении
            self.x0s.append(xmins[0])
            # расчет новой сигмы - усредненная близость по каждому из параметров к другим точкам с меньшим функционалам * на некий sigmacoeff
            self.sigma = deltas.mean(0)*sigmacoeff

            # генерация "социума" ребенка
            for i, x in enumerate(xmins):
                print(f'{self.gen} gen, {i + 1} chld')
                self.x0 = x
                self.generate(points, method = 'old_gaussian')
            try:
                self.corner_plot(self.x, self.r, self.gen)
            except Exception as err: 
                logprint(f'Ошибка построения corner: {err}')

            # пишем что лучшее вышло на текущем шаге
            best = self.get('x', self.getmins(1))[0]
            logprint([", ".join(best.astype(str))])
            array_rec_params.append(best)
            functional_array.append(self.get('r', self.getmins(1)))
            self.analise_plot(functional_array, array_rec_params, self.gen)

            # сохраняем информацию о восстановленном спектре для анализа
            y_min = self.get('y', self.getmins(1))[0]
            spectrum_L.append(y_min[:Nf])
            spectrum_R.append(y_min[Nf:])

            self.plot_spectrum(spectrum_L, spectrum_R, self.gen)
            self.analise_plot(functional_array, array_rec_params, self.gen)
            np.savez('diagnostic_backup.npz', freqs = np.array(freqs), ngens = np.array(ngenerations), nchild = np.array(nchildren), sigmacoeff = np.array(sigmacoeff), point = np.array(points), method = np.array(method), params=array_rec_params, functional=functional_array, spectrum_L = spectrum_L, spectrum_R = spectrum_R)
            
            # небольшая передышка
            time.sleep(1)
            end = time.time()
            logprint(f"Время поколения - {(end-start)/60} min")

        # результат для последнего поколения        
        self.x0 = self.get('x', self.getmins(1))[0]
        self.x0s.append(self.x0)

        # пишем лучшее, что получилось
        logprint([", ".join(self.x0.astype(str))])
        logprint(f'ngenerations = {ngenerations}, nchildren = {nchildren}, sigmacoeff = {sigmacoeff}, point = {points}')

        # рисуем график
        # if do_plot: 
            # self.plot_error_rate(refx, self.gen, ngenerations, nchildren, sigmacoeff, points, method)
            # self.functional_plot(functional_array, ngenerations, ngenerations, nchildren, sigmacoeff, points, method)
        # plt.show()
        
        
                