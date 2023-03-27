import numpy as np
import corner
import matplotlib.pyplot as plt
import time
from params import freqs, ParmLocal, Lparms, Rparms, NSteps, Nf, recoverable_params,  recoverable_params_indexes, limits_of_gen_ParmLocal, names_of_ParmLocal
from utils import Calc_I
import matplotlib
matplotlib.rcParams.update({'font.size': 25})

class generatingModels:
    #calcfunc(x), x=[x0,x1,x2,...,x(dim-1)]
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
        
    def corner_plot(self, x, r, number_of_gen, ngenerations, nchildren, sigmacoeff, points, method):

        # получение координат истинной точки для отображения
        truths = []
        for i in recoverable_params_indexes:
            truths.append(ParmLocal[i])
        # получение интервалов генерации точек для отображения
        ranges = []
        for i in recoverable_params_indexes:
            # переводим в формат int
            one_range = tuple(map(int, limits_of_gen_ParmLocal[i]))
            ranges.append(one_range) 
        print(ranges)
        # получение подписей для графиков
        titles = []
        for i in recoverable_params_indexes:
            titles.append(names_of_ParmLocal[i]) 

        corner_figure = plt.figure(figsize=(20, 20))
        corner.corner(data = x, weights = (1/r).ravel(), titles = titles, fig = corner_figure, truths = truths, title_fmt = None, show_titles = True, range = ranges) 
        corner_figure.tight_layout()
        # , plot_datapoints=False
        corner_figure.savefig(f'corner_plot_{number_of_gen}_gen_$_len.freqs = {len(freqs)}_$_ngenerations = {ngenerations}, nchildren = {nchildren}, sigmacoeff = {sigmacoeff}, point = {points}, method = {method}.png')

    def generate(self, points, method):
        filex = open(f'{self.fname}_gen{self.gen}_x.txt', 'a')
        filey = open(f'{self.fname}_gen{self.gen}_y.txt', 'a')
        filer = open(f'{self.fname}_gen{self.gen}_r.txt', 'a')

        # генерация координат
        if not method or method == 'gaussian':
            # (начальная точка, ширина, (количество точек, размерность))
            x = np.random.normal(self.x0, self.sigma, (points, self.ndim))

        elif method == 'random':
            # (точка слева, точка справа, (количество точек, размерность))
            x = np.random.uniform(self.x0 - self.sigma, self.x0 + self.sigma, (points, self.ndim))

        elif method == 'new_random_first_gen':
            # (точка слева, точка справа, (количество точек, размерность))
            # выходной массив
            x = []
            # для всех точек по их количеству
            for i in range(points):
                # массив второго уровня хранящий одну точку
                one_point = []
                # проходимся по индексам и берем границы генерации 
                for k in recoverable_params_indexes:
                    num = np.random.uniform(limits_of_gen_ParmLocal[k][0], limits_of_gen_ParmLocal[k][1])
                    # num_1 = np.random.uniform(limits_of_gen_ParmLocal[k][0], limits_of_gen_ParmLocal[k][1])
                    # num_2 = None
                    # np.random.randint(2, 8)
                    one_point.append(num)
                x.append(one_point)
            x = np.array(x)

        # вбрасываем немного случайных координат вне зависимости от области генерации

        x_additional = []
        # для всех точек по их количеству
        for i in range(points//4):
            # массив второго уровня хранящий одну точку
            one_point = []
            # проходимся по индексам и берем границы генерации 
            for k in recoverable_params_indexes:
                num = np.random.uniform(limits_of_gen_ParmLocal[k][0], limits_of_gen_ParmLocal[k][1])
                one_point.append(num)
            x_additional.append(one_point)
        x_additional = np.array(x_additional)
        x = np.concatenate((x, x_additional), axis=0)
        # np.append(x, [6e+09, 1.80e+02, 8.0e+01, 1.e+06])
        # удаление отрицательных координат
        # ищем отрицательную координату в каждом наборе координат
        mask = x.min(1) > 0
        # удаление из массива точек с отрицательными координатами
        x = x[mask]
        self.x = x
        # транспонируем - так удобнее потом
        y = self.func(x.T)
        self.y = y
        # расчет отклонения от истинных параметров - функционал
        r = self.minif(y)
        self.r = r

        # координаты
        np.savetxt(filex,x)
        # интенсивности
        np.savetxt(filey,y)
        # значения функционалов
        np.savetxt(filer,r)
        filex.close()
        filey.close()
        filer.close()

    # n - количество точек, находит индексы n минимальных
    def getmins(self, n):
        r = np.loadtxt(f'{self.fname}_gen{self.gen}_r.txt')
        return r.argsort()[:n]
    
    def get(self, ax, indexes = None):
        # если индексы не заданы возвращает полностью массив
        if isinstance(indexes, type(None)): return np.loadtxt(f'{self.fname}_gen{self.gen}_{ax}.txt')
        # если индексы заданы возвращает по ним
        return np.loadtxt(f'{self.fname}_gen{self.gen}_{ax}.txt')[indexes]
    
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
        plt.xlabel("Поколение", fontsize=28)
        plt.ylabel("Относительная ошибка", fontsize=28)
        plt.legend()
        # plt.ylim(1e-4, 1e1)
        plt.tight_layout()
        plt.savefig(f'error_rate_{number_of_gen}_gen_$_freqs = {len(freqs)}_$_ngenerations = {ngenerations}, nchildren = {nchildren}, sigmacoeff = {sigmacoeff}, point = {points}, method = {method}.png')

    def plot_spectrum(self, L, R, freqs, number_of_gen, ngenerations, nchildren, sigmacoeff, points, method):
        # подсчет спектра по модели
        model_spectrum = Calc_I(freqs, recoverable_params, recoverable_params_indexes, ParmLocal, Lparms, Rparms, NSteps, Nf)[:,5:].ravel()
        # отрисовка
        fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(25, 12))
        axs[0].grid(True, which="both", linestyle='--')
        axs[1].grid(True, which="both", linestyle='--')
        # list_params = [r'$n_0$', r'$B$', r'$\theta$', r'$n_e$', r'$\delta_1$']
        # for i, err in enumerate(deltarefrerence.T):
        #     plt.plot(np.abs(err/refx[i]), label = f"Параметр {i+1} - {list_params[i]}")

        for i, intensivity_L in enumerate(L):
            axs[0].plot(freqs, intensivity_L, label = f"Спектр на поколении {i+1}", linewidth = 5*(i+1)*0.3)
        for i, intensivity_R in enumerate(R):
            axs[1].plot(freqs, intensivity_R, label = f"Спектр на поколении {i+1}", linewidth = 5*(i+1)*0.3)
        axs[0].plot(freqs, model_spectrum[0::2], label = f"Реальный спектр", linewidth = 8, linestyle = ':', color = 'darkblue')
        axs[1].plot(freqs, model_spectrum[1::2], label = f"Реальный спектр", linewidth = 8, linestyle = ':', color = 'darkblue')
        axs[0].set_xlabel("Частота", fontsize=28)
        axs[1].set_xlabel("Частота", fontsize=28)
        axs[0].set_title("Левая поляризация", fontsize=28)
        axs[1].set_title("Правая поляризация", fontsize=28)
        axs[0].set_ylabel(r"Интенсивность, $sfu$", fontsize=28)
        axs[0].legend(fontsize=20)
        axs[1].legend(fontsize=20)
        plt.tight_layout()
        plt.savefig(f'spectrum_{number_of_gen}_gen_$_freqs = {len(freqs)}_$_ngenerations = {ngenerations}, nchildren = {nchildren}, sigmacoeff = {sigmacoeff}, point = {points}, method = {method}.png')
    
    # функция генерации
    # (количество поколений, количество потомков, коэфиициент изменения ширины генерации, количество точек на ребенка)
    def Generating(self, ngenerations, nchildren, sigmacoeff, points, method, do_plot = False, refx = None, number = None):

        spectrum_L, spectrum_R = [], []

        # если еще нету нулевого поколения - генерируем точки и значения к ним
        if not self.gen: 
            self.generate(points * nchildren, method)
            self.corner_plot(self.x, self.r, 0, ngenerations, nchildren, sigmacoeff, points, method)
            spectrum_L.append(list(self.get('y', self.getmins(1))[0][0::2]))
            spectrum_R.append(list(self.get('y', self.getmins(1))[0][1::2]))
            self.plot_spectrum(spectrum_L, spectrum_R, freqs, self.gen + 1, ngenerations, nchildren, sigmacoeff, points, method)
            
        # переменная счетчик
        gen = self.gen
        # цикл генерации n поколений
        # поколоние с которого начали генерировать - текущее поколение, нужно для генерации + поколений к расчитанным

        while self.gen-gen < ngenerations:
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
                self.generate(points, method = 'gaussian')
            self.corner_plot(self.x, self.r, self.gen, ngenerations, nchildren, sigmacoeff, points, method)

            # если включена отрисовка графика и есть более двух точек - рисуем
            if do_plot and self.gen > 1: 
                self.plot_error_rate(refx, self.gen, ngenerations, nchildren, sigmacoeff, points, method)
                self.plot_spectrum(spectrum_L, spectrum_R, freqs, self.gen, ngenerations, nchildren, sigmacoeff, points, method)
            # пишем что лучшее вышло на текущем шаге
            print(self.get('x', self.getmins(1))[0])
            # сохраняем информацию о восстановленном спектре для анализа
            spectrum_L.append(list(self.get('y', self.getmins(1))[0][0::2]))
            spectrum_R.append(list(self.get('y', self.getmins(1))[0][1::2]))
            # небольшая передышка
            time.sleep(4)

        # результат для последнего поколения        
        self.x0 = self.get('x', self.getmins(1))[0]
        self.x0s.append(self.x0)

        # пишем лучшее, что получилось
        print(self.x0)
        print(f'ngenerations = {ngenerations}, nchildren = {nchildren}, sigmacoeff = {sigmacoeff}, point = {points}')

        # рисуем график
        if do_plot: self.plot_error_rate(refx, self.gen, ngenerations, nchildren, sigmacoeff, points, method)
        # plt.show()
                