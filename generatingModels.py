import numpy as np
import matplotlib.pyplot as plt
import time

import matplotlib
matplotlib.rcParams.update({'font.size': 16})

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
        
    def generate(self, points, method):
        filex = open(f'{self.fname}_gen{self.gen}_x.txt', 'a')
        filey = open(f'{self.fname}_gen{self.gen}_y.txt', 'a')
        filer = open(f'{self.fname}_gen{self.gen}_r.txt', 'a')
        # генерация координат
        if not method or method == 'gaussian':
            # (начальная точка, ширина, (количество точек, размерность))
            x = np.random.normal(self.x0, self.sigma, (points, self.ndim))
            # (точка слева, точка справа, (количество точек, размерность))
        elif method == 'random':
            x = np.random.uniform(self.x0 - self.sigma, self.x0 + self.sigma, (points, self.ndim))
        # удаление отрицательных координат
        # ищем минимальную координату в каждом наборе координат
        mask = x.min(1) > 0
        # удаление из массива точек с отрицательными координатами
        x = x[mask]
        # транспонируем - так удобнее потом
        y = self.func(x.T)
        # расчет отклонения от истинных параметров - функционал
        r = self.minif(y)
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
    
    def plot(self, refx):
        # расчет относительной ошибки
        deltarefrerence = np.array(self.x0s) - refx
        # отрисовка
        plt.figure(1, figsize = (12,8))
        plt.cla()
        plt.yscale('log')
        plt.grid(True, which="both", linestyle='--')

        # list_params = [r'$n_0$', r'$B$', r'$\theta$', r'$n_e$', r'$\delta_1$']
        # for i, err in enumerate(deltarefrerence.T):
        #     plt.plot(np.abs(err/refx[i]), label = f"Параметр {i+1} - {list_params[i]}")

        for i, err in enumerate(deltarefrerence.T):
            plt.plot(np.abs(err/refx[i]), label = f"параметр {i+1}")
        plt.xlabel("Поколение", fontsize=20)
        plt.ylabel("Относительная ошибка", fontsize=20)
        plt.legend()
        # plt.ylim(1e-6, 1e1)
        plt.tight_layout()
        plt.pause(1)
    
    # функция генерации
    # (количество поколений, количество потомков, коэфиициент изменения ширины генерации, количество точек на ребенка)
    def Generating(self, ngenerations, nchildren, sigmacoeff, points, method, do_plot = False, refx = None):
        # если еще нету нулевого поколения - генерируем точки и значения к ним
        if not self.gen: self.generate(points * nchildren, method)
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
            self.gen+=1
            # записываем искаемые параметры на этом поколении
            self.x0s.append(xmins[0])
            # расчет новой сигмы - усредненная близость по каждому из параметров к другим точкам с меньшим функционалам * на некий sigmacoeff
            self.sigma = deltas.mean(0)*sigmacoeff
            # генерация "социума" ребенка
            for i, x in enumerate(xmins):
                print(f'{self.gen}gen, {i}chld')
                self.x0 = x
                self.generate(points, method)
            # если включена отрисовка графика и есть более двух точек - рисуем
            if do_plot and self.gen > 1: self.plot(refx)
            # пишем что лучшее вышло на текущем шаге
            print(self.get('x',self.getmins(1))[0])
            # небольшая передышка
            time.sleep(4)

        # результат для последнего поколения        
        self.x0 = self.get('x', self.getmins(1))[0]
        self.x0s.append(self.x0)

        # пишем лучшее, что получилось
        print(self.x0s)
        print(f'ngenerations = {ngenerations}, nchildren = {nchildren}, sigmacoeff = {sigmacoeff}, point = {points}')

        # рисуем график
        if do_plot: self.plot(refx)
        plt.show()
                