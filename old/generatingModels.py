import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
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
        
    def generate(self, points):
        filex = open(f'{self.fname}_gen{self.gen}_x.txt', 'a')
        filey = open(f'{self.fname}_gen{self.gen}_y.txt', 'a')
        filer = open(f'{self.fname}_gen{self.gen}_r.txt', 'a')
        # генерация координат
        # (начальная точка, ширина, (количество точек, размерность))
        x = np.random.normal(self.x0, self.sigma, (points, self.ndim))
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
        # если индексы не заданы возвращает полностью массив
        return np.loadtxt(f'{self.fname}_gen{self.gen}_{ax}.txt')[indexes]
    
    def plot(self, recoverable_parameters):
        # расчет относительной ошибки
        deltarefrerence = np.array(self.x0s) - recoverable_parameters
        # отрисовка
        plt.figure(1, figsize = (14,8))
        plt.cla()
        plt.yscale('log')
        plt.grid(True, which="both", linestyle='--')
        list_params = [r'$n_0$', r'$B$', r'$\theta$', r'$n_e$', r'$\delta_1$']
        for i, err in enumerate(deltarefrerence.T):
            plt.plot(np.abs(err/recoverable_parameters[i]), label = f"Параметр {i+1} - {list_params[i]}")
        plt.xlabel("Поколение", fontsize=20)
        plt.ylabel("Относительная ошибка", fontsize=20)
        plt.legend()
        plt.tight_layout()
        plt.ylim(1e-6, 1e1)
        plt.pause(1)

    
    # функция генерации
    # (количество поколений, количество потомков, коэфиициент изменения ширины генерации, количество точек на ребенка)
    def Generating(self, ngenerations, nchildren, sigmacoeff, points, do_plot = False, recoverable_parameters = None):
        # если еще нету нулевого поколения - генерируем точки и значения к ним
        if not self.gen: self.generate(points*nchildren)
        # переменная счетчик
        gen = self.gen
        # цикл генерации n поколений
        # поколоние с которого начали генерировать - текущее поколение, нужно для генерации +поколений к расчитанным
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
                # считаем относительное положение (по модулю) ближайших к минимуму точек (радиус)
                xr = (((all_x - x) / self.sigma) ** 2).sum(1)
                # первая точка - сама точка, поэтому и расстояние до нее 0, поэтому берем вторую точку
                deltas[i] = np.abs(all_x[xr.argsort()[1]]-x) 
            # увеличиваем счетчик поколений
            self.gen+=1
            # записываем искаемые параметры на этом поколении
            self.x0s.append(xmins[0])
            # расчет новой сигмы - усредненная близость по каждому из параметров к другим точкам с меньшим функционалам * на некий sigmacoeff
            self.sigma = deltas.mean(0)*sigmacoeff
            # генерация социума ребенка
            for i, x in enumerate(xmins):
                print(f'{self.gen + 1} gen, {i + 1} child')
                self.x0 = x
                self.generate(points)
            # если включена отрисовка графика и более двух точек - рисуем
            if do_plot and self.gen > 1: self.plot(recoverable_parameters)

            # пишем что лучшее вышло на текущем шаге
            print(self.get('x',self.getmins(1))[0])
            time.sleep(2)

        # результат для последнего поколения
        self.x0 = self.get('x',self.getmins(1))[0]
        self.x0s.append(self.x0)
        print(self.x0s)
        print(f'ngenerations = {ngenerations}, nchildren = {nchildren}, sigmacoeff = {sigmacoeff}, point = {points}')
        if do_plot: self.plot(recoverable_parameters)
        plt.show()
                