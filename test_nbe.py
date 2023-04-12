# import numpy as np
# import matplotlib.pyplot as plt

# mu = 0
# sigma = 1

# data = np.random.lognormal(mu, sigma, 10000)

# # Определение гистограммы распределения
# count, bins, ignored = plt.hist(data, 100, density=True, align='mid')

# # Определение функции плотности вероятности логарифмического гауссового распределения
# pdf = (np.exp(-(np.log(bins) - mu)**2 / (2 * sigma**2))
#        / (bins * sigma * np.sqrt(2 * np.pi)))

# # Построение графика распределения и функции плотности вероятности
# plt.plot(bins, pdf, linewidth=2, color='r')

# # Установка логарифмической шкалы на оси x
# plt.xscale('log')

# # Отображение графика
# plt.show()


# import numpy as np

# mu = 0
# sigma = 1

# # Генерация трехмерного массива с размером 100x100x100
# data = np.random.lognormal(mu, sigma, (100, 3))

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # Создание фигуры и трехмерной оси
# fig = plt.figure()

# # Обычные оси
# ax = fig.add_subplot(121, projection='3d')
# ax.scatter(data[:, :, 0], data[:, :, 1], data[:, :, 2], s=1)

# # Логарифмические оси
# ax = fig.add_subplot(122, projection='3d')
# ax.scatter(np.log(data[:, :, 0]), np.log(data[:, :, 1]), np.log(data[:, :, 2]), s=1)

# # Показ графика
# plt.show()


import numpy as np
import corner
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Заданные sigma и mu для каждой из осей
sigmas = [0.5, 0.5, 0.5]
mus = [0, 0, 0]

# Создаем логарифмические распределения по первой и четвертой оси
dist_lognorm = [np.random.lognormal(mean=mus[i], sigma=sigmas[i], size=5000) if i == 0 or i == 3 else np.random.normal(loc=mus[i], scale=sigmas[i], size=5000) for i in range(3)]

# Объединяем все распределения в один
dist = np.column_stack(dist_lognorm)

# Проверяем размерность полученного массива
# print(dist) # (10000, 5)

# Создаем 3D график
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Отображаем распределение
ax.scatter(dist[:,0], dist[:,1], dist[:,2], c='r', marker='o')

# Устанавливаем названия осей
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# Показываем график
plt.show()

corner.corner(data=dist, axes_scale=['log', 'linear', 'linear'], title_fmt = None, titles = ['X Label', 'Y Label', 'Z Label'], show_titles = True)
plt.show()