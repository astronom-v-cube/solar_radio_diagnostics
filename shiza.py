# import numpy as np

# # Интервал [a, b]
# a = 0
# b = 10

# # Параметры нормального распределения
# x0 = [5, 5, 5]
# sigma = 4

# # Размерность массива
# points = 15
# ndim = 3

# # Генерация массива случайных чисел, распределенных по нормальному закону
# z = np.random.normal(x0, sigma, (points, ndim))

# # Преобразование Бокса-Мюллера
# u1 = np.random.rand(points, ndim)
# u2 = np.random.rand(points, ndim)
# z1 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
# z2 = np.sqrt(-2 * np.log(u1)) * np.sin(2 * np.pi * u2)

# # Масштабирование массива
# mean = np.mean(z)
# std = np.std(z)
# x = (z - mean) / std
# x = (x * (b - a) / 2) + ((b + a) / 2)

# print(x)
# print(z)

import numpy as np

# задать параметры
x0 = 5  # среднее значение
sigma = 4  # стандартное отклонение
points = 3  # количество точек
ndim = 3  # размерность

# задать границы интервала
a = 0
b = 10

# сгенерировать массив случайных чисел
arr = np.random.normal(x0, sigma, (points, ndim))
print(arr)

# преобразовать значения в нужный интервал
arr = (b - a) * (arr - arr.min()) / (arr.max() - arr.min()) + a

# вывести результат
print(arr)



