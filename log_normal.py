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