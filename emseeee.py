# import matplotlib.pyplot as plt
# import numpy as np
# import emcee
# import corner

# # Определяем функцию, которую хотим минимизировать
# def log_likelihood(theta):
#     x, y = theta
#     return 0.5 * (x**2 + y**2) + 2  # Пример функции: просто сумма квадратов

# # Определяем априорное распределение (в данном случае - равномерное)
# def log_prior(theta):
#     x, y = theta
#     if -10.0 < x < 10.0 and -10.0 < y < 10.0:
#         return 0.0
#     return -np.inf

# # Полное логарифмическое правдоподобие
# def log_probability(theta):
#     lp = log_prior(theta)
#     if not np.isfinite(lp):
#         return -np.inf
#     return lp - log_likelihood(theta)

# # Настройка MCMC
# ndim = 2  # Количество измерений (в данном случае, два параметра x и y)
# nwalkers = 128  # Количество ходячих
# nsteps = 10000  # Количество шагов в MCMC

# # Инициализация начальных точек
# initial = np.random.randn(nwalkers, ndim)

# # Инициализация ходячих
# sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, threads=12)

# # Запуск MCMC
# sampler.run_mcmc(initial, nsteps, progress=True)

# # Получение цепей Марковских
# samples = sampler.get_chain(discard=100, thin=15, flat=True)

# # Найдем минимум в цепи Марковских
# best_fit_index = np.argmax(sampler.flatlnprobability)
# best_fit_params = sampler.flatchain[best_fit_index]

# # Построение графика с использованием библиотеки corner
# fig = corner.corner(samples, labels=['x', 'y'], truths=best_fit_params, plot_contours=True, fill_contours=True, levels=[0.2, 0.5, 0.8, 0.95], color='k', sharex = True)

# plt.show()

# plt.scatter(samples[:, 0], samples[:, 1])
# plt.show()


import matplotlib.pyplot as plt
import numpy as np
import emcee
import corner

# Определяем функцию, которую хотим минимизировать
def log_likelihood(theta):
    x, y, z = theta
    return (x**2 + y**2 + z**2)  # Пример функции: просто сумма квадратов

# Определяем априорное распределение (в данном случае - равномерное)
def log_prior(theta):
    x, y, z = theta
    if -10.0 < x < 10.0 and -10.0 < y < 10.0 and -10.0 < z < 10.0:
        return 0.0
    return -np.inf

# Полное логарифмическое правдоподобие
def log_probability(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp - log_likelihood(theta)

# Настройка MCMC
ndim = 3  # Количество измерений (в данном случае, два параметра x и y)
nwalkers = 512  # Количество ходячих
nsteps = 10000  # Количество шагов в MCMC

# Инициализация начальных точек
initial = np.random.randn(nwalkers, ndim)

# Инициализация ходячих
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, threads=12)

# Запуск MCMC
sampler.run_mcmc(initial, nsteps, progress=True)

# Получение Марковских цепей
samples = sampler.get_chain(discard=100, thin=15, flat=True)

# Найдем минимум в цепи Марковских
best_fit_index = np.argmax(sampler.flatlnprobability)
best_fit_params = sampler.flatchain[best_fit_index]
print(best_fit_params)

# Построение графика с использованием библиотеки corner
fig = corner.corner(samples, labels=['x', 'y', 'z'], truths=best_fit_params, plot_contours=True, fill_contours=True, levels=[0.5, 0.75, 0.9, 0.999], color='k', sharex = True)
plt.show()
