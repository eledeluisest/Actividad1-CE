"""
En este módulo vamos a evaluar los rsultados obtenidos del algoritmo genético para la resolución del TSP

"""
# tasa de éxito
# las curvas de progreso
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

simple_00 = pd.read_csv('data/CE_resultado_100_10000_0.0.csv', sep=';',
                        names=['n_iter', 'n_prom', 'n_inst', 'time', 'maximo', 'minimo', 'media', 'desvstd'],
                        header=None)
simple_00 = simple_00.loc[simple_00.n_prom != simple_00.n_prom.max(), :]

simple_025 = pd.read_csv('data/CE_resultado_100_10000_0.25.csv', sep=';',
                         names=['n_iter', 'n_prom', 'n_inst', 'time', 'maximo', 'minimo', 'media', 'desvstd'],
                         header=None)
simple_025 = simple_025.loc[simple_025.n_prom != simple_025.n_prom.max(), :]

simple_05 = pd.read_csv('data/CE_resultado_100_10000_0.5.csv', sep=';',
                        names=['n_iter', 'n_prom', 'n_inst', 'time', 'maximo', 'minimo', 'media', 'desvstd'],
                        header=None)
simple_05 = simple_05.loc[simple_05.n_prom != simple_05.n_prom.max(), :]

simple_075 = pd.read_csv('data/CE_resultado_100_10000_0.75.csv', sep=';',
                         names=['n_iter', 'n_prom', 'n_inst', 'time', 'maximo', 'minimo', 'media', 'desvstd'],
                         header=None)
simple_075 = simple_075.loc[simple_075.n_prom != simple_075.n_prom.max(), :]

simple_1 = pd.read_csv('data/CE_resultado_100_10000_1.0.csv', sep=';',
                       names=['n_iter', 'n_prom', 'n_inst', 'time', 'maximo', 'minimo', 'media', 'desvstd'],
                       header=None)
simple_1 = simple_1.loc[simple_1.n_prom != simple_1.n_prom.max(), :]

simple_00['sp'] = 0
simple_025['sp'] = 0.25
simple_05['sp'] = 0.5
simple_075['sp'] = 0.75
simple_1['sp'] = 1

total = pd.concat([simple_00.reset_index(),
                   simple_025.reset_index(),
                   simple_05.reset_index(),
                   simple_075.reset_index(),
                   simple_1.reset_index()], axis=0)

total['distancia_min'] = total.maximo.apply(lambda x: 1. / np.exp(float(x.replace(',', '.'))))
total['distancia_max'] = total.minimo.apply(lambda x: 1. / np.exp(float(x.replace(',', '.'))))
total['distancia_med'] = total.media.apply(lambda x: 1. / np.exp(float(x.replace(',', '.'))))

boxplot = pd.DataFrame()
agrupado = total.groupby(['sp', 'n_prom']).distancia_min.count().reset_index().drop('n_prom', axis=1)
boxplot['0.0'] = agrupado.loc[agrupado.sp == 0, 'distancia_min'].reset_index(drop=True)
boxplot['0.25'] = agrupado.loc[agrupado.sp == 0.25, 'distancia_min'].reset_index(drop=True)
boxplot['0.5'] = agrupado.loc[agrupado.sp == 0.5, 'distancia_min'].reset_index(drop=True)
boxplot['0.75'] = agrupado.loc[agrupado.sp == 0.75, 'distancia_min'].reset_index(drop=True)
boxplot['1'] = agrupado.loc[agrupado.sp == 1, 'distancia_min'].reset_index(drop=True)

mejor_solucion = total['distancia_min'].min()
agg2 = total.groupby(['sp', 'n_prom']).distancia_min.min().reset_index().drop_duplicates()
tasa_exito005 = agg2.loc[agg2['distancia_min'] <= mejor_solucion * 1.005, :].groupby(
    'sp').n_prom.count() / (total.groupby('sp').n_prom.max()+1) * 100
tasa_exito010 = agg2.loc[agg2['distancia_min'] <= mejor_solucion * 1.015, :].groupby(
    'sp').n_prom.count() / (total.groupby('sp').n_prom.max()+1) * 100
tasa_exito015 = agg2.loc[agg2['distancia_min'] <= mejor_solucion * 1.05, :].groupby(
    'sp').n_prom.count() / (total.groupby('sp').n_prom.max()+1) * 100
tasa_exito020 = agg2.loc[agg2['distancia_min'] <= mejor_solucion * 1.1, :].groupby(
    'sp').n_prom.count() / (total.groupby('sp').n_prom.max()+1) * 100
tasa_exito025 = agg2.loc[agg2['distancia_min'] <= mejor_solucion * 1.15, :].groupby(
    'sp').n_prom.count() / (total.groupby('sp').n_prom.max()+1) * 100

tasa_exito = pd.DataFrame()
#tasa_exito['Tolerancia 15%'] = tasa_exito025.dropna()
tasa_exito['Tolerancia 5%'] = tasa_exito015.dropna()
tasa_exito['Tolerancia 1.5%'] = tasa_exito010.dropna()
tasa_exito['Tolerancia 0.5%'] = tasa_exito005.dropna()


#tasa_exito['Tolerancia 10%'] = tasa_exito020.dropna()

# Tasa de éxito
plt.figure()
tasa_exito.plot(style='.--')
plt.xlabel('Probabilidad de mutación')
plt.ylabel('% Obtención de la solución óptima')
plt.title('Tasa de éxito')
plt.savefig('img/tasa_exito_comp.png')
plt.plot()

# Tiempo vs paso del algoritmo
plt.figure()
total.time.apply(lambda x: float(x.replace(',', '.'))).plot(style='.')
plt.xlabel('Paso del algoritmo')
plt.ylabel('Tiempo de máquina (s)')
plt.title('Relacion entre paso del algoritmo y tiempo')
plt.savefig('img/tiempo_vs_algoritmo_comp.png')
plt.plot()

# Evolucion1

plt.figure()
total.loc[(total.sp == 0.75) & (total.n_prom == 0), ['distancia_min', 'distancia_max', 'distancia_med']].plot()
plt.xlabel('paso del algoritmo')
plt.ylabel('Fitness')
plt.title('Probabilidad de mutación 0.75')
plt.savefig('img/evolucion1_comp.png')
plt.plot()

# Evolución media por sp
fig, ax = plt.subplots(figsize=(8, 6))
total.loc[total.n_prom == 0, ['sp', 'distancia_med']].groupby('sp')['distancia_med'].plot(ax=ax)
plt.xlabel('paso del algoritmo')
plt.ylabel('Fitness')
plt.legend()
plt.title('Evolución del fitness medio en función de la probabilidad de mutación')
plt.savefig('img/evolucion_med_comp.png')
plt.plot()

# Evolución minima por sp
fig, ax = plt.subplots(figsize=(8, 6))
total.loc[total.n_prom == 5, ['sp', 'distancia_min']].groupby('sp')['distancia_min'].plot(ax=ax)
plt.xlabel('paso del algoritmo')
plt.ylabel('Fitness')
plt.legend()
plt.title('Evolución del fitness medio en función de la probabilidad de mutación')
plt.savefig('img/evolucion_min_comp.png')
plt.plot()

# Calculamos la tasa de éxito
plt.figure()
"""
b : blue.
g : green.
r : red.
c : cyan.
m : magenta.
"""
plt.subplot(321)
total.loc[total.sp == 0, :].groupby('n_prom').distancia_min.min(). \
    hist(label="swap prob. = 0.0", color='b')
plt.axvline(x=mejor_solucion, color='k', linestyle='--', label='mejor solucion')
plt.xlim(150, 500)
plt.xlabel('Distancia mínima')
plt.legend()
plt.grid(True)

plt.subplot(322)
total.loc[total.sp == 0.25, :].groupby('n_prom').distancia_min.min(). \
    hist(label="swap prob = 0.25", color='g')
plt.axvline(x=mejor_solucion, color='k', linestyle='--', label='mejor solucion')
plt.xlim(150, 500)
plt.xlabel('Distancia mínima')
plt.legend()
plt.grid(True)

plt.subplot(323)
total.loc[total.sp == 0.5, :].groupby('n_prom').distancia_min.min(). \
    hist(label="swap prob = 0.5", color='r')
plt.axvline(x=mejor_solucion, color='k', linestyle='--', label='mejor solucion')
plt.xlim(150, 500)
plt.xlabel('Distancia mínima')
plt.legend()
plt.grid(True)

plt.subplot(324)
total.loc[total.sp == 0.75, :].groupby('n_prom').distancia_min.min(). \
    hist(label="swap prob = 0.75", color='c')
plt.axvline(x=mejor_solucion, color='k', linestyle='--', label='mejor solucion')
plt.xlim(150, 500)
plt.xlabel('Distancia mínima')
plt.legend()
plt.grid(True)

plt.subplot(325)
total.loc[total.sp == 1, :].groupby('n_prom').distancia_min.min(). \
    hist(label="swap prob = 1.0", color='m')
plt.axvline(x=mejor_solucion, color='k', linestyle='--', label='mejor solucion')
plt.xlim(150, 500)
plt.xlabel('Distancia mínima')
plt.legend()
plt.grid(True)

plt.subplot(326)
boxplot.boxplot()
plt.xlabel('Probabilidad de mutación')
plt.grid(True)

# Format the minor tick labels of the y-axis into empty strings with
# `NullFormatter`, to avoid cumbering the axis with too many labels.
plt.gca().yaxis.set_minor_formatter(NullFormatter())
# Adjust the subplot layout, because the logit one may take more space
# than usual, due to y-tick labels like "1 - 10^{-3}"
plt.subplots_adjust(top=0.92, bottom=0.12, left=0.10, right=0.95, hspace=0.25, wspace=0.75)
plt.savefig('img/probabilidad_acierto_comp.png')
plt.show()
