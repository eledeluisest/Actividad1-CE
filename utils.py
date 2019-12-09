"""
Generamos clases para cada una de las fases del EA

Dependencias:
import time
import random
import os
import numpy as np
import warnings

"""
import time
import random
import os
import numpy as np
import warnings


class generador_ejemplos:
    """
    Esta clase genera ejmplos de problemas para la optimización con Algoritmos genéticos u otros
    """
    def __init__(self, problema, seed_gen=None, seed_ea=None, data_path=None):
        self.__problema__ = problema
        self.__seed_gen__ = seed_gen
        self.__seed_ea__ = seed_ea
        print('Se generaran instancias del problema: ' + problema)
        if type(data_path) is type(None):
            self.__problema_path__ = 'data/CE_instancia_' + problema + '_' + str(time.time()) + '.csv'
            if not os.path.exists('data/'):
                os.makedirs('data/')
        else:
            self.__problema_path__ = data_path
        if problema == 'VIAJANTE':
            print(
                'Utilizar metodos viajante_ini para fijar los parametros y genera instancia para generar las instancias.')

    def __viajante_gen__(self):
        puntos = []
        p = []
        if type(self.__seed_gen__) is not type(None):
            random.seed = self.__seed_gen__
        for n in range(self.__NPuntos__):
            for limite in self.__limites__:
                p.append(str(random.uniform(0, limite)))
            puntos.append(p)
            p = []
        return ';'.join([','.join(x) for x in puntos]) + '\n'

    def viajante_ini(self, n, lista_lim):
        self.__NPuntos__ = n
        self.__dimension__ = len(lista_lim)
        self.__limites__ = lista_lim

    def __args__(self):
        print(' Estamos ante el problema ' + self.__problema__ + ' con: ')
        print('Puntos: ' + self.__NPuntos__)
        print('Dimension: ' + self.__dimension__)
        print('Limites: ')
        [print(l) for l in self.__limites__]

    def genera_instancias(self, n, mode='file'):
        if mode == 'file':
            with open(self.__problema_path__, 'w+') as f:
                for i in range(n):
                    if self.__problema__ == 'VIAJANTE':
                        # print(self.__viajante_gen__())
                        f.write(self.__viajante_gen__())


class algoritmo_genetico():
    def __init__(self):
        print("Implementacion de un algoritmo genetico")
        self.__tiempos__ = []
        self.__tiempos__.append(time.time())
        self.__minimos__ = []
        self.__maximos__ = []
        self.__medias__ = []
        self.__stds__ = []

        self.__fit_pob__ = None

    def carga_instancia(self, ruta, sep=';'):
        instancia = []
        with open(ruta, 'r') as f:
            for linea in f.readlines():
                instancia.append(linea.split(sep))
        self.__n_ejemplos__ = len(instancia)
        self.__n_puntos__ = len(instancia[0])
        self.__instancias__ = instancia

    def elige_instancia(self, indice_instancia=0):
        self.__indice_instancia__ = indice_instancia
        self.__instancia__ = self.__instancias__[self.__indice_instancia__]

    def codifica_fenotipo_viajante(self, sep_punt=','):
        fenotipo = [x.split(sep_punt) for x in self.__instancia__]
        self.__fenotipo__ = {}
        self.__permutaciones__ = []
        i = 0
        for x, y in fenotipo:
            self.__fenotipo__[i] = [float(x), float(y)]
            self.__permutaciones__.append(i)
            i = i + 1

    def genera_poblacion(self, n_pob=1000):
        self.__poblacion__ = []
        self.__n_pob__ = n_pob
        l_tmp = self.__permutaciones__.copy()
        for i in range(self.__n_pob__):
            random.shuffle(l_tmp)
            tmp = l_tmp.copy()
            self.__poblacion__.append(tmp)

    def __fit_distancia__(self, permutacion):
        posiciones = [self.__fenotipo__[x] for x in permutacion]
        distancias = []
        for i in range(len(posiciones)):
            if i != len(posiciones) - 1:
                distancias.append(np.linalg.norm(np.array(posiciones[i]) - np.array(posiciones[i + 1])))
            else:
                distancias.append(np.linalg.norm(np.array(posiciones[i]) - np.array(posiciones[0])))
        return np.log(1.0 / sum(distancias))

    def __torneo__(self, *individuos):
        distancias = []
        for individuo in individuos:
            distancias.append(self.__fit_distancia__(individuo))
        return individuos[distancias.index(max(distancias))]

    def seleccion_parental(self, k):
        self.__padres__ = []
        for i in range(len(self.__poblacion__)):
            # con reemplazamiento
            self.__padres__.append(self.__torneo__(*random.choices(self.__poblacion__, k=k)))
            # si no queremos reemplazamiento utilizaremos random.sample()

    def __part_map_cross__(self, padre1, padre2):
        hijo1 = [None] * self.__n_puntos__
        hijo2 = [None] * self.__n_puntos__

        # primero seleccionamos dos numeros aleatroios y colocamos el segmento resultante en el hijo.
        punto_1 = int(random.uniform(0, self.__n_puntos__))
        # print(punto_1)
        punto_2 = int(random.uniform(0, self.__n_puntos__))
        # print(punto_2)
        # Nos aseguramos de coger un segmento y no un único punto igual que el segmento no sea demaisiado largo
        dist = abs(punto_2 - punto_1)
        # print(dist)
        while (dist <= 2) or ((punto_1 in [0, self.__n_puntos__ - 1]) and (punto_2 in [0, self.__n_puntos__ - 1])):
            punto_2 = int(random.uniform(0, self.__n_puntos__))
            punto_1 = int(random.uniform(0, self.__n_puntos__))
            dist = abs(punto_2 - punto_1)
        #Definimos el segmento que será común para el hijo 1 y el hijo 2
        derecha = max(punto_2, punto_1)
        izquierda = min(punto_2, punto_1)
        index_hijo1 = [i for i in range(izquierda, derecha, 1)]
        index_hijo2 = [i for i in range(izquierda, derecha, 1)]

        index_res1 = [i for i in range(izquierda)]
        index_res1.extend([i for i in range(derecha, self.__n_puntos__)])

        index_res2 = [i for i in range(izquierda)]
        index_res2.extend([i for i in range(derecha, self.__n_puntos__)])

        index_segmento = [i for i in range(izquierda, derecha, 1)]
        val_segmento1 = []
        val_segmento2 = []

        # Hasta aqui index_hijo1 es igual que index_hijo2 que es el segmento
        for i in index_segmento:
            hijo1[i] = padre1[i]
            val_segmento1.append(padre1[i])

            hijo2[i] = padre2[i]
            val_segmento2.append(padre2[i])
        # Recorremos los elementos del segmento para ambos descendientes a la vez
        for i in index_segmento:
            # print(val_segmento, padre2[i], padre2.index(hijo[i]), index_hijo)
            if padre2[i] not in val_segmento1:
                i_tmp = i
                # Para el hijo 1
                while padre2.index(hijo1[i_tmp]) in index_hijo1:
                    # print(i_tmp, padre2[i_tmp], hijo[i_tmp], padre2.index(hijo[i_tmp]))
                    i_tmp = padre2.index(hijo1[i_tmp])
                hijo1[padre2.index(hijo1[i_tmp])] = padre2[i]
                index_hijo1.append(padre2.index(hijo1[i_tmp]))
                try:
                    index_res1.pop(index_res1.index(padre2.index(hijo1[i_tmp])))
                except:
                    warnings.warn('Actualizamos segmento interior')

                # Para el hijo 2
                if padre1[i] not in val_segmento2:
                    i_tmp = i
                    while padre1.index(hijo2[i_tmp]) in index_hijo2:
                        # print(i_tmp, padre2[i_tmp], hijo[i_tmp], padre2.index(hijo[i_tmp]))
                        i_tmp = padre1.index(hijo2[i_tmp])
                    hijo2[padre1.index(hijo2[i_tmp])] = padre1[i]
                    index_hijo2.append(padre1.index(hijo2[i_tmp]))
                    try:
                        index_res2.pop(index_res2.index(padre1.index(hijo2[i_tmp])))
                    except:
                        warnings.warn('Actualizamos segmento interior')
        # rellenamos el hijo con los valores del padre 2
        for i in index_res1:
            hijo1[i] = padre1[i]
            index_hijo1.append(i)

        for i in index_res2:
            hijo2[i] = padre2[i]
            index_hijo2.append(i)
        """
        # Parta para una ejecucion que compruebe la integridad en cada paso
        if len(hijo1) == self.__n_puntos__ and np.mean(self.__permutaciones__) == np.mean(hijo1)\
                and len(hijo2) == self.__n_puntos__ and np.mean(self.__permutaciones__) == np.mean(hijo2):
            return hijo1, hijo2
        else:
            # print(punto_1, punto_2, index_hijo1, index_res1, index_segmento, padre1, padre2, hijo1)
            # print(punto_1, punto_2, index_hijo2, index_res2, index_segmento, padre1, padre2, hijo2)
            print(len(hijo1), np.mean(hijo1))
            print(len(hijo2), np.mean(hijo2))
            raise AttributeError('El hijo no tiene la longitud correcta')
        """
        return hijo1, hijo2

    def __swap_mutation__(self, individuo, swap_prob):
        self.__swap_prob__ = swap_prob
        if swap_prob > 1:
            raise OverflowError('La probabilidad no puede tener valores mayores que 1')

        if random.uniform(0, 1) <= swap_prob:
            punto_1 = int(random.uniform(0, self.__n_puntos__))
            punto_2 = int(random.uniform(0, self.__n_puntos__))

            tmp = individuo[punto_1]
            individuo[punto_1] = individuo[punto_2]
            individuo[punto_2] = tmp

        return individuo

    def __genera_descenencia__(self, swap_prob):
        self.__descendencia__ = []
        for i in (range(self.__n_pob__ // 2)):
            padre1, padre2 = random.choices(self.__padres__, k=2)
            hijo1, hijo2 = self.__part_map_cross__(padre1, padre2)
            self.__descendencia__.append(self.__swap_mutation__(hijo1, swap_prob=swap_prob))
            self.__descendencia__.append(self.__swap_mutation__(hijo2, swap_prob=swap_prob))
        if self.__n_pob__ % 2 != 0:
            padre1, padre2 = random.choices(self.__padres__, k=2)
            self.__descendencia__.append(self.__swap_mutation__(self.__part_map_cross__(padre1, padre2)[0], swap_prob=swap_prob))

        """
        Comprobacion de integridad
        if len(self.__descendencia__) != self.__n_pob__:
            raise AttributeError('La descendencia tiene un número de individuos diferene a la población')
        """
    def modelo_generacional(self, swap_prob=0.4, elitismo=True):
        # Generamos la descendencia
        self.__genera_descenencia__(swap_prob=swap_prob)

        # Actualizamos el fitness de la población actual evitando reprocesar
        if type(self.__fit_pob__) is type(None):
            self.__fit_pob__ = [self.__fit_distancia__(x) for x in self.__poblacion__]
        else:
            self.__fit_pob__ = self.__fit_desc__

        self.__fit_desc__ = [self.__fit_distancia__(x) for x in self.__descendencia__]
        # Prácticamos el elitismo si procede
        print(np.mean([np.mean(x) for x in self.__poblacion__]))
        if elitismo:
            self.__descendencia__[self.__fit_desc__.index(min(self.__fit_desc__))] = \
                self.__poblacion__[self.__fit_pob__.index(max(self.__fit_pob__))]

            self.__fit_desc__[self.__fit_desc__.index(min(self.__fit_desc__))] = \
                self.__fit_pob__[self.__fit_pob__.index(max(self.__fit_pob__))]

        # Actualizamos la población

        self.__poblacion__ = self.__descendencia__.copy()
        print(np.mean([np.mean(x) for x in self.__poblacion__]))

    def __metricas__(self):
        self.__tiempos__.append(time.time())
        self.__minimos__.append(min(self.__fit_desc__))
        self.__maximos__.append(max(self.__fit_desc__))
        self.__medias__.append(np.mean(self.__fit_desc__))
        self.__stds__.append(np.std(self.__fit_desc__))

    def comprueba_descendencia(self):
        media = np.mean(self.__permutaciones__)
        std = np.std(self.__permutaciones__)
        for individuo in self.__descendencia__:
            if media != np.mean(individuo) or std != np.std(individuo):
                raise AttributeError('Error de integridad')
            else:
                return 0

    def escribe_resultados(self, interaccion, repeticion, instancia, data_path=None):
        self.__metricas__()
        if type(data_path) is type(None):
            self.__resultado_path__ = 'data/CE_resultado_' + str(self.__n_pob__) \
                                      + '_' + str(self.__n_puntos__) \
                                      + '_' + str(self.__swap_prob__) + '.csv'
            if not os.path.exists('data/'):
                os.makedirs('data/')
        else:
            self.__resultado_path__ = data_path
        with open(self.__resultado_path__, 'a') as f:
            f.write(';'.join([str(instancia), str(repeticion), str(interaccion),
                              str(self.__tiempos__[instancia]).replace('.', ','),
                              str(self.__maximos__[instancia]).replace('.', ','),
                              str(self.__minimos__[instancia]).replace('.', ','),
                              str(self.__medias__[instancia]).replace('.', ','),
                              str(self.__stds__[instancia]).replace('.', ',')]) + '\n')

    def solucion(self, criterio='MAX'):
        if criterio == 'MAX':
            return self.__descendencia__[self.__fit_desc__.index(max(self.__fit_desc__))]


def ejecuta(sp,n_pob,n_punt, max_iter):
    os.system("C:\\Users\\luis_\\Anaconda3\\python.exe problema_viajante_mt_sp.py "+
              " ".join([str(sp), str(n_pob), str(n_punt), str(max_iter)]))
