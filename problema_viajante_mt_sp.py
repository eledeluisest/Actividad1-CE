import sys
from utils import algoritmo_genetico

N_ITERACIONES = int(sys.argv[4])
# DIF_SALIR_100 = 0.0005
DIF_SALIR = 0.00001  # calculamos la diferencia entre una generacion y la siguiente y la dividimos entre la media de la ultima
INSTANCIAS = 1
MIN_ITERACIONES = 5
N_POB = int(sys.argv[2])
N_PUNTOS = int(sys.argv[3])
# K_TORNEO_100 = 5
K_TORNEO = 5
N_PROMEDIO = 30

ruta_10000 = 'data/viajante_10000_10.csv'
ruta_100 = 'data/viajante_100_50.csv'
sp = float(sys.argv[1])
for instancia in range(INSTANCIAS):
    for repeticion in range(N_PROMEDIO):
        iteracion = 0
        dif = DIF_SALIR + 1
        # Inicializamos el algoritmo genetico
        AE_viajante = algoritmo_genetico()
        # Cargamos las instancias
        AE_viajante.carga_instancia('data/viajante_'+sys.argv[3]+'.csv')
        # Elegimos la instancia para poder iterar por ellas
        AE_viajante.elige_instancia(instancia)
        print(sp, N_POB , instancia, repeticion)
        AE_viajante.codifica_fenotipo_viajante()
        # AE_viajante.__fenotipo__
        AE_viajante.genera_poblacion(n_pob=N_POB)

        while (iteracion < N_ITERACIONES and dif > DIF_SALIR) or iteracion < MIN_ITERACIONES:
            # AE_viajante.__poblacion__
            AE_viajante.seleccion_parental(k=K_TORNEO)
            # AE_viajante.genera_descenencia()
            AE_viajante.modelo_generacional(swap_prob=sp)
            # Guardamos los resultados
            AE_viajante.escribe_resultados(instancia, repeticion, iteracion)
            # Actualizamos condicion de salida
            if iteracion > 1:
                dif = (abs(AE_viajante.__medias__[iteracion] -
                           AE_viajante.__medias__[iteracion - 1]) + abs(AE_viajante.__medias__[iteracion] -
                                                                        AE_viajante.__medias__[iteracion - 2])) / (
                                  abs(2 * AE_viajante.__medias__[iteracion]))
                # print(sp, instancia, repeticion,dif)
            iteracion = iteracion + 1

        with open('data/soluciones.csv', 'a') as f:
            f.write(';'.join([str(sp), str(instancia), str(repeticion)])+';'+ '-'.join(
                [str(x) for x in AE_viajante.solucion()])+'\n')
