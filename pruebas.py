from utils import *

# Este codigo genera ejemplos del problema del viajante de forma aleatoria uniforme
viajante = generador_ejemplos(problema='VIAJANTE', data_path='data/ejemplo1.csv')
puntos = 5
limites = [10,10]
viajante.viajante_ini(n=5,lista_lim=[10,10])
viajante.genera_instancias(1)

from utils import *
# Empezamos con el modelo genetico
AE_viajante = algoritmo_genetico()
AE_viajante.carga_instancia('data/ejemplo1.csv')
AE_viajante.elige_instancia()
AE_viajante.codifica_fenotipo_viajante()
# AE_viajante.__fenotipo__
AE_viajante.genera_poblacion(n_pob=10)
# AE_viajante.__poblacion__
for i in range(1000):
    AE_viajante.seleccion_parental(k=5)
    # AE_viajante.genera_descenencia()
    AE_viajante.modelo_generacional()
# AE_viajante.comprueba_descendencia()





