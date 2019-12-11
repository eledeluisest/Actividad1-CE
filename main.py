
from multiprocessing import Process
from utils import ejecuta

jobs = []
"""
N_POB = sys.argv[2]
N_PUNTOS = sys.argv[3]
"""
if __name__ == '__main__':
    for sp in [0,0.25,0.5,0.75,1]:
        # ejecución del algoritmo para los distintos parámetros
        p = Process(target=ejecuta, args=(sp, 10, 100, 1000))
        j = Process(target=ejecuta, args=(sp, 100, 10000, 100))
        p.start()
        j.start()
        jobs.append(p)
        jobs.append(j)

    for job in jobs:
        job.join()

