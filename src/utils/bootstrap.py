import numpy as np
from utils.profit_calculator import calcular_ganancia

def bootstrap(predicciones, ingreso_por_unidad, n_iter=1000, n_top=200):
    ganancias = []
    for _ in range(n_iter):
        # Seleccionar aleatoriamente n_top predicciones
        muestra = np.random.choice(predicciones, size=n_top, replace=True)
        # Calcular la ganancia para la muestra
        ganancia, _ = calcular_ganancia(muestra, ingreso_por_unidad, n_top)
        ganancias.append(ganancia)
    
    ganancias = np.array(ganancias)
    beneficio_promedio = ganancias.mean()
    intervalo_confianza = np.percentile(ganancias, [2.5, 97.5])
    riesgo_perdidas = (ganancias < 0).mean() * 100  # Porcentaje de veces que la ganancia fue negativa
    
    return beneficio_promedio, intervalo_confianza, riesgo_perdidas