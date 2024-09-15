import pandas as pd

def calcular_ganancia(predicciones, ingreso_por_unidad, top_n=200):
    if not isinstance(predicciones, pd.Series):
        predicciones = pd.Series(predicciones)
    predicciones_seleccionadas = predicciones.nlargest(top_n)
    volumen_total_reservas = predicciones_seleccionadas.sum()
    ganancia_total = volumen_total_reservas * ingreso_por_unidad
    
    return ganancia_total, volumen_total_reservas