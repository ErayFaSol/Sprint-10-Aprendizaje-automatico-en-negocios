# %% [markdown]
# # Proyecto 10: Aprendizaje automático en negocios

# %% [markdown]
# ## Descripcion del proyecto

# %% [markdown]
# Trabajas en la compañía de extracción de petróleo OilyGiant. Tu tarea es encontrar los mejores lugares donde abrir 200 pozos nuevos de petróleo.
# 
# Para completar esta tarea, tendrás que realizar los siguientes pasos:
# 
# - Leer los archivos con los parámetros recogidos de pozos petrolíferos en la región seleccionada: calidad de crudo y volumen de reservas.
# - Crear un modelo para predecir el volumen de reservas en pozos nuevos.
# - Elegir los pozos petrolíferos que tienen los valores estimados más altos.
# - Elegir la región con el beneficio total más alto para los pozos petrolíferos seleccionados.
# 
# Tienes datos sobre muestras de crudo de tres regiones. Ya se conocen los parámetros de cada pozo petrolero de la región. Crea un modelo que ayude a elegir la región con el mayor margen de beneficio. Analiza los beneficios y riesgos potenciales utilizando la técnica bootstrapping.

# %% [markdown]
# ## Condiciones
# 

# %% [markdown]
# - Solo se debe usar la regresión lineal para el entrenamiento del modelo.
# - Al explorar la región, se lleva a cabo un estudio de 500 puntos con la selección de los mejores 200 puntos para el cálculo del beneficio.
# - El presupuesto para el desarrollo de 200 pozos petroleros es de 100 millones de dólares.
# - Un barril de materias primas genera 4.5 USD de ingresos. El ingreso de una unidad de producto es de 4500 dólares (el volumen de reservas está expresado en miles de barriles).
# - Después de la evaluación de riesgo, mantén solo las regiones con riesgo de pérdidas inferior al 2.5%. De las que se ajustan a los criterios, se debe seleccionar la región con el beneficio promedio más alto.
# 
# Los datos son sintéticos: los detalles del contrato y las características del pozo no se publican.

# %% [markdown]
# ## Descripcion de los datos
# 

# %% [markdown]
# Los datos de exploración geológica de las tres regiones se almacenan en archivos:
# 
# - geo_data_0.csv. 
# - geo_data_1.csv. 
# - geo_data_2.csv. 
# 
# - id — identificador único de pozo de petróleo
# - f0, f1, f2 — tres características de los puntos (su significado específico no es importante, pero las características en sí son significativas)
# - product — volumen de reservas en el pozo de petróleo (miles de barriles).

# %% [markdown]
# ## Instrucciones del proyecto
# 

# %% [markdown]
# 1. Descarga y prepara los datos. Explica el procedimiento.
# 2. Entrena y prueba el modelo para cada región en geo_data_0.csv:
# 
#     2.1 Divide los datos en un conjunto de entrenamiento y un conjunto de validación en una proporción de 75:25
# 
#     2.2 Entrena el modelo y haz predicciones para el conjunto de validación.
# 
#     2.3 Guarda las predicciones y las respuestas correctas para el conjunto de validación.
# 
#     2.4 Muestra el volumen medio de reservas predicho y RMSE del modelo.
# 
#     2.5 Analiza los resultados.
# 
#     2.6 Coloca todos los pasos previos en funciones, realiza y ejecuta los pasos 2.1-2.5 para los archivos 'geo_data_1.csv' y 'geo_data_2.csv'.
# 
# 3. Prepárate para el cálculo de ganancias:
# 
#     3.1 Almacena todos los valores necesarios para los cálculos en variables separadas.
# 
#     3.2 Dada la inversión de 100 millones por 200 pozos petrolíferos, de media un pozo petrolífero debe producir al menos un valor de 500,000 dólares en unidades para evitar pérdidas (esto es equivalente a 111.1 unidades). Compara esta cantidad con la cantidad media de reservas en cada región.
# 
#     3.3 Presenta conclusiones sobre cómo preparar el paso para calcular el beneficio.
# 
# 4. Escribe una función para calcular la ganancia de un conjunto de pozos de petróleo seleccionados y modela las predicciones:
# 
#     4.1 Elige los 200 pozos con los valores de predicción más altos de cada una de las 3 regiones (es decir, archivos 'csv').
# 
#     4.2 Resume el volumen objetivo de reservas según dichas predicciones. Almacena las predicciones para los 200 pozos para cada una de las 3 regiones.
# 
#     4.3 Calcula la ganancia potencial de los 200 pozos principales por región. Presenta tus conclusiones: propón una región para el desarrollo de pozos petrolíferos y justifica tu elección.
# 
# 5. Calcula riesgos y ganancias para cada región:
# 
#     5.1 Utilizando las predicciones que almacenaste en el paso 4.2, emplea la técnica del bootstrapping con 1000 muestras para hallar la distribución de los beneficios.
# 
#     5.2 Encuentra el beneficio promedio, el intervalo de confianza del 95% y el riesgo de pérdidas. La pérdida es una ganancia negativa, calcúlala como una probabilidad y luego exprésala como un porcentaje.
# 
#     5.3 Presenta tus conclusiones: propón una región para el desarrollo de pozos petrolíferos y justifica tu elección. ¿Coincide tu elección con la elección anterior en el punto 4.3?

# %% [markdown]
# ## Evaluación del proyecto

# %% [markdown]
# Hemos definido los criterios de evaluación para el proyecto. Lee esto con atención antes de pasar al ejercicio.
# 
# Esto es lo que los revisores buscarán cuando evalúen tu proyecto:
# 
# - ¿Cómo preparaste los datos para el entrenamiento?
# - ¿Seguiste todos los pasos de las instrucciones?
# - ¿Consideraste todas las condiciones del negocio?
# - ¿Cuáles son tus hallazgos sobre el estudio de tareas?
# - ¿Aplicaste correctamente la técnica bootstrapping?
# - ¿Sugeriste la mejor región para el desarrollo de pozos? ¿Justificaste tu elección?
# - ¿Evitaste la duplicación de código?
# - ¿Mantuviste la estructura del proyecto y el código limpio?
# 
# Ya tienes tus hojas informativas y los resúmenes de los capítulos, por lo que todo está listo para continuar con el proyecto.
# 
# ¡Buena suerte!

# %% [markdown]
# ## Paso 1

# %% [markdown]
# ### Descarga y prepara los datos. Explica el procedimiento.

# %%
# Importar bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt

# %%
# Importar data frame 
data_0 = pd.read_csv('datasets/geo_data_0.csv')
data_1 = pd.read_csv('datasets/geo_data_1.csv')
data_2 = pd.read_csv('datasets/geo_data_2.csv')

# Data 3 ------------------------------
print(('===== Dataframe 0 ====='))
print(data_0.head())
print(data_0.info())
print(data_0.describe())
print(('--------------------------'))

# Data 1  -----------------------------
print(('===== Dataframe 1 ====='))
print(data_1.head())
print(data_1.info())
print(data_1.describe())
print(('--------------------------'))

# Data 2 ------------------------------
print(('===== Dataframe 2 ====='))
print(data_2.head())
print(data_2.info())
print(data_2.describe())
print(('--------------------------'))


# comentario
comentario = """ 
==== Comentario sobre la exploracion inicial del archivo ====
Podemos notar que los datos en los tres dataframe estan complentos, no hay valores faltantes en las columnas  y todos presentan las mismas columnas que se mencionan en la descripcion del proyecto 
No se observan valores atípicos aparentes o incoherencias en las características `f0`, `f1`, `f2`, y `product`. 

"""
print((comentario))



# %% [markdown]
# ## Paso 2

# %% [markdown]
# ### Entrena y prueba el modelo para cada región 

# %%
# Funcion para crear modelos
def process_data(data):
     # División de los datos
    X = data[['f0', 'f1', 'f2']]
    y = data['product']
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=123)
    
    # Entrenamiento del modelo
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predicciones y evaluación
    predictions = model.predict(X_valid)
    rmse = sqrt(mean_squared_error(y_valid, predictions))
    mean_product = predictions.mean()
    
    return rmse, mean_product, predictions

# Aplicar funcion
results = {}
for i, data in enumerate([data_0, data_1, data_2], start=1):
    rmse, mean_product, predictions = process_data(data)
    results[f'Region {i}'] = {'RMSE': rmse, 'Mean Product': mean_product}

results

# %%
# Datos
regiones = list(results.keys())
rmse = [results[region]['RMSE'] for region in regiones]
volumen_promedio = [results[region]['Mean Product'] for region in regiones]

# Gráfico de barras para RMSE
plt.figure(figsize=(10, 5))
plt.bar(regiones, rmse, color='skyblue')
plt.xlabel('Región')
plt.ylabel('RMSE')
plt.title('RMSE por Región')
plt.show()

# Gráfico de barras para volumen promedio de reservas
plt.figure(figsize=(10, 5))
plt.bar(regiones, volumen_promedio, color='lightgreen')
plt.xlabel('Región')
plt.ylabel('Volumen Promedio de Reservas (miles de barriles)')
plt.title('Volumen Promedio de Reservas por Región')
plt.show()



# %% [markdown]
# El root mean square error (RMSE) mide la diferencia entre los valores predichos del modelo y los valores reales
# 
# Region 1: RMSE de 37.65 nos indica que hay un porcentaje moderado de errores en comparacion con los valores reales
# 
# Region 2: RMSE de 0.89 es un modelo altamente preciso con valores cercanos a valores reales. Este es un resultado demasiado bueno sin embargo se tiene que considerar la posibilidad de un sobreajuste.
# 
# Region 3: RMSE de 40.13 similar a la region 1 podemos decir lo mismo sobre que tiene un grado de error moderado.
# 
# Mean product: 
# 
# Region 1: El volumen medio es de 92.55 miles de barriles, esto indica una buena produccion.
# 
# Region 2: Volumen medio de 69.28 miles de barriles siendo esta la region con menor produccion, refleja un menor potencial
# 
# Region 3: Volumen medio de 95.10 miles de barriles, esta region tiene la mayor produccion de las 3
# 
# La region 2 muestra la mayor precision de estos 3 modelos sin embargo tambien es la region que presenta un volumen menor de produccion, existe la posibilidad de que se trate de un modelo sobreajustado.
# 
# La region 3 tiene un volumen medio de reserva entre las regiones evaluadas, esto indica una mayor produccion por lo cual puede conducir a mayores ganancias a alargo plazo. Esta region parece ser la mas optima para calcular la ganancia debido a su potencial de produccion por eso se decide contiunar los calculos de ganacia con base a esta region
# 

# %% [markdown]
# ## Paso 3
# 

# %% [markdown]
# ### Prepárate para el cálculo de ganancias:

# %%
# Definir las condiciones dadas
presupuesto_total = 100_000_000  # 100 millones de dólares
numero_de_pozos = 200
ingreso_por_barril = 4.5
ingreso_por_unidad = 4500  # Ingresos por mil barriles

# Calcular el costo por pozo y el volumen necesario por pozo para evitar pérdidas
costo_por_pozo = presupuesto_total / numero_de_pozos
unidades_necesarias_por_pozo = costo_por_pozo / ingreso_por_unidad

# Volumen necesario por pozo en miles de barriles para evitar pérdidas
volumen_necesario_por_pozo = unidades_necesarias_por_pozo  # Esto ya está en miles de barriles

volumen_necesario_por_pozo

# %% [markdown]
# ## Paso 4

# %% [markdown]
# ### Escribe una función para calcular la ganancia de un conjunto de pozos de petróleo seleccionados y modela las predicciones:

# %%
def calcular_ganancia(predicciones, ingreso_por_unidad, top_n=200):
    if not isinstance(predicciones, pd.Series):
        predicciones = pd.Series(predicciones)
    predicciones_seleccionadas = predicciones.nlargest(top_n)
    volumen_total_reservas = predicciones_seleccionadas.sum()
    ganancia_total = volumen_total_reservas * ingreso_por_unidad
    
    return ganancia_total, volumen_total_reservas

predictions_region_3 = process_data(data_2) 
predictions_region_3 = predictions_region_3[2]

ganancia, volumen = calcular_ganancia(predictions_region_3, ingreso_por_unidad)

print(f"Region 3: Volumen total: {volumen:.2f} miles de barriles, ganancia: ${ganancia:,.2f}")

# %% [markdown]
# ## Paso 5
# 
# 

# %% [markdown]
# ### Calcula riesgos y ganancias para cada región:

# %%
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

# Debes tener tus predicciones como un array de NumPy para cada región
# Ejemplo de cómo llamar a la función para una región específica (usando predicciones simuladas aquí)
beneficio_promedio, intervalo_confianza, riesgo_perdidas = bootstrap(predictions_region_3, ingreso_por_unidad)

print(f"Beneficio promedio: ${beneficio_promedio:,.2f}")
print(f"Intervalo de confianza del 95%: ${intervalo_confianza[0]:,.2f} a ${intervalo_confianza[1]:,.2f}")
print(f"Riesgo de pérdidas: {riesgo_perdidas:.2f}%")



# %% [markdown]
# ## Conclusion:
# 

# %% [markdown]
# Tras analizar los datos se concluye lo siguiente: 
# 
# - La region 3 muestra el mayor potencial de produccion con el volumen medio de reservas mas alto y preciso aceptable en las predicciones del modelo
# 
# - En el caso de la region 2 que mostro mas precision su volumen medio fue el mas bajo de entre los
# 
# - La region 1 fue muy similar en precision que la region 3 sin embargo esta presente un volumen medio inferior a esta.
# 
# Se considera la region 3 como la mas favorable para el desarrollo de pozos petroliferos basandonos en sus resultados en las precdicciones del modelo 


