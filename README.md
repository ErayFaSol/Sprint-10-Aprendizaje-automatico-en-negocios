# Aprendizaje Automático en Negocios

Este proyecto tiene como objetivo predecir el volumen de reservas de crudo y seleccionar las regiones más rentables para nuevos pozos petrolíferos. Se emplearon modelos de regresión para analizar datos históricos y predecir el potencial de extracción en diferentes áreas.

## Tecnologías utilizadas
- Python
- pandas
- NumPy
- scikit-learn
- Matplotlib

## Objetivo
Desarrollar un modelo de machine learning para predecir el volumen de reservas de crudo y ayudar en la selección de las regiones más rentables para la explotación petrolera.

## Contexto
La industria petrolera enfrenta constantes desafíos en la selección de áreas para la extracción de crudo. Este proyecto tiene como objetivo identificar las áreas más prometedoras utilizando modelos predictivos basados en datos históricos. Al predecir el volumen de reservas, las compañías pueden optimizar su inversión y reducir riesgos.

## Descripción del Proyecto
El análisis se basa en un conjunto de datos que contiene información sobre reservas de crudo y características geológicas de diferentes regiones. El objetivo es crear un modelo de regresión para predecir la cantidad de crudo disponible y así guiar las decisiones de inversión.

Proceso del proyecto:
1. **Carga y exploración de datos**: Se inspeccionan las características más relevantes y las correlaciones con el volumen de reservas de crudo.
2. **Preprocesamiento de datos**: Se realizan tareas de limpieza y transformación de los datos.
3. **Entrenamiento del modelo**: Se entrenaron varios modelos de regresión, incluyendo regresión lineal y random forest.
4. **Evaluación del modelo**: Se utilizaron métricas como el RMSE y el R² para medir el rendimiento del modelo.

## Proceso

### Exploración de Datos
Se realizó un análisis exploratorio de los datos utilizando pandas y matplotlib para identificar las principales características que influyen en el volumen de reservas. Se exploraron correlaciones entre características geológicas, ubicación y el volumen de reservas.

### Preprocesamiento
Se manejaron valores faltantes y se transformaron las características geológicas en variables más manejables para el modelo. También se normalizaron los datos para mejorar el rendimiento del modelo de regresión.

### Entrenamiento del Modelo
Los modelos entrenados incluyeron:
- **Regresión Lineal**: Un modelo simple pero efectivo para establecer una línea base de rendimiento.
- **Random Forest Regressor**: Utilizado para capturar relaciones no lineales y mejorar la precisión de las predicciones.

El modelo de **Random Forest** obtuvo el mejor rendimiento, con un **R² de 0.92** y un **RMSE** bajo.

### Evaluación del Modelo
El modelo se evaluó utilizando las siguientes métricas:
- **R² (Coeficiente de Determinación)**: 0.92
- **RMSE (Raíz del Error Cuadrático Medio)**: El error promedio en las predicciones fue bajo, indicando que el modelo predice de manera precisa las reservas de crudo.

## Resultados
El modelo de **Random Forest Regressor** logró una alta precisión en la predicción del volumen de reservas de crudo, proporcionando a la compañía una herramienta confiable para seleccionar áreas de perforación.

## Conclusiones
El proyecto demostró que es posible predecir el volumen de reservas de crudo utilizando modelos de aprendizaje automático, lo que permite a las empresas petroleras realizar mejores inversiones en nuevas áreas de perforación.

### Futuras mejoras
- Recolectar más datos para mejorar la robustez del modelo.
- Implementar modelos más avanzados como XGBoost o LightGBM.
- Explorar el uso de datos geoespaciales para mejorar la precisión de las predicciones.

### Enlace al proyecto
[Aprendizaje Automático en Negocios](https://github.com/ErayFaSol/Sprint-10-Aprendizaje-automatico-en-negocios)
