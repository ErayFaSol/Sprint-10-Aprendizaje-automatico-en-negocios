# Proyecto 10: Aprendizaje Automático en Negocios - OilyGiant

## Descripción
Este proyecto implementa un modelo de aprendizaje automático para la compañía de extracción de petróleo OilyGiant. El objetivo es encontrar las mejores ubicaciones para abrir 200 nuevos pozos de petróleo. Se utilizan datos geológicos de tres regiones para predecir el volumen de reservas y seleccionar las ubicaciones más rentables.

## Estructura del Proyecto
- **preprocessing/**: Carga y preprocesamiento de datos.
- **models/**: Entrenamiento del modelo de regresión lineal.
- **utils/**: Cálculo de ganancias y riesgos mediante bootstrapping.
- **visualizations/**: Gráficos de los resultados.
- **datasets/**: Datos utilizados (excluidos del repositorio).
  
## Ejecucion

1. Clonar el repositorio
2. Instalar las dependencias
   ```
   pip install -r requirements.txt
   ```
3. Ejecuta el script principal
   ``` 
   python src/main.py
   ```
4. La ejecucion creara un archivo llamado *reporte_final.html*