from preprocessing.preprocessing import load_data
from models.model_trainer import process_data
from utils.bootstrap import bootstrap
from utils.profit_calculator import calcular_ganancia
from utils.generate_report import save_report
from visualizations.plotter import plot_rmse_by_region, plot_regions_average_volume

# Cargar datos 
data_0 = load_data('datasets/geo_data_0.csv')
data_1 = load_data('datasets/geo_data_1.csv')
data_2 = load_data('datasets/geo_data_2.csv')

# Procesar modelos
results = {}
for i, data in enumerate([data_0, data_1, data_2], start=1):
    rmse, mean_product, predictions = process_data(data)
    results[f'Region {i}'] = {'RMSE': rmse, 'Mean Product': mean_product}

# Preparar graficos
regiones = list(results.keys())
rmse = [results[region]['RMSE'] for region in regiones]
volumen_promedio = [results[region]['Mean Product'] for region in regiones]

# Graficar RMSE
plot_rmse_by_region(rmse, regiones)

# Calcular volumen promedio
plot_regions_average_volume(regiones, volumen_promedio)

# Calcular ganancia y riesgo
predictions_region_3 = process_data(data_2)[2]  # Extraer predicciones
ganancia, volumen = calcular_ganancia(predictions_region_3, 4500)
print(f"Ganancia Proyectada Región 3: ${ganancia:,.2f} \nVolumen: {volumen}")


beneficio_promedio, intervalo_confianza, riesgo_perdidas = bootstrap(predictions_region_3, 4500)
print(f"Beneficio promedio Región 3: ${beneficio_promedio:,.2f}")
print(f"Intervalo de confianza (95%): ${intervalo_confianza[0]:,.2f} a ${intervalo_confianza[1]:,.2f}")
print(f"Riesgo de pérdidas Región 3: {riesgo_perdidas:.2f}%")

# Generar reporte

save_report(
    analysis_results=results,
    rmse=rmse,
    volumen_promedio=volumen_promedio,
    ganancia=ganancia,
    volumen=volumen,
    beneficio_promedio=beneficio_promedio,
    intervalo_confianza=intervalo_confianza,
    riesgo_perdidas=riesgo_perdidas
)