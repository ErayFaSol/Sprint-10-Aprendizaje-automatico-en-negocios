def save_report(analysis_results, rmse, volumen_promedio, ganancia, volumen, beneficio_promedio, intervalo_confianza, riesgo_perdidas, filepath="final_report.html"):
    with open(filepath, 'w') as file:
        # Título del reporte
        file.write("<h1>Reporte Final de Análisis de Pozos Petrolíferos</h1>")
        
        # Introducción
        file.write("<h2>1. Introducción</h2>")
        file.write("<p>Este análisis tiene como objetivo evaluar tres regiones para identificar el mejor lugar donde abrir 200 pozos nuevos de petróleo utilizando técnicas de aprendizaje automático.</p>")
        
        # Resultados del modelo
        file.write("<h2>2. Resultados del Modelo</h2>")
        file.write("<ul>")
        for i, region in enumerate(analysis_results.keys(), start=1):
            file.write(f"<li><b>Región {i}</b>: RMSE = {rmse[i-1]:.2f}, Volumen promedio de reservas = {volumen_promedio[i-1]:.2f} miles de barriles.</li>")
        file.write("</ul>")
        
        # Gráficas
        file.write("<h3>Gráficas de Resultados</h3>")
        file.write('<img src="images/rmse_by_region.png" alt="RMSE por Región" width="500">')
        file.write('<img src="images/average_volume_by_region.png" alt="Volumen Promedio por Región" width="500">')
        
        # Ganancia y Volumen
        file.write("<h2>3. Análisis de Ganancias</h2>")
        file.write(f"<p>Ganancia proyectada en Región 3: ${ganancia:,.2f}. Volumen total: {volumen:.2f} miles de barriles.</p>")
        
        # Evaluación de riesgo
        file.write("<h2>4. Evaluación de Riesgos</h2>")
        file.write(f"<p>Beneficio promedio: ${beneficio_promedio:,.2f}. Intervalo de confianza (95%): ${intervalo_confianza[0]:,.2f} a ${intervalo_confianza[1]:,.2f}. Riesgo de pérdidas: {riesgo_perdidas:.2f}%.</p>")
        
        # Conclusiones
        file.write("<h2>5. Conclusiones</h2>")
        file.write("<p>La Región 3 ofrece el mayor volumen de reservas y el beneficio promedio más alto, con un bajo riesgo de pérdidas. Se recomienda avanzar con esta región.</p>")
