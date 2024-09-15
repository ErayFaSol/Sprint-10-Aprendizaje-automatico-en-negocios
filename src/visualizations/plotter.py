import matplotlib.pyplot as plt
import os

if not os.path.exists('images'):
    os.makedirs('images')


def plot_rmse_by_region(rmse_values, regiones, filename='rmse_by_region.png'):
    plt.figure(figsize=(10, 5))
    plt.bar(regiones, rmse_values, color='skyblue')
    plt.xlabel('Región')
    plt.ylabel('RMSE')
    plt.title('RMSE por Región')
    plt.savefig('images/' + filename)
    plt.close()

def plot_regions_average_volume(regiones, volumen_promedio, filename='average_volume_by_region.png'):
    plt.figure(figsize=(10, 5))
    plt.bar(regiones, volumen_promedio, color='lightgreen')
    plt.xlabel('Región')
    plt.ylabel('Volumen Promedio de Reservas (miles de barriles)')
    plt.title('Volumen Promedio de Reservas por Región')
    plt.savefig('images/' + filename)
    plt.close()
