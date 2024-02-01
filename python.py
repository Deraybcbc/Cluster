import json
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import sys
import os

# Leer el JSON desde la entrada estándar
json_data = sys.stdin.read()

# Convertir el JSON a un diccionario de Python
data = json.loads(json_data)

# Crear un DataFrame desde el diccionario
df = pd.DataFrame(data["clientes"])

# Escalar los datos para que tengan media cero y desviación estándar unitaria
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Aplicar el algoritmo k-means para agrupar en 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_data)

# Visualizar los resultados (opcional)
plt.scatter(df['Edad'], df['Gasto Mensual (USD)'], c=df['Cluster'], cmap='viridis')
plt.title('Clusters de Clientes basados en Edad y Gasto Mensual')
plt.xlabel('Edad')
plt.ylabel('Gasto Mensual (USD)')

# Crear la carpeta 'graficos' si no existe
if not os.path.exists('graficos'):
    os.makedirs('graficos')

# Guardar la imagen en la carpeta 'graficos'
plt.savefig('graficos/clusters_clientes.png')

# Imprimir el resultado o cualquier otro dato que desees devolver al servidor Node.js
print("Proceso Python completado")

plt.scatter(df['Edad'], df['Gasto Mensual (USD)'], c=df['Cluster'], cmap='viridis')
plt.title('Clusters de Clientes basados en Edad y Gasto Mensual')
plt.xlabel('Edad')
plt.ylabel('Gasto Mensual (USD)')

# Crear la carpeta 'graficos' si no existe
if not os.path.exists('graficos'):
    os.makedirs('graficos')

# Guardar la imagen en la carpeta 'graficos'
plt.savefig('graficos/clusters_clientes.png')

# Mostrar la imagen
plt.show()