import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Desactivar el modo interactivo de Matplotlib
plt.ioff()

# Leer datos de los archivos CSV
training_faces = pd.read_csv("/home/gugui/Documentos/Austral/Austral/Data mining avanzado/Trabajo_practico/DMA-Grupo3-main/csvs/faces_train.csv", delimiter=",")
training_names = pd.read_csv("/home/gugui/Documentos/Austral/Austral/Data mining avanzado/Trabajo_practico/DMA-Grupo3-main/csvs/names_train.csv", header=None).values
training_files = pd.read_csv("/home/gugui/Documentos/Austral/Austral/Data mining avanzado/Trabajo_practico/DMA-Grupo3-main/csvs/files_train.csv", header=None).values


# Validar las dimensiones del conjunto de datos
print("Dimensiones del conjunto de datos de entrenamiento:", training_faces.shape)

if training_faces.shape[0] <= 1 or training_faces.shape[1] <= 1:
    raise ValueError("El conjunto de datos no tiene suficientes filas o columnas para aplicar PCA.")

# Convertir datos de `training_faces` a tipo float64 para precisión numérica
training_faces = training_faces.astype('float64')

# Normalizar valores a rango 0-1 (suponiendo datos originales en rango 0-255)
training_faces /= 255.0

# Ajustar `n_components` al mínimo entre el número de muestras y características
n_components = min(60, training_faces.shape[0], training_faces.shape[1])
print(f"Usando {n_components} componentes principales para PCA.")

# Configuración y ejecución de PCA
pca = PCA(n_components=n_components, svd_solver='full')
pca.fit(training_faces)

# Transformar los datos originales al espacio PCA
training_Z = pca.transform(training_faces)

# Imprimir información sobre el PCA
print(f"Número de componentes utilizados en el PCA: {pca.n_components_}")

# (Opcional) Graficar la variación explicada por los componentes principales
plt.figure(figsize=(8, 5))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.title('Varianza explicada acumulada por los componentes principales')
plt.xlabel('Número de componentes principales')
plt.ylabel('Proporción de varianza explicada')
plt.grid()
plt.show()

