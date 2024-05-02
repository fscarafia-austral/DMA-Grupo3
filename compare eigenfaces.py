import numpy as np
import pandas as pd
from sklearn.decomposition import PCA  

#leer el numpy arrange
flat_faces_np = np.loadtxt("C:\\Users\\Franc\\Downloads\\TP-Nuestras-Caras\\faces_numpy.csv", delimiter=",")
print(len(flat_faces_np))

names_faces_np = pd.read_csv("C:\\Users\\Franc\\Downloads\\TP-Nuestras-Caras\\names_numpy.csv", delimiter=",",header = None)
names_faces = names_faces_np.values

files_faces_np = pd.read_csv("C:\\Users\\Franc\\Downloads\\TP-Nuestras-Caras\\files_numpy.csv", delimiter=",",header = None)
files_faces = files_faces_np.values

pca = PCA(n_components= 60)
pca.fit(flat_faces_np)
Z = pca.transform(flat_faces_np)

# Calcular todas las distancias euclidianas entre vectores de diferentes clases
distancia_minima = float('inf')
persona1 = None
persona2 = None
foto1 = None
foto2 = None

for i in range(len(Z)):
    for j in range(i+1, len(Z)):
        if names_faces[i] != names_faces[j]:
            distancia = np.linalg.norm(Z[i] - Z[j])
            if distancia < distancia_minima:
                distancia_minima = distancia
                persona1 = names_faces[i]
                persona2 = names_faces[j]
                foto1 = i
                foto2 = i+j
                
print(distancia_minima, persona1, persona2, files_faces[foto1], files_faces[foto2])