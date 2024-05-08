import numpy as np
import pandas as pd
from sklearn.decomposition import PCA  

#leer el numpy arrange
flat_faces_np = np.loadtxt(".\\faces_numpy.csv", delimiter=",")

names_faces_np = pd.read_csv(".\\names_numpy.csv", delimiter=",",header = None)
names_faces = names_faces_np.values

files_faces_np = pd.read_csv(".\\files_numpy.csv", delimiter=",",header = None)
files_faces = files_faces_np.values

pca = PCA(n_components= 63)
pca.fit(flat_faces_np)
U_aux = pca.components_[3:]
pca.components_ = U_aux
Z = pca.transform(flat_faces_np)

names_faces_uniq = np.unique(names_faces)
Z_cent = []

# Calcular centroides por persona
# Centroide = promedio a traves de las columnas para cada foto  
for i in names_faces_uniq: #barro
    face_position = np.where(names_faces == i)[0]
    Z_aux = Z[face_position]
    Z_prom = np.mean(Z_aux, axis=0)
    Z_cent.append(Z_prom)

# Calcular todas las distancias euclidianas entre vectores de diferentes clases
distancia_minima = float('inf')
persona1 = None
persona2 = None

for i in range(len(Z_cent)):
    for j in range(i+1, len(Z_cent)):
        # if names_faces[i] != names_faces[j]:
        distancia = np.linalg.norm(Z_cent[i] - Z_cent[j])
        if distancia < distancia_minima:
            distancia_minima = distancia
            persona1 = names_faces_uniq[i]
            persona2 = names_faces_uniq[j]
                
print(distancia_minima, persona1, persona2)

person_pos_1 = np.where(names_faces == persona1)[0]
person_pos_2 = np.where(names_faces == persona2)[0]

distancia_minima_2 = float('inf')
foto1 = None
foto2 = None

for i in person_pos_1:
    for j in person_pos_2:
        distancia_2 = np.linalg.norm(Z[i] - Z[j])
        if distancia_2 < distancia_minima_2:
            distancia_minima_2 = distancia_2
            foto1 = files_faces[i]
            foto2 = files_faces[j]
            
print(distancia_minima_2, foto1, foto2)