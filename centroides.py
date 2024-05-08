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
E_max = []

# Calcular centroides por persona
# Centroide = promedio a traves de las columnas para cada foto  
for i in names_faces_uniq: #barro
    face_position = np.where(names_faces == i)[0]
    Z_aux = Z[face_position]
    Z_prom = np.mean(Z_aux, axis=0)
    Z_cent.append(Z_prom)
    distancia_max = 0
    for j in range(len(Z_aux)):
        distancia = np.linalg.norm(Z_aux[j] - Z_prom)
        if distancia > distancia_max:
            distancia_max = distancia
    E_max.append(distancia_max)
    
#print(E_max)
#print(Z_cent)

