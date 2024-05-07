import numpy as np
import pandas as pd
from sklearn.decomposition import PCA  
import random

#leer el numpy arrange
flat_faces_np = np.loadtxt("/home/gugui/Documentos/Austral/Data mining avanzado/Trabajo_practico/DMA-Grupo3-main/faces_numpy.csv", delimiter=",")

names_faces_np = pd.read_csv("/home/gugui/Documentos/Austral/Data mining avanzado/Trabajo_practico/DMA-Grupo3-main/names_numpy.csv", delimiter=",",header = None)
names_faces = names_faces_np.values

files_faces_np = pd.read_csv("/home/gugui/Documentos/Austral/Data mining avanzado/Trabajo_practico/DMA-Grupo3-main/files_numpy.csv", delimiter=",",header = None)
files_faces = files_faces_np.values

# Separacion en training y test
training_faces = []
test_faces = []

training_names = []
test_names = []

training_files = []
test_files = []

# Nombres únicos de personas
names_faces_uniq = np.unique(names_faces)

#Semilla aleatoria
np.random.seed(1957)

# Loop para seleccionar 3 fotos de cada uno p/el test y el resto quedan para training
for i in names_faces_uniq:
    face_position = np.where(names_faces == i)[0]
    pos_for_test = np.random.choice(face_position, 3, replace=False)
    pos_for_train = np.setdiff1d(face_position, pos_for_test)
    
    test_faces.append(flat_faces_np[pos_for_test])
    training_faces.append(flat_faces_np[pos_for_train])
    
    test_names.append(names_faces[pos_for_test])
    training_names.append(names_faces[pos_for_train])
    
    test_files.append(files_faces[pos_for_test])
    training_files.append(files_faces[pos_for_train])
    
print(test_files)

#6 Pasamos de rango 0-255 a 0-1 por cuestiones numéricas

# Change integers to 32-bit floating point numbers
train_img_array_flatten = train_img_array_flatten.astype('float64')

train_img_array_flatten = train_img_array_flatten/255.0

# Change integers to 32-bit floating point numbers
test_img_array_flatten = test_img_array_flatten.astype('float64')

test_img_array_flatten = test_img_array_flatten/255.0

#7 Se ejecuta PCA y se calcula la cantidad de componentes
# que explican el 90% de variabilidad 

pca = PCA(n_components=.9, svd_solver = 'full')
pca.fit(train_img_array_flatten)

print(f'Nro de componentes a utilizar en el PCA : {pca.n_components_}')

#Algoritmo de Gustavo Denicolay
# 16 Definiciones preliminares del algoritmo de Denicolay para RN con 1 capa oculta
