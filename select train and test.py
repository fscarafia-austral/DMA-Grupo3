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

pos_for_test = []
pos_for_train = []

# Nombres únicos de personas
names_faces_uniq = np.unique(names_faces)

#Semilla aleatoria
np.random.seed(1957)

# Loop para seleccionar 3 fotos de cada uno p/el test y el resto quedan para training
for i in names_faces_uniq:
    face_position = np.where(names_faces == i)[0]
    aux_for_test = np.random.choice(face_position, 3, replace=False)
    aux_for_train = np.setdiff1d(face_position, aux_for_test)
    pos_for_test.append(aux_for_test)
    pos_for_train.append(aux_for_train)

pos_for_test = np.concatenate(pos_for_test)  
pos_for_train = np.concatenate(pos_for_train) 
    
test_faces = flat_faces_np[pos_for_test]
training_faces = flat_faces_np[pos_for_train]  
    
test_names = names_faces[pos_for_test]
training_names = names_faces[pos_for_train]
    
test_files = files_faces[pos_for_test]
training_files = files_faces[pos_for_train]
    
print(test_files)

#6 Pasamos de rango 0-255 a 0-1 por cuestiones numéricas

# Change integers to 32-bit floating point numbers

training_faces = np.array(training_faces, dtype = object) 
training_faces = training_faces.astype('float64')

training_faces = training_faces/255.0

# Change integers to 32-bit floating point numbers

test_faces = np.array(test_faces)
test_faces = test_faces.astype('float64')

test_faces = test_faces/255.0

#7 Se ejecuta PCA y se calcula la cantidad de componentes
# que explican el 90% de variabilidad 

pca = PCA(n_components = 63, svd_solver = 'full')
pca.fit(training_faces)
traning_Z = pca.transform(training_faces)[:,3:] 

print(f'Nro de componentes a utilizar en el PCA : {pca.n_components_}')

#Algoritmo de Gustavo Denicolay
# 16 Definiciones preliminares del algoritmo de Denicolay para RN con 1 capa oculta
##
# backpropagation, just one hidden layer
# lo hago con  matrices de pesos
# puedo tener tantos inputs como quiera
# puedo tener tantas neuronas ocultas como quiera
# puedo tener tantas neuronas de salida como quiera
# fuera de este codigo esta la decision que tomo segun el valor de salida de cada neurona de salida

