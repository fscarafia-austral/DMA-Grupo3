import numpy as np
import pandas as pd
from sklearn.decomposition import PCA  
import math
import matplotlib.pyplot as plt

# Desactivar el modo interactivo
plt.ioff()

#leer el numpy arrange
#leer el numpy arrange
training_faces = np.loadtxt("./faces_train.csv", delimiter=",")

training_names = pd.read_csv("./names_train.csv", delimiter=",",header = None)
training_names = training_names.values

training_files = pd.read_csv("./files_train.csv", delimiter=",",header = None)
training_files = training_files.values

# Nombres únicos de personas
names_faces_uniq = np.unique(training_names)

# Change integers to 32-bit floating point numbers
training_faces = np.array(training_faces, dtype = object) 
training_faces = training_faces.astype('float64')

# Pasamos de rango 0-255 a 0-1 por cuestiones numéricas
training_faces = training_faces/255.0

#7 Se ejecuta PCA y se calcula la cantidad de componentes
pca = PCA(n_components = 60, svd_solver = 'full')
pca.fit(training_faces)
traning_Z = pca.transform(training_faces) 

print(f'Nro de componentes a utilizar en el PCA : {pca.n_components_}')