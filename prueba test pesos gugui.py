import numpy as np
import pandas as pd
from sklearn.decomposition import PCA  
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer

#Funciones auxiliares
def func_eval(fname, x):
    match fname:
        case "purelin":
            y = x
        case "logsig":
            y = 1.0 / ( 1.0 + math.exp(-x) )
        case "tansig":
            y = 2.0 / ( 1.0 + math.exp(-2.0*x) ) - 1.0
    return y

func_eval_vec = np.vectorize(func_eval)

def deriv_eval(fname, y):  #atencion que y es la entrada y=f( x )
    match fname:
        case "purelin":
            d = 1.0
        case "logsig":
            d = y*(1.0-y)
        case "tansig":
            d = 1.0 - y*y
    return d

deriv_eval_vec = np.vectorize(deriv_eval)

# Lectura de pesos Gugui
W1 = np.load("./pesos y biases/weight_layer_0.npy")
X01 = np.load("./pesos y biases/bias_layer_0.npy")
W2 = np.load("./pesos y biases/weight_layer_1.npy")
X02 = np.load("./pesos y biases/bias_layer_1.npy")
W3 = np.load("./pesos y biases/weight_layer_2.npy")
X03 = np.load("./pesos y biases/bias_layer_2.npy")
W4 = np.load("./pesos y biases/weight_layer_3.npy")
X04 = np.load("./pesos y biases/bias_layer_3.npy")

# Lectura archivos test
test_faces = np.loadtxt("./csvs/faces_test.csv", delimiter=",")

test_names = pd.read_csv("./csvs/names_test.csv", delimiter=",",header = None)
test_names = test_names.values

test_files = pd.read_csv("./csvs/files_test.csv", delimiter=",",header = None)
test_files = test_files.values

# Nombres únicos de personas
names_faces_uniq = np.unique(test_names)

# Change integers to 32-bit floating point numbers
test_faces = np.array(test_faces, dtype = object) 
test_faces = test_faces.astype('float64')

# Pasamos de rango 0-255 a 0-1 por cuestiones numéricas
test_faces = test_faces/255.0

# PCA
test_pca = PCA(n_components = 60, svd_solver = 'full')
test_pca.fit(test_faces)
test_Z = test_pca.transform(test_faces)

# encoding para backpropagation
test_label_encoder = LabelEncoder()
test_label_binarizer = LabelBinarizer()
test_label_encoder.fit(test_names)
test_names_num = test_label_binarizer.fit_transform(test_names)
test_int_num = test_label_encoder.transform(test_names)

# Preparativos
entrada = test_Z
salida = test_names_num
hidden_FUNC = 'logsig'  # uso la logistica en ambas capa ocultas
output_FUNC = 'logsig'  # uso la logistica en ambas capa ocultas

# Paso las listas a numpy
X = np.array(entrada)
Y = np.array(salida)

# Seteo de capas segun pesos gugui
hidden_estimulos_1 = W1 @ X.T + X01
hidden_salidas_1 = func_eval_vec(hidden_FUNC, hidden_estimulos_1)
hidden_estimulos_2 = W2 @ hidden_salidas_1 + X02
hidden_salidas_2 = func_eval_vec(hidden_FUNC, hidden_estimulos_2)
hidden_estimulos_3 = W3 @ hidden_salidas_2 + X03
hidden_salidas_3 = func_eval_vec(hidden_FUNC, hidden_estimulos_3)
output_estimulos = W4 @ hidden_salidas_3 + X04
output_salidas = func_eval_vec(output_FUNC, output_estimulos)

# calculo el error promedio
Error= np.mean( (Y.T - output_salidas)**2 )
# 20 Detalle de accuracy y error para testing set de RN de Denicolay de 1 capa oculta

predicciones = np.argmax(output_salidas, axis=0)
ierror = (predicciones - np.array(test_int_num) != 0)

cont = 0
for i in range(76):
    if ierror[i]:
        cont += 1
        print(f'Error nro {cont}')
        print(f'Valor real: {test_names[i]}')
        print(f'Valor predicho: {names_faces_uniq[np.argmax(output_salidas, axis=0)[i]]}\n')
    
    if not ierror[i]:
        print('Predicción correcta')
        print(f'Valor real: {test_names[i]}')
        print(f'Valor predicho: {names_faces_uniq[np.argmax(output_salidas, axis=0)[i]]}\n')

print(output_salidas)
print(np.argmax(output_salidas, axis=0))
# Cuántos hay
print('Hay {} errores en el conjunto de testing sobre un total de {} imagenes'.format(np.sum(ierror),
len(test_Z)))

print(f'Error medio cuadrático {Error} en testing')

print(f'Accuracy {(len(test_Z) - np.sum(ierror))/len(test_Z)} en testing')