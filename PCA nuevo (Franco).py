import numpy as np
import pandas as pd
from sklearn.decomposition import PCA  
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer

# Desactivar el modo interactivo
plt.ioff()

#leer el numpy arrange
#leer el numpy arrange
training_faces = np.loadtxt("./csvs/faces_train.csv", delimiter=",")

training_names = pd.read_csv("./csvs/names_train.csv", delimiter=",",header = None)
training_names = training_names.values

training_files = pd.read_csv("./csvs/files_train.csv", delimiter=",",header = None)
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

# print(f'Nro de componentes a utilizar en el PCA : {pca.n_components_}')
# quitamos las primeras 3 componentes
traning_Z = traning_Z[:,3:]

# BACKPROPAGATION

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

label_encoder = LabelEncoder()
label_binarizer = LabelBinarizer()
label_encoder.fit(training_names)
training_names_num = label_binarizer.fit_transform(training_names)
training_int_num = label_encoder.transform(training_names)

#Leo el dataset del cero completo y lo separo en entrada y salida
dataframe = traning_Z
entrada = traning_Z
salida = training_names_num

# Paso las listas a numpy
X = np.array(entrada)
Y = np.array(salida)

filas_qty = len(X)
input_size = X.shape[1]
hidden_size_1 = 50  # neuronas capa oculta 1
hidden_size_2 = 30  # neuronas capa oculta 2
hidden_size_3 = 20  # neuronas capa oculta 3
output_size = Y.shape[1]  # 1 neurona capa final output

# defino las funciones de activacion de cada capa
hidden_FUNC = 'logsig'  # uso la logistica en ambas capa ocultas
output_FUNC = 'logsig'  # uso la logistica en ambas capa ocultas

# Incializo las matrices de pesos azarosamente
# W1 son los pesos que van del input a la capa oculta 1
# W2 son los pesos que van de la capa oculta 1 a la capa oculta 2
# W3 son los pesos que van de la capa oculta 2 a la capa de salida
np.random.seed(1988) #mi querida random seed para que las corridas sean reproducibles
W1 = np.random.uniform(-0.5, 0.5, [hidden_size_1, input_size])
X01 = np.random.uniform(-0.5, 0.5, [hidden_size_1, 1] )
W2 = np.random.uniform(-0.5, 0.5, [hidden_size_2, hidden_size_1])
X02 = np.random.uniform(-0.5, 0.5, [hidden_size_2, 1] )
W3 = np.random.uniform(-0.5, 0.5, [hidden_size_3, hidden_size_2])
X03 = np.random.uniform(-0.5, 0.5, [hidden_size_3, 1] )
W4 = np.random.uniform(-0.5, 0.5, [output_size, hidden_size_3])
X04 = np.random.uniform(-0.5, 0.5, [output_size, 1] )

# Avanzo la red, forward
# para TODOS los X al mismo tiempo ! 
#  @ hace el producto de una matrix por un vector_columna
hidden_estimulos_1 = W1 @ X.T + X01
hidden_salidas_1 = func_eval_vec(hidden_FUNC, hidden_estimulos_1)
hidden_estimulos_2 = W2 @ hidden_salidas_1 + X02
hidden_salidas_2 = func_eval_vec(hidden_FUNC, hidden_estimulos_2)
hidden_estimulos_3 = W3 @ hidden_salidas_2 + X03
hidden_salidas_3 = func_eval_vec(hidden_FUNC, hidden_estimulos_3)
output_estimulos = W4 @ hidden_salidas_3 + X04
output_salidas = func_eval_vec(output_FUNC, output_estimulos)

# calculo el error promedi general de TODOS los X
Error= np.mean( (Y.T - output_salidas)**2 )

# Inicializo
epoch_limit = 1000  # para terminar si no converge
Error_umbral = 1.0e-11
learning_rate = 0.3
Error_last = 10    # lo debo poner algo dist a 0 la primera vez
epoch = 0

while (math.fabs(Error_last-Error)>Error_umbral and (epoch < epoch_limit)):
    epoch += 1
    Error_last = Error

    # recorro siempre TODA la entrada
    for fila in range(filas_qty): #para cada input x_sub_fila del vector X
        # propagar el x hacia adelante
        hidden_estimulos_1 = W1 @ X[fila:fila+1, :].T + X01
        hidden_salidas_1 = func_eval_vec(hidden_FUNC, hidden_estimulos_1)
        hidden_estimulos_2 = W2 @ hidden_salidas_1 + X02
        hidden_salidas_2 = func_eval_vec(hidden_FUNC, hidden_estimulos_2)
        hidden_estimulos_3 = W3 @ hidden_salidas_2 + X03
        hidden_salidas_3 = func_eval_vec(hidden_FUNC, hidden_estimulos_3)
        output_estimulos = W4 @ hidden_salidas_3 + X04
        output_salidas = func_eval_vec(output_FUNC, output_estimulos)

        # calculo los errores en la capa hidden y la capa output
        ErrorSalida = Y[fila:fila+1,:].T - output_salidas
        # output_delta es un solo numero
        output_delta = ErrorSalida * deriv_eval_vec(output_FUNC, output_salidas)
        # hidden_delta_1 y hidden_delta_2 son vectores columna
        hidden_delta_3 = deriv_eval_vec(hidden_FUNC, hidden_salidas_3)*(W4.T @ output_delta)
        hidden_delta_2 = deriv_eval_vec(hidden_FUNC, hidden_salidas_2)*(W3.T @ hidden_delta_3)
        hidden_delta_1 = deriv_eval_vec(hidden_FUNC, hidden_salidas_1)*(W2.T @ hidden_delta_2)

        # ya tengo los errores que comete cada capa
        # corregir matrices de pesos, voy hacia atras
        # backpropagation
        W1 = W1 + learning_rate * (hidden_delta_1 @ X[fila:fila+1, :] )
        X01 = X01 + learning_rate * hidden_delta_1
        W2 = W2 + learning_rate * (hidden_delta_2 @ hidden_salidas_1.T )
        X02 = X02 + learning_rate * hidden_delta_2
        W3 = W3 + learning_rate * (hidden_delta_3 @ hidden_salidas_2.T )
        X03 = X03 + learning_rate * hidden_delta_3
        W4 = W4 + learning_rate * (output_delta @ hidden_salidas_3.T)
        X04 = X04 + learning_rate * output_delta

    # ya recalcule las matrices de pesos
    # ahora avanzo la red, feed-forward
    hidden_estimulos_1 = W1 @ X.T + X01
    hidden_salidas_1 = func_eval_vec(hidden_FUNC, hidden_estimulos_1)
    hidden_estimulos_2 = W2 @ hidden_salidas_1 + X02
    hidden_salidas_2 = func_eval_vec(hidden_FUNC, hidden_estimulos_2)
    hidden_estimulos_3 = W3 @ hidden_salidas_2 + X03
    hidden_salidas_3 = func_eval_vec(hidden_FUNC, hidden_estimulos_3)
    output_estimulos = W4 @ hidden_salidas_3 + X04
    output_salidas = func_eval_vec(output_FUNC, output_estimulos)


    # calculo el error promedio general de TODOS los X
    Error= np.mean( (Y.T - output_salidas)**2 )
    # print(epoch)
    
    # calculo accuracy para cada epoch
    ierror = (np.argmax(output_salidas, axis=0) - np.array(training_int_num) != 0)
    print(f'Accuracy {(len(traning_Z) - np.sum(ierror))/len(traning_Z)} en epoch {epoch}')
   
# 22 Cálculo de Accuracy para training set de RN de Denicolay con 2 capas ocultas

ierror = (np.argmax(output_salidas, axis=0) - np.array(training_int_num) != 0)

# Cuántos hay
print(epoch)

print('Hay {} errores en el conjunto de training sobre un total de {} imagenes'.format(np.sum(ierror), len(traning_Z)))

print(f'Error medio cuadrático {Error} en training')

print(f'Accuracy {(len(traning_Z) - np.sum(ierror))/len(traning_Z)} en training')

###################################################################################################
# 19 Cálculo de accuracy para testing set de RN de Denicolay de 1 capa oculta

# Leer archivos test
test_faces = np.loadtxt("./csvs/faces_test.csv", delimiter=",")

test_names = pd.read_csv("./csvs/names_test.csv", delimiter=",",header = None)
test_names = test_names.values

test_files = pd.read_csv("./csvs/files_test.csv", delimiter=",",header = None)
test_files = test_files.values

# Change integers to 32-bit floating point numbers
test_faces = np.array(test_faces, dtype = object) 
test_faces = test_faces.astype('float64')

# Pasamos de rango 0-255 a 0-1 por cuestiones numéricas
test_faces = test_faces/255.0

# PCA
test_pca = PCA(n_components = 60, svd_solver = 'full')
test_pca.fit(test_faces)
test_Z = test_pca.transform(test_faces)

# quitamos las primeras 3 componentes
test_Z = test_Z[:,3:]

test_label_encoder = LabelEncoder()
test_label_binarizer = LabelBinarizer()
test_label_encoder.fit(test_names)
test_names_num = test_label_binarizer.fit_transform(test_names)
test_int_num = test_label_encoder.transform(test_names)

entrada = test_Z
salida = test_names_num

# Paso las listas a numpy
X = np.array(entrada)
#Y = np.array(salida).reshape(len(X),1)
Y = np.array(salida)

#print(Y)

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