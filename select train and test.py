import numpy as np
import pandas as pd
from sklearn.decomposition import PCA  
import random
import math
import matplotlib.pyplot as plt
from mlxtend.preprocessing import standardize
from sklearn.preprocessing import LabelEncoder
#from keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt

# Desactivar el modo interactivo
plt.ioff()

#leer el numpy arrange
flat_faces_np = np.loadtxt("./faces_numpy.csv", delimiter=",")

names_faces_np = pd.read_csv("./names_numpy.csv", delimiter=",",header = None)
names_faces = names_faces_np.values

files_faces_np = pd.read_csv("./files_numpy.csv", delimiter=",",header = None)
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
    aux_for_test = np.random.choice(face_position, 4, replace=False) #Seleccionamos 4 fotos para test en vez de 3, para poder hacer el PCA de Test
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
input_size = X.shape[1]   # 188 entradas
hidden_size_1 = 100  # neuronas capa oculta 1
hidden_size_2 = 50  # neuronas capa oculta 2
output_size = Y.shape[1]  # 1 neurona capa final output


# defino las funciones de activacion de cada capa
hidden_FUNC = 'logsig'  # uso la logistica en ambas capa ocultas
output_FUNC = 'logsig'  # uso la logistica en ambas capa ocultas


# Incializo las matrices de pesos azarosamente
# W1 son los pesos que van del input a la capa oculta 1
# W2 son los pesos que van de la capa oculta 1 a la capa oculta 2
# W3 son los pesos que van de la capa oculta 2 a la capa de salida
np.random.seed(1906) #mi querida random seed para que las corridas sean reproducibles
W1 = np.random.uniform(-0.5, 0.5, [hidden_size_1, input_size])
X01 = np.random.uniform(-0.5, 0.5, [hidden_size_1, 1] )
W2 = np.random.uniform(-0.5, 0.5, [hidden_size_2, hidden_size_1])
X02 = np.random.uniform(-0.5, 0.5, [hidden_size_2, 1] )
W3 = np.random.uniform(-0.5, 0.5, [output_size, hidden_size_2])
X03 = np.random.uniform(-0.5, 0.5, [output_size, 1] )


# Avanzo la red, forward
# para TODOS los X al mismo tiempo ! 
#  @ hace el producto de una matrix por un vector_columna
hidden_estimulos_1 = W1 @ X.T + X01
hidden_salidas_1 = func_eval_vec(hidden_FUNC, hidden_estimulos_1)
hidden_estimulos_2 = W2 @ hidden_salidas_1 + X02
hidden_salidas_2 = func_eval_vec(hidden_FUNC, hidden_estimulos_2)
output_estimulos = W3 @ hidden_salidas_2 + X03
output_salidas = func_eval_vec(output_FUNC, output_estimulos)


# calculo el error promedi general de TODOS los X
Error= np.mean( (Y.T - output_salidas)**2 )


# Inicializo
epoch_limit = 100000  # para terminar si no converge
Error_umbral = 1.0e-22
learning_rate = 10.0
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
        output_estimulos = W3 @ hidden_salidas_2 + X03
        output_salidas = func_eval_vec(output_FUNC, output_estimulos)


        # calculo los errores en la capa hidden y la capa output
        ErrorSalida = Y[fila:fila+1,:].T - output_salidas
        # output_delta es un solo numero
        output_delta = ErrorSalida * deriv_eval_vec(output_FUNC, output_salidas)
        # hidden_delta_1 y hidden_delta_2 son vectores columna
        hidden_delta_2 = deriv_eval_vec(hidden_FUNC, hidden_salidas_2)*(W3.T @ output_delta)
        hidden_delta_1 = deriv_eval_vec(hidden_FUNC, hidden_salidas_1)*(W2.T @ hidden_delta_2)




        # ya tengo los errores que comete cada capa
        # corregir matrices de pesos, voy hacia atras
        # backpropagation
        W1 = W1 + learning_rate * (hidden_delta_1 @ X[fila:fila+1, :] )
        X01 = X01 + learning_rate * hidden_delta_1
        W2 = W2 + learning_rate * (hidden_delta_2 @ hidden_salidas_1.T )
        X02 = X02 + learning_rate * hidden_delta_2
        W3 = W3 + learning_rate * (output_delta @ hidden_salidas_2.T)
        X03 = X03 + learning_rate * output_delta


    # ya recalcule las matrices de pesos
    # ahora avanzo la red, feed-forward
    hidden_estimulos_1 = W1 @ X.T + X01
    hidden_salidas_1 = func_eval_vec(hidden_FUNC, hidden_estimulos_1)
    hidden_estimulos_2 = W2 @ hidden_salidas_1 + X02
    hidden_salidas_2 = func_eval_vec(hidden_FUNC, hidden_estimulos_2)
    output_estimulos = W3 @ hidden_salidas_2 + X03
    output_salidas = func_eval_vec(output_FUNC, output_estimulos)


    # calculo el error promedio general de TODOS los X
    Error= np.mean( (Y.T - output_salidas)**2 )
   
# 22 Cálculo de Accuracy para training set de RN de Denicolay con 2 capas ocultas

ierror = (np.argmax(output_salidas, axis=0) - np.array(training_int_num) != 0)

# Cuántos hay
print(epoch)

print('Hay {} errores en el conjunto de training sobre un total de {} imagenes'.format(np.sum(ierror), len(traning_Z)))

print(f'Error medio cuadrático {Error} en training')

print(f'Accuracy {(len(traning_Z) - np.sum(ierror))/len(traning_Z)} en training')

###################################################################################################
# 19 Cálculo de accuracy para testing set de RN de Denicolay de 1 capa oculta

test_pca = PCA(n_components = 63, svd_solver = 'full')
test_pca.fit(test_faces)
test_Z = test_pca.transform(test_faces)[:,3:] 

test_label_encoder = LabelEncoder()
test_label_binarizer = LabelBinarizer()
test_label_encoder.fit(test_names)
test_names_num = test_label_binarizer.fit_transform(test_names)
test_int_num = test_label_encoder.transform(test_names)

entrada = test_Z
salida = test_names_num

#print(salida.shape)

# Paso las listas a numpy
X = np.array(entrada)
#Y = np.array(salida).reshape(len(X),1)
Y = np.array(salida)

#print(Y)

hidden_estimulos_1 = W1 @ X.T + X01
hidden_salidas_1 = func_eval_vec(hidden_FUNC, hidden_estimulos_1)
hidden_estimulos_2 = W2 @ hidden_salidas_1 + X02
hidden_salidas_2 = func_eval_vec(hidden_FUNC, hidden_estimulos_2)
output_estimulos = W3 @ hidden_salidas_2 + X03
output_salidas = func_eval_vec(output_FUNC, output_estimulos)

# calculo el error promedio
Error= np.mean( (Y.T - output_salidas)**2 )
# 20 Detalle de accuracy y error para testing set de RN de Denicolay de 1 capa oculta

ierror = (np.argmax(output_salidas, axis=0) - np.array(test_int_num) != 0)

cont = 0
for i in range(76):
    if ierror[i]:
        cont += 1
        print(f'Error nro {cont}')
        print(f'Valor real: {test_names[i]}')
        print(f'Valor predicho: {test_names[np.argmax(output_salidas, axis=0)[i]]}\n')

print(output_salidas)
print(np.argmax(output_salidas, axis=0))
# Cuántos hay
print('Hay {} errores en el conjunto de testing sobre un total de {} imagenes'.format(np.sum(ierror),
len(test_Z)))

print(f'Error medio cuadrático {Error} en testing')

print(f'Accuracy {(len(test_Z) - np.sum(ierror))/len(test_Z)} en testing')