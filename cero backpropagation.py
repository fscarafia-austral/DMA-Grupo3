import math
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.preprocessing import standardize
import pandas as pd

from graficos import perceptron_plot 

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

#Leo el dataset del cero completo y lo separo en entrada y salida
dataframe_cero = pd.read_csv("C:/dma-ros/datasets/cero.txt", delimiter="\t")
entrada = dataframe_cero[['x1','x2']]
salida = dataframe_cero[['y']]

# Paso las listas a numpy
X = np.array(entrada)
Y = np.array(salida).reshape(len(X),1)

filas_qty = len(X)
input_size = X.shape[1]   # 2 entradas
hidden_size_1 = 8  # neuronas capa oculta 1
hidden_size_2 = 2  # neuronas capa oculta 2
output_size = Y.shape[1]  # 1 neurona capa final output

# defino las funciones de activacion de cada capa
hidden_FUNC = 'logsig'  # uso la logistica en ambas capa ocultas
output_FUNC = 'tansig'  # uso la tangente hiperbolica en la output

# incializo los graficos
grafico = perceptron_plot(X, np.array(salida), 0.0)

# Incializo las matrices de pesos azarosamente
# W1 son los pesos que van del input a la capa oculta 1
# W2 son los pesos que van de la capa oculta 1 a la capa oculta 2
# W3 son los pesos que van de la capa oculta 2 a la capa de salida
np.random.seed(201006) #mi querida random seed para que las corridas sean reproducibles
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
epoch_limit = 10000    # para terminar si no converge
Error_umbral = 1.0e-12
learning_rate = 0.2
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
    # tengo que hacer X01.T[0]  para que pase el vector
    grafico.graficarVarias(W1, X01.T[0], epoch, -1)