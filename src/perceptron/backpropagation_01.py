# backpropagation, just one hidden layer
# lo hago con  matrices de pesos
# puedo tener tantos inputs como quiera
# puedo tener tantas neuronas ocultas como quiera
# puedo tener tanas neuronas de salida como quiera
# fuera de este codigo esta la decision que tomo segun el valor de salida de cada neurona de salida

import math
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.preprocessing import standardize

from graficos import perceptron_plot


def func_eval(fname, x):
    match fname:
        case "purelin":
            y = x
        case "logsig":
            y = 1.0 / (1.0 + math.exp(-x))
        case "tansig":
            y = 2.0 / (1.0 + math.exp(-2.0 * x)) - 1.0
    return y


func_eval_vec = np.vectorize(func_eval)


def deriv_eval(fname, y):  # atencion que y es la entrada y=f( x )
    match fname:
        case "purelin":
            d = 1.0
        case "logsig":
            d = y * (1.0 - y)
        case "tansig":
            d = 1.0 - y * y
    return d


deriv_eval_vec = np.vectorize(deriv_eval)


# Ejemplo a resolver  el simplisimo  XOR
entrada = [[-1, -1], [-1, 1], [1, -1], [1, 1]]
salida = [-1, 1, 1, -1]

# Paso las listas a numpy
X = np.array(entrada)
Y = np.array(salida).reshape(len(X), 1)

filas_qty = len(X)
input_size = X.shape[1]  # 2 entradas
hidden_size = 2  # neuronas capa oculta
output_size = Y.shape[1]  # 1 neurona

# defino las funciones de activacion de cada capa
hidden_FUNC = "logsig"  # uso la logistica
output_FUNC = "tansig"  # uso la tangente hiperbolica

# incializo los graficos
grafico = perceptron_plot(X, np.array(salida), 0.0)


# Incializo las matrices de pesos azarosamente
# W1 son los pesos que van del input a la capa oculta
# W2 son los pesos que van de la capa oculta a la capa de salida
np.random.seed(1021)  # mi querida random seed para que las corridas sean reproducibles
W1 = np.random.uniform(-0.5, 0.5, [hidden_size, input_size])
X01 = np.random.uniform(-0.5, 0.5, [hidden_size, 1])
W2 = np.random.uniform(-0.5, 0.5, [output_size, hidden_size])
X02 = np.random.uniform(-0.5, 0.5, [output_size, 1])

# Avanzo la red, forward
# para TODOS los X al mismo tiempo !
#  @ hace el producto de una matrix por un vector_columna
hidden_estimulos = W1 @ X.T + X01
hidden_salidas = func_eval_vec(hidden_FUNC, hidden_estimulos)
output_estimulos = W2 @ hidden_salidas + X02
output_salidas = func_eval_vec(output_FUNC, output_estimulos)

# calculo el error promedi general de TODOS los X
Error = np.mean((Y.T - output_salidas) ** 2)

# Inicializo
epoch_limit = 2000  # para terminar si no converge
Error_umbral = 0.001
learning_rate = 0.8
Error_last = 10  # lo debo poner algo dist a 0 la primera vez
epoch = 0

while math.fabs(Error_last - Error) > Error_umbral and (epoch < epoch_limit):
    epoch += 1
    Error_last = Error

    # recorro siempre TODA la entrada
    for fila in range(filas_qty):  # para cada input x_sub_fila del vector X
        # propagar el x hacia adelante
        hidden_estimulos = W1 @ X[fila : fila + 1, :].T + X01
        hidden_salidas = func_eval_vec(hidden_FUNC, hidden_estimulos)
        output_estimulos = W2 @ hidden_salidas + X02
        output_salidas = func_eval_vec(output_FUNC, output_estimulos)

        # calculo los errores en la capa hidden y la capa output
        ErrorSalida = Y[fila : fila + 1, :].T - output_salidas
        # output_delta es un solo numero
        output_delta = ErrorSalida * deriv_eval_vec(output_FUNC, output_salidas)
        # hidden_delta es un vector columna
        hidden_delta = deriv_eval_vec(hidden_FUNC, hidden_salidas) * (
            W2.T @ output_delta
        )

        # ya tengo los errores que comete cada capa
        # corregir matrices de pesos, voy hacia atras
        # backpropagation
        W1 = W1 + learning_rate * (hidden_delta @ X[fila : fila + 1, :])
        X01 = X01 + learning_rate * hidden_delta
        W2 = W2 + learning_rate * (output_delta @ hidden_salidas.T)
        X02 = X02 + learning_rate * output_delta

    # ya recalcule las matrices de pesos
    # ahora avanzo la red, feed-forward
    hidden_estimulos = W1 @ X.T + X01
    hidden_salidas = func_eval_vec(hidden_FUNC, hidden_estimulos)
    output_estimulos = W2 @ hidden_salidas + X02
    output_salidas = func_eval_vec(output_FUNC, output_estimulos)

    # calculo el error promedio general de TODOS los X
    Error = np.mean((Y.T - output_salidas) ** 2)

    # tengo que hacer X01.T[0]  para que pase el vector
    grafico.graficarVarias(W1, X01.T[0], epoch, -1)
