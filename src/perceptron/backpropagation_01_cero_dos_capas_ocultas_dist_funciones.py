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


def standardize_column(dataframe, column):
    """
    Estandarizar columna

    Parametros
    ----------
    dataframe: pd.DataFrame
        Conjunto de datos
    column: str
        Nombre de la columna a estandarizar
    """
    mean = dataframe[column].mean()
    std = dataframe[column].std()
    return (dataframe[column] - mean) / std


# importo dataset cero
import pandas as pd

# df_cero = pd.read_csv("src/perceptron/cero.txt", delimiter="\t")
df_cero = pd.read_csv("cero.txt", delimiter="\t")

# df_cero.loc[df_cero["y"] == 0, "y"] = -1

df_cero["x1"] = standardize_column(df_cero, "x1")
df_cero["x2"] = standardize_column(df_cero, "x2")

entrada = [[x1, x2] for x1, x2 in zip(df_cero["x1"], df_cero["x2"])]
salida = [y for y in df_cero["y"]]

# Paso las listas a numpy
X = np.array(entrada)
Y = np.array(salida).reshape(len(X), 1)

filas_qty = len(X)
input_size = X.shape[1]  # 2 entradas
hidden_size1 = 8  # neuronas primera capa oculta
hidden_size2 = 2  # neuronas segunda capa oculta #2 #8
output_size = Y.shape[1]  # 1 neurona

# defino las funciones de activacion de cada capa
hidden_FUNC_1 = "logsig"  # uso la logística
hidden_FUNC_2 = "logsig"  # uso la logística
output_FUNC = "tansig"  # uso la tangente hiperbólica

# incializo los graficos
grafico = perceptron_plot(X, np.array(salida), 0.0)

# Incializo las matrices de pesos aleatoriamente
np.random.seed(1021)  # mi querida random seed para que las corridas sean reproducibles
W1 = np.random.uniform(-0.5, 0.5, [hidden_size1, input_size])
X01 = np.random.uniform(-0.5, 0.5, [hidden_size1, 1])
W2 = np.random.uniform(-0.5, 0.5, [hidden_size2, hidden_size1])
X02 = np.random.uniform(-0.5, 0.5, [hidden_size2, 1])
W3 = np.random.uniform(-0.5, 0.5, [output_size, hidden_size2])
X03 = np.random.uniform(-0.5, 0.5, [output_size, 1])

# Avanzo la red, forward
hidden_estimulos1 = W1 @ X.T + X01
hidden_salidas1 = func_eval_vec(hidden_FUNC_1, hidden_estimulos1)
hidden_estimulos2 = W2 @ hidden_salidas1 + X02
hidden_salidas2 = func_eval_vec(hidden_FUNC_2, hidden_estimulos2)
output_estimulos = W3 @ hidden_salidas2 + X03
output_salidas = func_eval_vec(output_FUNC, output_estimulos)

# Calculo el error promedio general de todos los X
Error = np.mean((Y.T - output_salidas) ** 2)

# Inicializo
epoch_limit = 20000  # para terminar si no converge
Error_umbral = 0.00000000001  # 0.00000001
learning_rate = 1  # 0.9
Error_last = 10  # lo debo poner algo dist a 0 la primera vez
epoch = 0

while math.fabs(Error_last - Error) > Error_umbral and (epoch < epoch_limit):
    epoch += 1
    Error_last = Error

    # Recorro siempre TODA la entrada
    for fila in range(filas_qty):
        # Propagar el x hacia adelante
        hidden_estimulos1 = W1 @ X[fila : fila + 1, :].T + X01
        hidden_salidas1 = func_eval_vec(hidden_FUNC_1, hidden_estimulos1)
        hidden_estimulos2 = W2 @ hidden_salidas1 + X02
        hidden_salidas2 = func_eval_vec(hidden_FUNC_2, hidden_estimulos2)
        output_estimulos = W3 @ hidden_salidas2 + X03
        output_salidas = func_eval_vec(output_FUNC, output_estimulos)

        # Calcular los errores en la capa hidden y la capa output
        ErrorSalida = Y[fila : fila + 1, :].T - output_salidas
        output_delta = ErrorSalida * deriv_eval_vec(output_FUNC, output_salidas)
        hidden_delta2 = deriv_eval_vec(hidden_FUNC_2, hidden_salidas2) * (
            W3.T @ output_delta
        )
        hidden_delta1 = deriv_eval_vec(hidden_FUNC_1, hidden_salidas1) * (
            W2.T @ hidden_delta2
        )

        # Corregir matrices de pesos, voy hacia atrás (backpropagation)
        W1 = W1 + learning_rate * (hidden_delta1 @ X[fila : fila + 1, :])
        X01 = X01 + learning_rate * hidden_delta1
        W2 = W2 + learning_rate * (hidden_delta2 @ hidden_salidas1.T)
        X02 = X02 + learning_rate * hidden_delta2
        W3 = W3 + learning_rate * (output_delta @ hidden_salidas2.T)
        X03 = X03 + learning_rate * output_delta

    # Avanzo la red, feed-forward
    hidden_estimulos1 = W1 @ X.T + X01
    hidden_salidas1 = func_eval_vec(hidden_FUNC_1, hidden_estimulos1)
    hidden_estimulos2 = W2 @ hidden_salidas1 + X02
    hidden_salidas2 = func_eval_vec(hidden_FUNC_2, hidden_estimulos2)
    output_estimulos = W3 @ hidden_salidas2 + X03
    output_salidas = func_eval_vec(output_FUNC, output_estimulos)

    # Calcular el error promedio general de TODOS los X
    Error = np.mean((Y.T - output_salidas) ** 2)

    grafico.graficarVarias(W1, X01.T[0], epoch, -1)
    display(f"Error: {Error}")


# predict (para clasificar un nuevo punto luego de ajustar la red!)
x_new = np.array([[-0.4, -0.7]])
hidden_estimulos1 = W1 @ x_new.T + X01
hidden_salidas1 = func_eval_vec(hidden_FUNC_1, hidden_estimulos1)
hidden_estimulos2 = W2 @ hidden_salidas1 + X02
hidden_salidas2 = func_eval_vec(hidden_FUNC_2, hidden_estimulos2)
output_estimulos = W3 @ hidden_salidas2 + X03
output_salidas = func_eval_vec(output_FUNC, output_estimulos)
output_salidas
