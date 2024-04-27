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


# importo dataset cero
import pandas as pd

# df_cero = pd.read_csv("src/perceptron/cero.txt", delimiter="\t")
df_cero = pd.read_csv("cero.txt", delimiter="\t")

df_cero["x1"] = standardize_column(df_cero, "x1")
df_cero["x2"] = standardize_column(df_cero, "x2")

entrada = [[x1, x2] for x1, x2 in zip(df_cero["x1"], df_cero["x2"])]
salida = [y for y in df_cero["y"]]

# Convert lists to numpy arrays
X = np.array(entrada)
Y = np.array(salida).reshape(len(X), 1)

filas_qty = len(X)
input_size = X.shape[1]  # Number of input features
hidden_size = 8  # Neurons in first hidden layer
hidden_size2 = 2  # Neurons in second hidden layer
output_size = Y.shape[1]  # Number of output neurons

# Activation functions for layers
hidden_FUNC = "logsig"
output_FUNC = "tansig"

# Initialize plot
grafico = perceptron_plot(X, np.array(salida), 0.0)

# Initialize weight matrices with random values
np.random.seed(1021)
W1 = np.random.uniform(-0.5, 0.5, [hidden_size, input_size])
X01 = np.random.uniform(-0.5, 0.5, [hidden_size, 1])
W21 = np.random.uniform(-0.5, 0.5, [hidden_size2, hidden_size])
X021 = np.random.uniform(-0.5, 0.5, [hidden_size2, 1])
W2 = np.random.uniform(-0.5, 0.5, [output_size, hidden_size2])
X02 = np.random.uniform(-0.5, 0.5, [output_size, 1])

# Forward pass (propagation)
hidden_estimulos = W1 @ X.T + X01
hidden_salidas = func_eval_vec(hidden_FUNC, hidden_estimulos)

hidden2_estimulos = W21 @ hidden_salidas + X021
hidden2_salidas = func_eval_vec(hidden_FUNC, hidden2_estimulos)

output_estimulos = W2 @ hidden2_salidas + X02
output_salidas = func_eval_vec(output_FUNC, output_estimulos)

# Calculate overall mean squared error
Error = np.mean((Y.T - output_salidas) ** 2)

# Training parameters
epoch_limit = 2000
Error_umbral = 1.0e-04
learning_rate = 0.8
Error_last = 10  # Initialize with a non-zero value

epoch = 0
while math.fabs(Error_last - Error) > Error_umbral and (epoch < epoch_limit):
    epoch += 1
    Error_last = Error

    # Backpropagation for all training examples
    for fila in range(filas_qty):
        # Forward pass for a single example
        hidden_estimulos = W1 @ X[fila : fila + 1, :].T + X01
        hidden_salidas = func_eval_vec(hidden_FUNC, hidden_estimulos)

        hidden2_estimulos = W21 @ hidden_salidas + X021

        hidden2_salidas = func_eval_vec(hidden_FUNC, hidden2_estimulos)

        output_estimulos = W2 @ hidden2_salidas + X02
        output_salidas = func_eval_vec(output_FUNC, output_estimulos)

        # Calculate errors
        ErrorSalida = Y[fila : fila + 1, :].T - output_salidas
        output_delta = ErrorSalida * deriv_eval_vec(output_FUNC, output_salidas)

        # Error in second hidden layer
        hidden2_delta = deriv_eval_vec(hidden_FUNC, hidden2_salidas) * (
            W2.T @ output_delta
        )

        # **Correction:** Reshape hidden_salidas for compatible multiplication
        hidden_delta = deriv_eval_vec(hidden_FUNC, hidden_salidas.reshape(-1, 1)) * (
            W21.T @ hidden2_delta
        )

        # Update weight matrices
        W1 = W1 + learning_rate * (hidden_delta @ X[fila : fila + 1, :])
        X01 = X01 + learning_rate * hidden_delta

        W21 = W21 + learning_rate * (hidden2_delta @ hidden_salidas.T)
        X021 = X021 + learning_rate * hidden2_delta

        W2 = W2 + learning_rate * (output_delta @ hidden2_salidas.T)
        X02 = X02 + learning_rate * output_delta

    # Forward pass after weight updates (optional for visualization)
    hidden_estimulos = W1 @ X.T + X01
    hidden_salidas = func_eval_vec(hidden_FUNC, hidden_estimulos)

    hidden2_estimulos = W21 @ hidden_salidas + X021
    hidden2_salidas = func_eval_vec(hidden_FUNC, hidden2_estimulos)

    output_estimulos = W2 @ hidden2_salidas + X02
    output_salidas = func_eval_vec(output_FUNC, output_estimulos)

    # Calculate overall mean squared error again
    Error = np.mean((Y.T - output_salidas) ** 2)

    # Update plot (optional for visualization)
    grafico.graficarVarias(W1, X01.T[0], epoch, -1)
    grafico.graficarVarias(W21, X021.T[0], epoch, -2)

# Print final results (optional)
print("Final error:", Error)

plt
