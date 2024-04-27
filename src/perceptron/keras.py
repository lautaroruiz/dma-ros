import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf

# Cargar y preprocesar datos
data = pd.read_csv("src/perceptron/cero.txt", delimiter="\t")
X = data[["x1", "x2"]].values
y = data["y"].values

scaler = StandardScaler()
X = scaler.fit_transform(X)
y = y.reshape(-1, 1)  # Reshape y para que sea una matriz columna

# Dividir datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Definir modelo de red neuronal
model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(8, activation="sigmoid", input_shape=(2,)),
        tf.keras.layers.Dense(2, activation="sigmoid"),
        tf.keras.layers.Dense(1, activation="tanh"),
    ]
)

# Compilar el modelo
model.compile(optimizer="sgd", loss="mse")

# Entrenar el modelo
history = model.fit(
    X_train, y_train, epochs=200, validation_data=(X_test, y_test), verbose=0
)

# Evaluar el modelo en el conjunto de prueba
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error on Test Set:", mse)

# Visualizar la convergencia del error durante el entrenamiento
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.legend()
plt.show()
