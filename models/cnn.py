# Importación de librerías y dataset
import kagglehub
import pandas as pd
import numpy as np
import os
import sys
import librosa
import seaborn as sns
import matplotlib.pyplot as plt

# to play the audio files
from IPython.display import Audio

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten
from tensorflow.keras.callbacks import ReduceLROnPlateau

import keras
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.regularizers import l2

import joblib

def ejecutar_modelo_cnn(data_path="src/",
                   models_path="models/"):
  # 📦 Cargar datos de entrenamiento y prueba
  x_train, y_train, feature_names = joblib.load(os.path.join(data_path, "train.pkl"))
  x_test, y_test, feature_names = joblib.load(os.path.join(data_path, "test.pkl"))

  # 🔁 Aplanar entradas si están expandidas (para audio)
  if len(x_train.shape) > 2:
      x_train = x_train.reshape(x_train.shape[0], -1)
      x_test = x_test.reshape(x_test.shape[0], -1)

  # 🏷️ Cargar nombres de clases (emociones)
  class_labels_path = os.path.join(data_path, "class_labels.npy")
  if os.path.exists(class_labels_path):
      class_names = np.load(class_labels_path, allow_pickle=True)
  else:
      raise FileNotFoundError("❌ No se encontró el archivo class_labels.npy con los nombres de las emociones.")

  # 🔄 Decodificar etiquetas One-Hot a índices
  y_train_labels = np.argmax(y_train, axis=1)
  y_test_labels = np.argmax(y_test, axis=1)

  print("Forma de x_train:", x_train.shape)

  if len(x_train.shape) == 2:
    x_train = np.expand_dims(x_train, axis=2)

  input_shape = (x_train.shape[1], x_train.shape[2])

  # Parámetro de regularización L2
  l2_lambda = 0.001

  # Definir modelo con L2
  model = Sequential([
    Input(shape=input_shape),
    Conv1D(128, kernel_size=5, padding='same', activation='relu',
        kernel_regularizer=l2(l2_lambda)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),
    Conv1D(64, kernel_size=5, padding='same', activation='relu',
        kernel_regularizer=l2(l2_lambda)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),
    Conv1D(32, kernel_size=3, padding='same', activation='relu',
        kernel_regularizer=l2(l2_lambda)),
    BatchNormalization(),
    Flatten(),
    Dense(64, activation='relu', kernel_regularizer=l2(l2_lambda)),
    Dropout(0.3),
    Dense(6, activation='softmax')  # 6 clases
    ])

    # Compilación del modelo
  model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
  
  # Callbacks para control de sobreajuste
  early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
  reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5)

  model.summary()

  
  model_cnn = model
  # tf.keras.utils.plot_model(model_cnn, rankdir='LR',show_dtype=True)

# Entrenamiento
  history = model_cnn.fit(
        x_train, y_train,
        epochs=100,
        batch_size=32,
        #validation_split=0.2,
        validation_data=(x_test, y_test),
        callbacks=[early_stop, reduce_lr]
    )

  # 💾 Guardar modelo entrenado
  model_path = os.path.join(models_path, "cnn.pkl")
  joblib.dump(model_cnn, model_path)
  print(f"📦 Modelo guardado en: {model_path}")

  plt.figure(figsize=(12, 4))
  # Pérdida
  plt.subplot(1, 2, 1)
  plt.plot(history.history['loss'], label='Pérdida de Entrenamiento')
  plt.plot(history.history['val_loss'], label='Pérdida de Validación')
  plt.title('Pérdida durante el Entrenamiento')
  plt.xlabel('Épocas')
  plt.ylabel('Pérdida')
  plt.legend()

  # Precisión
  plt.subplot(1, 2, 2)
  plt.plot(history.history['accuracy'], label='Precisión de Entrenamiento')
  plt.plot(history.history['val_accuracy'], label='Precisión de Validación')
  plt.title('Precisión durante el Entrenamiento')
  plt.xlabel('Épocas')
  plt.ylabel('Precisión')
  plt.legend()

  plt.tight_layout()
  plt.show()

  # 6. Evaluación y Comparación del Modelo

  # Predicciones en datos de prueba
  # Cargar las clases guardadas
  class_labels = np.load(os.path.join(data_path, "class_labels.npy"), allow_pickle=True)  

  # Crear un nuevo encoder con esas clases
  encoder = OneHotEncoder(categories=[class_labels], handle_unknown='ignore', sparse_output=False)

  # "Ajustar" el encoder con los nombres de clase
  # Esto es necesario para que sklearn lo considere "fitted"
  encoder.fit(np.array(class_labels).reshape(-1, 1))

  pred_test = model.predict(x_test)
  y_pred = encoder.inverse_transform(pred_test)

  # Crear DataFrame de predicciones
  df = pd.DataFrame(columns=['Predicted Labels', 'Actual Labels'])
  df['Predicted Labels'] = y_pred.flatten()
  df['Actual Labels'] = encoder.inverse_transform(y_test).flatten()
  df.head(10)

  print("📈 Evaluación final en conjunto de prueba:")
  metrics_values(encoder.inverse_transform(y_test), y_pred, class_names)

  return model_cnn, x_test, feature_names

#ejecutar_modelo_cnn()
