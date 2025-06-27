import joblib
import numpy as np
import os
import matplotlib.pyplot as plt
from models.metrics import metrics_values
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import ReduceLROnPlateau

def run_rn(
    data_path="src/",
    models_path="models/"
):
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

    # 🚀 Entrenar modelo SVM (kernel RBF por defecto)
    model = Sequential()
    model.add(Input(shape=(x_train.shape[1], 1)))  # Mantener la forma de entrada
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(6, activation='softmax'))  # Ajustar según el número de clases
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=(x_test, y_test), callbacks=[ReduceLROnPlateau()])

    # Gráfico de rendimiento del modelo durante el entrenamiento
    plt.figure(figsize=(6, 3))
    plt.plot(history.history['accuracy'], label='accuracy en entrenamiento')
    plt.plot(history.history['val_accuracy'], label='accuracy de validación')
    plt.xlabel('épocas')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Rendimiento del modelo durante el entrenamiento')
    plt.grid()
    plt.show()

    # 💾 Guardar modelo entrenado
    model_path = os.path.join(models_path, "rn_model.pkl")
    joblib.dump(model, model_path)
    print(f"📦 Modelo Red Neuronal guardado en: {model_path}")

    # 🔍 Hacer predicciones
    y_pred = model.predict(x_test)

    # Convertir a etiquetas
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)

    metrics_values(y_test_labels, y_pred_labels, class_names)
    
    return model, x_test, feature_names
