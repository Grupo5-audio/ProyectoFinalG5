import os
import numpy as np
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import classification_report
from models.metrics import metrics_values
#from models.loss import CategoricalFocalLoss # Import the custom loss function

# Define the CategoricalFocalLoss class directly
class CategoricalFocalLoss(tf.keras.losses.Loss):
  def __init__(self, gamma=2.0, alpha=0.25, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE, name='categorical_focal_loss'):
        super().__init__(reduction=reduction, name=name)
        self.gamma = gamma
        self.alpha = alpha

  def call(self, y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        loss = self.alpha * tf.pow(1 - y_pred, self.gamma) * cross_entropy
        return tf.reduce_sum(loss, axis=-1)

  def get_config(self):
        config = super().get_config()
        config.update({
            "gamma": self.gamma,
            "alpha": self.alpha
        })
        return config


def grafico_perdida(history):
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


def run_mlp(
    data_path="src/",
    models_path="models/"
):
    # 📦 Cargar datos
    x_train, y_train, feature_names = joblib.load(os.path.join(data_path, "train.pkl"))
    x_val, y_val, _ = joblib.load(os.path.join(data_path, "val.pkl"))
    x_test, y_test, _ = joblib.load(os.path.join(data_path, "test.pkl"))

    # 🔁 Aplanar entradas si es necesario
    if len(x_train.shape) > 2:
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_val = x_val.reshape(x_val.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)

    # 🏷️ Cargar nombres de clases
    class_labels_path = os.path.join(data_path, "class_labels.npy")
    if os.path.exists(class_labels_path):
        class_names = np.load(class_labels_path, allow_pickle=True).tolist()
        print("✅ Clases cargadas:", class_names)
    else:
        raise FileNotFoundError("❌ No se encontró el archivo class_labels.npy.")

    n_classes = y_train.shape[1]
    input_dim = x_train.shape[1]

    # Convert one-hot encoded y to labels for feature selection
    y_train_labels = np.argmax(y_train, axis=1)
    y_val_labels = np.argmax(y_val, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)


    # ⭐ Aplicar selección de características
    from sklearn.feature_selection import SelectKBest, f_classif
    selector = SelectKBest(f_classif, k=200)
    x_train = selector.fit_transform(x_train, y_train_labels) # Use y_train_labels for fit_transform
    x_val = selector.transform(x_val)
    x_test = selector.transform(x_test)


    # 🧠 Construir MLP
    model = models.Sequential([
        layers.Input(shape=(x_train.shape[1],)), # Update input shape after feature selection
        layers.Dense(512, activation='relu'),
        #layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        #layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu'),
        #layers.Dropout(0.2),
        layers.Dense(n_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss=CategoricalFocalLoss(gamma=2.0),#'categorical_crossentropy',
        metrics=['accuracy']
    )

    # ⏱️ Callbacks
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

      

    # 🚆 Entrenar
    print("🚀 Entrenando MLP...")
    history = model.fit(
        x_train, y_train, # Use original y_train for training
        validation_data=(x_val, y_val), # Use original y_val for validation
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=2
    )

    # 💾 Guardar modelo
    model_path = os.path.join(models_path, "mlp_best.keras")
    model.save(model_path)
    print(f"📦 Modelo MLP guardado en: {model_path}")

    # grafico de perdia

    grafico_perdida(history)
    
    # 🧪 Evaluación en test
    y_pred_probs = model.predict(x_test)
    y_pred_labels = np.argmax(y_pred_probs, axis=1)
    #y_test_labels = np.argmax(y_test, axis=1) # Already created y_test_labels earlier

    print("📈 Evaluación final en conjunto de prueba:")
    metrics_values(y_test_labels, y_pred_labels, class_names)

    return model, x_test, feature_names
