from models.metrics import metrics_values
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import os
import numpy as np
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def run_random_forest(
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

    # 🔄 Decodificar etiquetas One-Hot
    y_train_labels = np.argmax(y_train, axis=1)
    y_val_labels = np.argmax(y_val, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)

    # 🧪 Combinar entrenamiento y validación para GridSearch
    x_combined = np.concatenate((x_train, x_val), axis=0)
    y_combined = np.concatenate((y_train_labels, y_val_labels), axis=0)

    # 🛠️ Definir parámetros a buscar
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'class_weight': ['balanced']
    }

    print("🔍 Buscando mejores hiperparámetros con GridSearchCV...")
    grid = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

    grid.fit(x_combined, y_combined)
    print(f"✅ Mejores parámetros encontrados: {grid.best_params_}")

    best_rf = grid.best_estimator_

    # 💾 Guardar modelo
    model_path = os.path.join(models_path, "random_forest_best.pkl")
    joblib.dump(best_rf, model_path)
    print(f"📦 Modelo optimizado guardado en: {model_path}")

    # 📈 Evaluación en test
    y_test_pred = best_rf.predict(x_test)
    print("📈 Evaluación final en conjunto de prueba:")
    metrics_values(y_test_labels, y_test_pred, class_names)

    return best_rf, x_test, feature_names

