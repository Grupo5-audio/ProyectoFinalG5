from sklearn.ensemble import RandomForestClassifier
from models.metrics import metrics_values
import os
import numpy as np
import joblib


def run_random_forest(
    data_path="src/",
    models_path="models/"
):
    # 📦 Cargar datos de entrenamiento y prueba
    x_train, y_train, feature_names = joblib.load(os.path.join(data_path, "train.pkl"))
    x_test, y_test, _ = joblib.load(os.path.join(data_path, "test.pkl"))

    # 🔁 Aplanar entradas si están expandidas (para audio)
    if len(x_train.shape) > 2:
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)

    # 🏷️ Cargar nombres de clases (emociones)
    class_labels_path = os.path.join(data_path, "class_labels.npy")
    if os.path.exists(class_labels_path):
        class_names = np.load(class_labels_path, allow_pickle=True).tolist()  # ✅ Convertir a lista
        print("✅ Clases cargadas desde class_labels.npy:")
        print(class_names)  # 👈 Aquí las ves en consola
    else:
        raise FileNotFoundError("❌ No se encontró el archivo class_labels.npy con los nombres de las emociones.")

    # 🔄 Decodificar etiquetas One-Hot a índices
    y_train_labels = np.argmax(y_train, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)

    # 🚀 Entrenar modelo Random Forest con ajuste por desbalanceo
    clf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced'  # ✅ Ajuste automático al desbalanceo
    )
    clf.fit(x_train, y_train_labels)

    # 💾 Guardar modelo entrenado
    model_path = os.path.join(models_path, "random_forest_model.pkl")
    joblib.dump(clf, model_path)
    print(f"📦 Modelo guardado en: {model_path}")

    # 🔍 Hacer predicciones
    y_pred = clf.predict(x_test)

    # 📊 Evaluar

    print(class_names)  # 👈 Aquí las ves en consola
    metrics_values(y_test_labels, y_pred, class_names)

    return clf, x_test, feature_names
