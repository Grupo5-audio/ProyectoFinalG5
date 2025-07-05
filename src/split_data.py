from sklearn.model_selection import train_test_split
import joblib  # Para guardar los datos

def split_data(X, Y, test_size=0.2, val_size=0.25):
    """
    Divide los datos en conjuntos de entrenamiento, prueba y validación.

    Args:
        X (np.array or pd.DataFrame): Las características de los datos.
        Y (np.array or pd.Series): Las etiquetas de los datos.
        test_size (float): El tamaño del conjunto de prueba (proporción del conjunto original).
        val_size (float): El tamaño del conjunto de validación (proporción del conjunto restante después de la prueba).

    Returns:
        tuple: Contiene X_train, X_test, X_val, Y_train, Y_test, Y_val.
    """
    # Primero dividimos en entrenamiento y el resto (prueba + validación)
    X_train, X_temp, Y_train, Y_temp = train_test_split(
        X, Y, test_size=test_size, random_state=0, stratify=Y
    )

    # Luego dividimos el resto en prueba y validación
    # Calculamos el tamaño del conjunto de validación en proporción al conjunto temporal
    val_size_temp = val_size / (1 - test_size)

    X_test, X_val, Y_test, Y_val = train_test_split(
        X_temp, Y_temp, test_size=val_size_temp, random_state=0, stratify=Y_temp
    )

    print("✅ Datos divididos:")
    print(f"Conjunto de entrenamiento: {X_train.shape} (Características), {Y_train.shape} (Etiquetas)")
    print(f"Conjunto de prueba: {X_test.shape} (Características), {Y_test.shape} (Etiquetas)")
    print(f"Conjunto de validación: {X_val.shape} (Características), {Y_val.shape} (Etiquetas)")

    return X_train, X_test, X_val, Y_train, Y_test, Y_val

# Ejemplo de uso:
# X_train, X_test, X_val, Y_train, Y_test, Y_val = split_data(X, Y)

#Función que verifica el dataset guardado en el pkl
def check_dataset(pkl_path):
    data = joblib.load(pkl_path)
    print(type(data))
