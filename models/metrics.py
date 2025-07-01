from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
    balanced_accuracy_score
)
import seaborn as sns
import matplotlib.pyplot as plt

def metrics_values(y_test_labels, y_pred, class_names):
    # 📊 Métricas de evaluación
    accuracy = accuracy_score(y_test_labels, y_pred)
    precision = precision_score(y_test_labels, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test_labels, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test_labels, y_pred, average='weighted', zero_division=0)
    balanced_acc = balanced_accuracy_score(y_test_labels, y_pred)

    print("\n📊 Métricas de evaluación:")
    print(f"✔️ Accuracy: {accuracy:.4f}")
    print(f"✔️ Precision (weighted): {precision:.4f}")
    print(f"✔️ Recall (weighted): {recall:.4f}")
    print(f"✔️ F1-score (weighted): {f1:.4f}")
    print(f"✔️ Balanced Accuracy: {balanced_acc:.4f}\n")

    # 📋 Reporte por clase
    print("📋 Reporte de Clasificación por clase:")
    print(classification_report(y_test_labels, y_pred, target_names=class_names))

    # 📉 Matriz de confusión
    cm = confusion_matrix(y_test_labels, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Matriz de Confusión")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.show()
